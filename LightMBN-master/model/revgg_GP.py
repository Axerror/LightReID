import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock
from .attention import BatchDrop, BatchFeatureErase_Top, SE_Module # 这里可以根据需要保留
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

# --- 新增：Coordinate Attention 模块定义 ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

class LMBN_n(nn.Module):
    def __init__(self, args):
        super(LMBN_n, self).__init__()

        # OSNet 预训练骨干网络
        osnet = osnet_x1_0(pretrained=True)

        self.backone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0]
        )

        conv3 = osnet.conv3[1:]

        # 方向一改进：将原有的三分支简化为双分支，并引入 Coordinate Attention
        # 1. 全局分支 (Global Branch) + 注意力增强
        self.global_branch = nn.Sequential(
            copy.deepcopy(conv3), 
            copy.deepcopy(osnet.conv4), 
            copy.deepcopy(osnet.conv5)
        )
        self.ca_attention = CoordAtt(512, 512) # 引入 CA 注意力增强特征

        # 2. 局部分支 (Partial Branch)
        self.partial_branch = nn.Sequential(
            copy.deepcopy(conv3), 
            copy.deepcopy(osnet.conv4), 
            copy.deepcopy(osnet.conv5)
        )

        # 删除了 channel_branch 及其相关组件，以实现模型轻量化

        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))

        # 特征降维层 (Reduction)
        self.reduction_g = nn.Sequential(
            nn.Linear(512, args.feats),
            nn.BatchNorm1d(args.feats),
            nn.ReLU(inplace=True)
        )
        
        self.reduction_p = nn.Sequential(
            nn.Linear(512, args.feats),
            nn.BatchNorm1d(args.feats),
            nn.ReLU(inplace=True)
        )

        # 分类器 (Classifiers)
        self.classifier_g = nn.Linear(args.feats, args.num_classes)
        self.classifier_p0 = nn.Linear(args.feats, args.num_classes)
        self.classifier_p1 = nn.Linear(args.feats, args.num_classes)

        if args.drop_block:
            self.drop = BatchFeatureErase_Top()
        else:
            self.drop = nn.Sequential()

    def forward(self, x):
        # 基础特征提取
        x = self.backone(x)

        # --- 全局分支计算 ---
        g_feat_map = self.global_branch(x)
        g_feat_map = self.ca_attention(g_feat_map) # 关键改进：嵌入坐标注意力
        g_feat = self.global_pooling(g_feat_map)
        g_feat = g_feat.view(g_feat.size(0), -1)
        f_g = self.reduction_g(g_feat)
        l_g = self.classifier_g(f_g)

        # --- 局部分支计算 ---
        p_feat_map = self.partial_branch(x)
        p_feat_map = self.drop(p_feat_map) # 保持原有的特征擦除以增强鲁棒性
        p_feat_p = self.partial_pooling(p_feat_map)
        
        # 将特征切分为两部分 (Part 0 和 Part 1)
        p_feat0, p_feat1 = torch.split(p_feat_p, 1, dim=2)
        p_feat0 = p_feat0.view(p_feat0.size(0), -1)
        p_feat1 = p_feat1.view(p_feat1.size(0), -1)
        
        f_p0 = self.reduction_p(p_feat0)
        f_p1 = self.reduction_p(p_feat1)
        
        l_p0 = self.classifier_p0(f_p0)
        l_p1 = self.classifier_p1(f_p1)

        # 测试模式：拼接全局特征和局部特征
        if not self.training:
            return torch.cat([f_g, f_p0, f_p1], dim=1)

        # 训练模式：返回预测概率列表和特征列表
        # 注意：因为减少了分支，返回值长度变短，请确保训练引擎（Engine）能自适应处理
        return [l_g, l_p0, l_p1], [f_g, f_p0, f_p1]

def make_model(args):
    return LMBN_n(args)