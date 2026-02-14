import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock
from .attention import BatchFeatureErase_Top, OcclusionAwareAttention
from .bnneck import BNNeck3
from torch.nn import functional as F

class LMBN_Global_SA(nn.Module):
    def __init__(self, args):
        super(LMBN_Global_SA, self).__init__()

        # 1. Backbone
        osnet = osnet_x1_0(pretrained=True)
        self.backone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0]
        )
        conv3 = osnet.conv3[1:]
        
        # 2. 全局分支卷积层
        self.global_branch = nn.Sequential(
            copy.deepcopy(conv3), 
            copy.deepcopy(osnet.conv4), 
            copy.deepcopy(osnet.conv5)
        )

        # 3. 两个处理模块
        self.batch_drop_block = BatchFeatureErase_Top(512, OSBlock, h_ratio=args.h_ratio)
        self.sa_module = OcclusionAwareAttention(512)
        
        # 池化层
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # 4. 核心：两个独立的 BNNeck (Reduction) 模块
        # reduction_avg: 处理普通平均池化路径
        self.reduction_avg = BNNeck3(512, args.num_classes, args.feats, return_f=True)
        # reduction_att: 处理 Self-Attention 后的路径
        self.reduction_att = BNNeck3(512, args.num_classes, args.feats, return_f=True)

    def forward(self, x):
        # Backbone 提取
        x = self.backone(x)
        glo_feat_map = self.global_branch(x) # [B, 512, H, W]

        # --- 路径 A: 纯平均池化 ---
        feat_avg_pooled = self.avg_pooling(glo_feat_map)
        # f_avg 预期结构: [feat_before_bn, logits, feat_after_bn]
        f_avg = self.reduction_avg(feat_avg_pooled)

        # --- 路径 B: 遮挡融合 + SA ---
        # 获取遮挡后的特征 (drop) 和 原始特征 (clean)
        glo_drop, glo_clean = self.batch_drop_block(glo_feat_map, bottleneck_features=True)
        # SA 融合：Q=全局, K/V=遮挡
        feat_sa_map = self.sa_module(x_global=glo_clean, x_occluded=glo_drop)
        feat_sa_pooled = self.avg_pooling(feat_sa_map)
        # f_att 预期结构: [feat_before_bn, logits, feat_after_bn]
        f_att = self.reduction_att(feat_sa_pooled)

        # --- 特征合并 (用于测试和度量学习) ---
        # 提取 BN 后的特征进行拼接 (通常是列表的最后一个元素)
        # 在 lmbn_n.py 的逻辑中，f_xxx[-1] 是用于计算距离的最终特征
        combined_feat = torch.cat([f_avg[-1], f_att[-1]], dim=1)

        if not self.training:
            # 测试时返回合并后的总特征
            return combined_feat

        # 训练阶段
        # 1. Logits 列表：交给 CrossEntropyLoss，会对两个分支分别算分类损失
        logits_list = [f_avg[1], f_att[1]]
        
        # 2. Features 列表：交给 TripletLoss
        # 包含两个独立分支特征，以及合并后的总特征（可选，看你实验需求）
        feats_list = [f_avg[-1], f_att[-1], combined_feat]
        
        return logits_list, feats_list