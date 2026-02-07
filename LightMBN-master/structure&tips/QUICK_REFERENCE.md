# LightMBN 快速参考指南

## 📋 项目概览速查表

| 方面 | 说明 |
|------|------|
| **项目名** | Lightweight Multi-Branch Network for Person Re-ID |
| **任务** | 人物重识别 (Person Re-Identification) |
| **框架** | PyTorch |
| **主入口** | `main.py` |
| **核心引擎** | `engine_v3.py` |
| **训练设备** | GPU (CUDA) / CPU |

---

## 🎯 各模块速查

### 1️⃣ 数据流向
```
数据集(Market/Duke/CUHK03)
    ↓
data_v2.ImageDataManager (加载数据)
    ↓
sampler.IdentitySampler (身份采样: 8人×6图)
    ↓
transforms (数据增强: 翻转、随机擦除等)
    ↓
训练批次输入模型
```

### 2️⃣ 模型选择
```
LMBN_n (推荐🌟)
├─ 骨干: OSNet (轻量级)
├─ 参数少
└─ 效果好

LMBN_r
├─ 骨干: ResNet50
├─ 参数多
└─ 效果略低

其他: ResNet50, OSNet, PCB, MGN等
```

### 3️⃣ 损失函数组合
```
常用组合:
• 0.5*CrossEntropy + 0.5*MSLoss     (官方推荐)
• 0.7*Triplet + 0.3*CenterLoss
• 1.0*CrossEntropy + 0.1*FocalLoss

格式: --loss "权重1*损失1+权重2*损失2+..."
```

### 4️⃣ 学习率策略
```
Warmup (预热)
    ↓ (开始从低lr升至目标lr)
最高学习率 (lr = 6e-4)
    ↓ (Cosine衰减)
Epoch 130时最低
```

### 5️⃣ 评估指标
```
Rank@1:  前1排序结果命中率 (%)
Rank@5:  前5排序结果命中率 (%)
mAP:     所有查询的平均精度 (%)

示例结果 (Market-1501):
LightMBN: Rank@1=96.3% mAP=91.5%
+ re-rank: Rank@1=96.8% mAP=95.3% (提升明显)
```

---

## 🔧 常用命令

### 基础训练
```bash
python main.py \
  --datadir /path/to/datasets \
  --data_train market1501 \
  --model LMBN_n \
  --epochs 130 \
  --lr 6e-4 \
  --loss "0.5*CrossEntropy+0.5*MSLoss"
```

### 使用配置文件 (推荐)
```bash
python main.py --config lmbn_config.yaml
```

### 测试预训练模型
```bash
python main.py --test_only --pre_train model.pth
```

### 跨域测试 (在A数据集训练，B数据集测试)
```bash
python main.py \
  --data_train market1501 \
  --data_test dukemtmc \
  --model LMBN_n
```

---

## 📁 文件树关键路径

```
LightMBN-master/
│
├── 📄 main.py                    ← 程序入口
├── 📄 option.py                  ← 参数定义
├── 📄 engine_v3.py               ← 训练循环
│
├── 📁 model/                     ← 神经网络模型
│   ├── __init__.py              (make_model工厂)
│   ├── lmbn_n.py                (LMBN with OSNet ⭐)
│   ├── lmbn_r.py                (LMBN with ResNet50)
│   ├── osnet.py                 (OSNet骨干)
│   ├── resnet50.py              (ResNet50骨干)
│   ├── attention.py             (注意力模块)
│   └── bnneck.py                (BNNeck特征处理)
│
├── 📁 loss/                      ← 损失函数
│   ├── __init__.py              (LossFunction组合器)
│   ├── triplet.py               (三元组损失)
│   ├── multi_similarity_loss.py  (MS损失 ⭐)
│   ├── focal_loss.py            (焦点损失)
│   ├── center_loss.py           (中心损失)
│   └── grouploss.py             (组损失)
│
├── 📁 data_v2/                   ← 数据管理 (当前版本)
│   ├── __init__.py              (ImageDataManager)
│   ├── datamanager.py           (数据集加载)
│   ├── sampler.py               (身份采样器 ⭐)
│   ├── transforms.py            (数据增强)
│   └── datasets/                (具体数据集实现)
│
├── 📁 optim/                     ← 优化器和调度器
│   ├── __init__.py              (make_optimizer/scheduler)
│   ├── warmup_scheduler.py      (Warmup学习率)
│   └── warmup_cosine_scheduler.py (Warmup+Cosine衰减 ⭐)
│
├── 📁 utils/                     ← 工具函数
│   ├── functions.py             (评估: Rank@k, mAP ⭐)
│   ├── re_ranking.py            (k-reciprocal重排序)
│   ├── random_erasing.py        (数据增强)
│   ├── model_complexity.py      (FLOP计算)
│   ├── utility.py               (Checkpoint管理)
│   ├── visualize_rank.py        (排序可视化)
│   └── rank_cylib/              (加速库)
│
├── 📁 experiment/               ← 实验结果输出
│   ├── epoch100/
│   │   ├── log.txt              (训练日志)
│   │   ├── model/
│   │   │   ├── model-latest.pth (最新模型)
│   │   │   └── model-best.pth   (最佳模型 ⭐)
│   │   └── results.json         (评估结果)
│
└── 📁 wandb/                     ← W&B云端日志
```

---

## ⚙️ 核心参数速查

### 数据参数
| 参数 | 含义 | 示例 |
|------|------|------|
| `--datadir` | 数据集根目录 | `/path/to/datasets` |
| `--data_train` | 训练数据集名 | `market1501`, `dukemtmc` |
| `--data_test` | 测试数据集名 | `market1501`, `dukemtmc` |
| `--height` | 输入图像高 | `256` |
| `--width` | 输入图像宽 | `128` |

### 批处理参数
| 参数 | 含义 | 示例 |
|------|------|------|
| `--batchid` | 每batch的人数 | `8` |
| `--batchimage` | 每人的图数 | `6` |
| `--batchtest` | 测试batch大小 | `32` |
| 总batch大小 | = batchid × batchimage | 8×6=48 |

### 模型参数
| 参数 | 含义 | 示例 |
|------|------|------|
| `--model` | 模型架构 | `LMBN_n`, `LMBN_r` |
| `--feats` | 特征维度 | `512` |
| `--num_classes` | 类别数 | 自动设置 |

### 训练参数
| 参数 | 含义 | 示例 |
|------|------|------|
| `--epochs` | 训练轮数 | `130` |
| `--lr` | 学习率 | `6e-4` |
| `--optimizer` | 优化器 | `ADAM`, `SGD` |
| `--loss` | 损失组合 | `"0.5*CE+0.5*MSLoss"` |
| `--margin` | 度量损失边界 | `0.7` |
| `--if_labelsmooth` | 标签平滑 | 布尔值 |
| `--random_erasing` | 随机擦除增强 | 布尔值 |

### 評估参数
| 参数 | 含义 | 示例 |
|------|------|------|
| `--test_only` | 仅测试模式 | 布尔值 |
| `--test_every` | 每N个epoch评估 | `20` |
| `--re_rank` | 使用re-ranking | 布尔值 |

### 输出参数
| 参数 | 含义 | 示例 |
|------|------|------|
| `--save` | 结果保存名 | `'experiment_01'` |
| `--nGPU` | GPU数量 | `1`, `2`, `4` |

---

## 🔍 常见任务清单

### ✅ 任务1: 标准训练
```bash
python main.py \
  --config lmbn_config.yaml \
  --save my_experiment
```
**输出**: `experiment/my_experiment/` 内含模型和日志

---

### ✅ 任务2: 快速验证 (小数据集)
```bash
python main.py \
  --config bag_of_tricks_config.yaml \
  --epochs 10 \
  --test_every 2
```
**输出**: 2小时内完成快速验证

---

### ✅ 任务3: 测试预训练模型
```bash
python main.py \
  --test_only \
  --config lmbn_config.yaml \
  --pre_train pretrained_model.pth
```
**输出**: 测试集评估结果

---

### ✅ 任务4: 跨数据集泛化性测试
```bash
python main.py \
  --data_train market1501 \
  --data_test dukemtmc \
  --model LMBN_n \
  --epochs 130 \
  --save cross_domain_test
```
**输出**: DukeMTMC上的泛化性能

---

### ✅ 任务5: 多损失函数组合
```bash
python main.py \
  --config lmbn_config.yaml \
  --loss "0.4*CrossEntropy+0.4*MSLoss+0.2*CenterLoss" \
  --save multi_loss_exp
```
**输出**: 多目标学习的结果对比

---

## 📊 实验结果示例

### Market-1501数据集上的基准结果

| 模型 | Rank@1 | mAP | +re-rank | 备注 |
|------|--------|-----|---------|------|
| LightMBN (OSNet) | 96.3% | 91.5% | 96.8% / 95.3% | ⭐推荐 |
| LightMBN (ResNet50) | 96.1% | 90.5% | - | 参数更多 |
| BoT (Bag of Tricks) | 94.2% | 85.4% | - | 基础方法 |
| PCB | 95.1% | 86.3% | - | 部件学习 |
| MGN | 94.7% | 87.5% | - | 多粒度网络 |

**关键发现**:
1. LightMBN在轻量级模型中表现最佳
2. re-ranking显著提升mAP (4%+)
3. MSLoss相比Triplet有明显改进

---

## 🚀 优化技巧

### 性能优化
1. **启用混合精度训练**: `--use_amp` (更快更省显存)
2. **增加batch size**: `--batchid 16 --batchimage 8` (更好收敛)
3. **启用re-ranking**: `--re_rank` (评估时激活)

### 准确性优化
1. **组合多个损失**: `--loss "0.5*CE+0.3*MSLoss+0.2*CenterLoss"`
2. **启用标签平滑**: `--if_labelsmooth`
3. **启用随机擦除**: `--random_erasing`
4. **增加epoch数**: `--epochs 200` (收益递减)

### 过拟合控制
1. **减小学习率**: `--lr 3e-4`
2. **增加Dropout**: 模型中调整
3. **数据增强**: `--random_erasing`
4. **降低batch size**: 增加梯度噪声

---

## 🐛 故障排查

### 问题1: CUDA内存不足
**解决**:
```bash
# 减小batch size
python main.py --batchid 4 --batchimage 6

# 或减小图像尺寸
python main.py --height 224 --width 112
```

### 问题2: 模型精度不提高
**检查**:
- ✓ 数据集路径是否正确
- ✓ 学习率是否过高/过低
- ✓ 损失函数是否合适
- ✓ 训练数据是否存在问题

### 问题3: 数据加载错误
**解决**:
```bash
# 检查数据集格式
ls /path/to/datasets/Market-1501/
# 应看到: bounding_box_train, bounding_box_test, query

# 检查权限
chmod -R 755 /path/to/datasets/
```

---

## 💡 最佳实践

### 1. 配置管理
```yaml
# lmbn_config.yaml 是官方推荐配置
# 修改参数时复制为 my_config.yaml，保留原文件作为参考
python main.py --config my_config.yaml
```

### 2. 实验记录
```bash
# 使用 --save 为每个实验命名
python main.py --save exp_001_baseline
python main.py --save exp_002_with_labelsmooth
python main.py --save exp_003_multi_loss

# 结果自动保存到 experiment/exp_00X/
```

### 3. 多卡训练
```bash
# N卡并行
python main.py --nGPU 4 --batchid 32  # 总batch=32×6=192

# 注意: 分布式训练需额外配置
```

### 4. 代码版本控制
```bash
# 记录精确的运行命令
echo "python main.py --config v1.yaml" >> experiment/log.txt

# 或使用W&B自动记录
python main.py --wandb --wandb_name LightMBN_Exp
```

---

## 📚 相关资源

- **Paper**: https://arxiv.org/abs/2101.10774
- **GitHub**: https://github.com/jixunbo/LightMBN
- **数据集下载**: https://github.com/jixunbo/ReIDataset
- **Bag of Tricks**: http://openaccess.thecvf.com/content_CVPRW_2019/...
- **OSNet论文**: https://arxiv.org/abs/1905.00953

---

## 📝 笔记模板

在 `experiment/` 中为每个重要实验创建 `README.md`:

```markdown
# 实验名: Exp_001_Baseline

## 配置
- 模型: LMBN_n
- 数据集: Market-1501
- 损失: 0.5*CE + 0.5*MSLoss
- 学习率: 6e-4
- Epoch: 130

## 结果
- Rank@1: 96.3% | mAP: 91.5%
- +re-rank Rank@1: 96.8% | mAP: 95.3%

## 耗时
- 训练时间: ~2小时 (单GPU)
- 评估时间: ~5分钟

## 关键发现
- ...

## 改进方向
- ...
```

---

**文档版本**: v1.0 | **更新日期**: 2025年2月
