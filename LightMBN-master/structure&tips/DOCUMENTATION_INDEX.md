# LightMBN 文档索引

> **一站式项目文档导航** - 快速找到你需要的信息

---

## 📚 文档导航地图

### 🚀 新手入门 (5分钟)
| 文档 | 内容 | 推荐度 |
|------|------|--------|
| [QUICK_REFERENCE.md](#quick-reference) | 30秒项目快速了解 + 常用命令 | ⭐⭐⭐⭐⭐ |
| [PROJECT_STRUCTURE.md](#project-structure) - 快速入门章节 | 5分钟快速入门示例 | ⭐⭐⭐⭐⭐ |

### 📖 理解架构 (30分钟)
| 文档 | 内容 | 推荐度 |
|------|------|--------|
| [PROJECT_STRUCTURE.md](#project-structure) - 工作流程图 | 理解整体架构 | ⭐⭐⭐⭐⭐ |
| [PROJECT_STRUCTURE.md](#project-structure) - 核心模块 | 各模块职责概览 | ⭐⭐⭐⭐⭐ |
| [QUICK_REFERENCE.md](#quick-reference) - 数据流向 | 数据如何流动 | ⭐⭐⭐⭐ |

### 🔬 深度学习 (1小时)
| 文档 | 内容 | 推荐度 |
|------|------|--------|
| [MODULES_DEEP_DIVE.md](#modules-deep-dive) | 每个模块的详细代码解析 | ⭐⭐⭐⭐⭐ |
| [PROJECT_STRUCTURE.md](#project-structure) - 主要特性 | 各种技巧的说明 | ⭐⭐⭐⭐ |

### 🎯 实践操作
| 需求 | 推荐文档 | 章节 |
|------|---------|------|
| 想快速训练一个模型 | QUICK_REFERENCE | 常用命令 |
| 想理解参数含义 | QUICK_REFERENCE | 核心参数速查 |
| 想自定义损失函数 | PROJECT_STRUCTURE | Loss模块详解 |
| 想添加新模型 | MODULES_DEEP_DIVE | Model模块详解 |
| 想优化训练速度 | QUICK_REFERENCE | 性能优化 |
| 出现错误想调试 | QUICK_REFERENCE | 故障排查 |

---

## 🗂️ 按主题的文档位置

### 🏗️ 项目结构相关
```
想了解文件组织？              → PROJECT_STRUCTURE.md > 核心文件结构
想看完整目录树？              → PROJECT_STRUCTURE.md > 文件树关键路径
模块之间怎么调用？            → PROJECT_STRUCTURE.md > 文件调用关系图
```

### 🤖 模型架构相关
```
有哪些模型可用？              → PROJECT_STRUCTURE.md > Model模块
LMBN_n vs LMBN_r 选哪个？     → MODULES_DEEP_DIVE.md > Model深度解析
什么是OSNet？                 → MODULES_DEEP_DIVE.md > OSNet骨干网络
什么是BNNeck？                → MODULES_DEEP_DIVE.md > BNNeck特征处理
想了解注意力机制？            → MODULES_DEEP_DIVE.md > 注意力机制
```

### 💔 损失函数相关
```
支持哪些损失函数？            → PROJECT_STRUCTURE.md > Loss模块
如何组合多个损失？            → QUICK_REFERENCE.md > 损失函数组合
什么是MSLoss？                → MODULES_DEEP_DIVE.md > Multi-Similarity Loss
什么是中心损失？              → MODULES_DEEP_DIVE.md > Center Loss
三元组损失怎么用？            → MODULES_DEEP_DIVE.md > Triplet Loss
```

### 📊 数据相关
```
支持哪些数据集？              → PROJECT_STRUCTURE.md > Data模块 > 支持的数据集
怎么自定义数据集？            → MODULES_DEEP_DIVE.md > 数据管理核心
什么是身份感知采样？          → MODULES_DEEP_DIVE.md > 身份感知采样
数据增强有哪些？              → PROJECT_STRUCTURE.md > 主要特性 > 数据增强
```

### ⚙️ 训练优化相关
```
有哪些优化器？                → QUICK_REFERENCE.md > 核心参数速查
什么是Warmup？                → MODULES_DEEP_DIVE.md > Warmup学习率
什么是余弦退火？              → MODULES_DEEP_DIVE.md > Warmup Cosine Scheduler
学习率怎么设置？              → QUICK_REFERENCE.md > 常用命令 > 基础训练
```

### 📈 评估相关
```
评估有哪些指标？              → PROJECT_STRUCTURE.md > 工作流程
什么是re-ranking？            → MODULES_DEEP_DIVE.md > k-reciprocal重排序
怎么查看结果？                → PROJECT_STRUCTURE.md > Experiment目录
```

### 🔧 实操相关
```
第一次怎么使用？              → QUICK_REFERENCE.md > 常用命令 > 基础训练
想用配置文件？                → QUICK_REFERENCE.md > 常用命令 > 使用配置文件
想跨域测试？                  → QUICK_REFERENCE.md > 常用任务 > 任务4
显存不够怎么办？              → QUICK_REFERENCE.md > 故障排查
训练太慢怎么办？              → QUICK_REFERENCE.md > 性能优化
```

---

## 📋 快速查询表 (一句话速查)

### 参数相关
| 问题 | 答案 | 文档位置 |
|-----|------|---------|
| `--batchid` 是什么？ | 每batch的人数(K) | QUICK_REFERENCE - 批处理参数 |
| `--batchimage` 是什么？ | 每人的图数(N) | QUICK_REFERENCE - 批处理参数 |
| `--margin` 是什么？ | 度量损失边界 | QUICK_REFERENCE - 核心参数速查 |
| `--test_every` 是什么？ | 每N个epoch评估一次 | QUICK_REFERENCE - 核心参数速查 |
| `--feats` 是什么？ | 特征向量维度 | QUICK_REFERENCE - 模型参数 |

### 模型相关
| 问题 | 答案 | 文档位置 |
|-----|------|---------|
| 推荐用哪个模型？ | LMBN_n (轻量级) | MODULES_DEEP_DIVE - LMBN_n vs LMBN_r |
| 参数最少的是？ | OSNet (~2.2M) | MODULES_DEEP_DIVE - OSNet设计目标 |
| 可扩展性最好的是？ | ResNet50 | QUICK_REFERENCE - 模型选择 |

### 损失函数相关
| 问题 | 答案 | 文档位置 |
|-----|------|---------|
| 推荐的组合是？ | 0.5*CE+0.5*MSLoss | QUICK_REFERENCE - 损失函数组合 |
| 哪个收敛最快？ | MSLoss | MODULES_DEEP_DIVE - Multi-Similarity Loss |
| 哪个最稳定？ | CrossEntropy+CenterLoss | MODULES_DEEP_DIVE - Center Loss |

### 数据相关
| 问题 | 答案 | 文档位置 |
|-----|------|---------|
| 推荐batch配置？ | 8×6=48 | QUICK_REFERENCE - 常用命令 |
| 什么是IdentitySampler？ | 身份感知采样器 | MODULES_DEEP_DIVE - 身份感知采样 |
| 默认图像大小？ | 256×128 | QUICK_REFERENCE - 数据参数 |

### 训练相关
| 问题 | 答案 | 文档位置 |
|-----|------|---------|
| 推荐学习率？ | 6e-4 | QUICK_REFERENCE - 常用命令 |
| 推荐epoch数？ | 130 | QUICK_REFERENCE - 常用命令 |
| 用什么优化器？ | ADAM | QUICK_REFERENCE - 常用命令 |
| 用什么学习率策略？ | Warmup+Cosine | MODULES_DEEP_DIVE - Warmup Cosine Scheduler |

### 评估相关
| 问题 | 答案 | 文档位置 |
|-----|------|---------|
| 主要指标是？ | Rank@1 和 mAP | PROJECT_STRUCTURE - 评估指标 |
| re-ranking能提升多少？ | ~3-5% mAP | MODULES_DEEP_DIVE - k-reciprocal重排序 |
| 结果保存在哪？ | experiment/ 目录 | PROJECT_STRUCTURE - Experiment目录 |

---

## 🎓 学习路径建议

### 👶 完全新手 (1-2小时)
```
1. 读 QUICK_REFERENCE.md 前两章 (30分钟)
   ↓ 了解项目基本概念和常用命令
2. 运行基础训练命令 (15分钟)
   python main.py --config lmbn_config.yaml
3. 读 PROJECT_STRUCTURE.md - 工作流程图 (20分钟)
   ↓ 理解代码执行流程
4. 查看训练日志和结果 (15分钟)
   ↓ 了解输出的含义
```

### 👨‍💼 有经验的研究者 (30分钟)
```
1. 快速浏览 PROJECT_STRUCTURE.md (10分钟)
   ↓ 获得全景图
2. 重点阅读感兴趣的模块深度解析 (15分钟)
   ↓ 例如损失函数、采样器
3. 浏览 QUICK_REFERENCE.md - 参数速查 (5分钟)
```

### 🤖 想添加自己的代码 (1-2小时)
```
1. MODULES_DEEP_DIVE.md - 相关模块详解 (30分钟)
   ↓ 理解模块设计
2. 查看相关的源代码 (30分钟)
   ↓ 对标现有实现
3. 编写和测试新代码 (30分钟)
```

---

## 📱 移动设备友好速查

### 最常用的3个命令
```bash
# 1. 训练
python main.py --config lmbn_config.yaml

# 2. 测试预训练模型
python main.py --test_only --pre_train model.pth

# 3. 自定义训练
python main.py --datadir /data --data_train market1501 --model LMBN_n --epochs 130
```

### 最常用的3个参数
```bash
--batchid 8        # 每batch人数
--batchimage 6     # 每人图数
--lr 6e-4          # 学习率
```

### 最常用的3个模块
```
LMBN_n     - 推荐轻量级模型
MSLoss     - 推荐损失函数
CUDA       - 推荐计算设备
```

---

## 🔍 文档搜索快捷方式

### 用 Ctrl+F 搜索关键词

| 想找... | 搜索词 | 推荐文档 |
|--------|--------|---------|
| 某个参数含义 | `--param_name` | QUICK_REFERENCE |
| 某个类的用途 | `class ClassName` | MODULES_DEEP_DIVE |
| 某个文件做什么 | `filename.py` | PROJECT_STRUCTURE |
| 某个概念怎么用 | concept name | 按主题找对应文档 |

---

## 📞 获取帮助

| 遇到问题 | 推荐阅读 |
|--------|---------|
| 命令不知道怎么用 | QUICK_REFERENCE - 常用命令 |
| 参数不知道什么意思 | QUICK_REFERENCE - 核心参数速查 |
| 模型/损失函数怎么选 | PROJECT_STRUCTURE - 推荐配置 |
| 代码怎么改 | MODULES_DEEP_DIVE - 相关模块 |
| 训练不收敛 | QUICK_REFERENCE - 故障排查 |
| 显存不够 | QUICK_REFERENCE - 故障排查 |
| 想要优化性能 | QUICK_REFERENCE - 性能优化 |
| 想要添加功能 | MODULES_DEEP_DIVE - 相关模块 |
| 实验结果怎么查看 | PROJECT_STRUCTURE - Experiment目录 |

---

## 🎯 按使用场景快速导航

### 场景1: "我想快速开始训练"
```
→ QUICK_REFERENCE.md
  ├─ 常用命令 > 基础训练
  └─ 核心参数速查
```

### 场景2: "我想理解代码怎么运行的"
```
→ PROJECT_STRUCTURE.md
  ├─ 工作流程图
  ├─ 文件调用关系图
  └─ 各个模块的入口函数
```

### 场景3: "我想自定义某个部分(如损失函数)"
```
→ MODULES_DEEP_DIVE.md > Loss模块深度解析
  ↓
→ 查看相关源代码
  ↓
→ 参考现有实现进行修改
```

### 场景4: "我想优化训练性能"
```
→ QUICK_REFERENCE.md > 性能优化
  ↓
或 QUICK_REFERENCE.md > 故障排查
```

### 场景5: "我想对比不同配置的效果"
```
→ QUICK_REFERENCE.md > 常见任务 > 任务5
  ↓
→ 按照建议修改损失函数或参数
```

---

## 💾 文档列表

| 文件 | 字数 | 阅读时间 | 难度 | 推荐人群 |
|------|------|---------|------|---------|
| **QUICK_REFERENCE.md** | ~5000 | 20分钟 | ⭐ 简单 | 所有人 |
| **PROJECT_STRUCTURE.md** | ~8000 | 30分钟 | ⭐⭐ 中等 | 想了解架构的人 |
| **MODULES_DEEP_DIVE.md** | ~10000 | 45分钟 | ⭐⭐⭐ 复杂 | 要修改代码的人 |
| **DOCUMENTATION_INDEX.md** | ~3000 | 10分钟 | ⭐ 简单 | 找资料的人 |

---

## ☑️ 目录完整性检查清单

### 必看内容 ✅
- [x] 项目名称和目标
- [x] 快速开始指南
- [x] 文件和模块说明
- [x] 参数详解
- [x] 常见命令示例

### 可选但重要 ✅
- [x] 深度技术解析
- [x] 性能优化建议
- [x] 故障排查指南
- [x] 文档导航索引

### 相关参考 ⏳
- [ ] 论文细节讨论
- [ ] 代码注释详解
- [ ] 贡献指南

---

## 🚨 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2025-02-07 | 初始版本：3份核心文档 |

---

## 📧 反馈和改进

如果有以下情况，欢迎反馈:
- 文档有错误或不清楚
- 缺少某个重要内容
- 有更好的解释方式
- 发现新的常见问题

---

**最后更新**: 2025-02-07 | **维护者**: LightMBN文档小组

**快速导航**:
- 🚀 [新手入门](#新手入门-5分钟)
- 📖 [理解架构](#理解架构-30分钟)
- 🔬 [深度学习](#深度学习-1小时)
- 🎯 [实践操作](#实践操作)

---

**💡 建议**: 将这个索引保存为书签，方便快速查阅！
