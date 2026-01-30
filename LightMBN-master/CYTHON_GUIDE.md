# Cython 编译指南

## 📋 当前设置

你的项目已配置为支持 Cython 算法优化。编译流程已自动集成到 `main.py` 启动时执行。

## 🔧 修改和重新编译流程

### 场景1：修改算法逻辑后编译

1. **修改源码**
   ```
   e:\Code\LightMBN-master\LightMBN-master\utils\rank_cylib\rank_cy.pyx
   ```

2. **完整重编译**（推荐）
   ```powershell
   python compile_cython.py --clean
   ```

3. **或直接运行训练**（自动编译）
   ```powershell
   python main.py --config lmbn_config.yaml --save ''
   ```

### 场景2：仅快速重编译（保留旧产物）
```powershell
python compile_cython.py
```

### 场景3：清理编译产物
```powershell
python compile_cython.py --clean
```

## ⚙️ 编译配置文件

**位置**: `utils/rank_cylib/setup.py`

关键配置项：
- `sources`: 指定 `.pyx` 源文件位置
- `language`: 编译语言（C/C++）
- `include_dirs`: NumPy 头文件目录
- `extra_compile_args`: 编译器参数

如需优化编译（例如 OpenMP 并行化），修改此文件：

```python
extra_compile_args=['-fopenmp'],    # 启用 OpenMP
extra_link_args=['-fopenmp'],
```

## 📊 编译依赖

```
✅ Cython==3.0.10
✅ numpy>=1.22.4
✅ Microsoft Visual C++ 14.0+（Windows 编译必需）
```

缺失 C++ 工具？下载安装：
https://visualstudio.microsoft.com/visual-cpp-build-tools/

## ❌ 故障排除

### 问题：编译失败 "ImportError: No module named 'Cython'"
```powershell
pip install cython
```

### 问题：编译失败 "Microsoft Visual C++ ... is required"
安装 C++ Build Tools（见上方链接）

### 问题：Python 版本不匹配
`.pyd` 文件名中的 `cp39` 表示 Python 3.9。如果你用 Python 3.10/3.11，需要重编：
```powershell
python compile_cython.py --clean
```

### 问题：修改后仍使用旧代码
必须清理旧产物再编译：
```powershell
python compile_cython.py --clean
```

## 🚀 最佳实践

1. **修改 `.pyx` 后，总是用 `--clean` 选项**
   ```powershell
   python compile_cython.py --clean
   ```

2. **编译成功后再启动训练**
   ```powershell
   python main.py --config lmbn_config.yaml --save ''
   ```

3. **如果不确定是否编译成功，检查日志**
   - 看输出中是否有 `[SUCCESS] Cython module compiled successfully!`
   - 看是否有 `[WARNING] Cython compilation failed`

## 📝 修改示例

假设要修改排序算法：

1. 打开 `utils/rank_cylib/rank_cy.pyx`
2. 修改 `eval_market1501_cy()` 函数逻辑
3. 保存文件
4. 运行编译：
   ```powershell
   python compile_cython.py --clean
   ```
5. 启动训练（自动使用新编译的模块）

## 💡 提示

- 编译通常需要 1-2 分钟
- 编译产物 `.pyd` 文件只能在当前 Python 版本运行
- 如果切换 Python 版本，需要重新编译
- 编译时会生成 `build/` 目录，可以安全删除
