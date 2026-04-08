# Protenix v2 试运行报告

## 📄 文档

- **[protenix_v2_report.pdf](protenix_v2_report.pdf)** — 完整 PDF 报告（LaTeX 编译）
- **[protenix_v2_report.tex](protenix_v2_report.tex)** — LaTeX 源码

报告内容包括：
- 环境信息与依赖安装
- v1 → v2 的核心升级点（ESM3、Diffusion Scheduler、Equivariance、Config System）
- 遇到的问题及详细解决方案
- CPU 运行脚本说明

## 🔧 CPU 运行脚本

- **[run_forward_cpu.py](run_forward_cpu.py)** — 在无 MLU/CUDA 环境下强制使用 PyTorch CPU 原生算子运行 Protenix v2

### 核心改动

```python
# 1. 环境变量（最早设置）
os.environ["LAYERNORM_TYPE"]  = "openfold"
os.environ["MLU_VISIBLE_DEVICES"] = ""   # 禁用 MLU

# 2. ESM 补丁
class _FakeEsm: ... 
sys.modules["esm"] = _FakeEsm()

# 3. 配置覆盖（强制 PyTorch 原生算子）
base_configs["triangle_multiplicative"] = "torch"
base_configs["triangle_attention"]       = "torch"

# 4. 全程 CPU
DEVICE = torch.device("cpu")
```

### 运行方式

```bash
cd /root/Protenix
python run_forward_cpu.py
```

> ⚠️ 完整推理（200步 diffusion）在 CPU 上非常慢，建议仅用于功能验证和调试。
