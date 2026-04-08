#!/usr/bin/env python3
"""
Protenix-v2 CPU Forward 验证脚本
=================================
验证模型在 CPU 上可完成一次完整 forward pass。
完整推理（200步 diffusion）建议用 MLU/GPU，CPU 太慢。

关键改动：
  LAYERNORM_TYPE=openfold  绕过 CUDA LayerNorm 扩展
  triangle_* = "torch"     绕过 cuequivariance_ops_torch (CUDA-only)
  device = cpu              全程跑在 CPU 上
"""

import sys, os, json, tempfile, shutil

# ★ 必须最早设置（在任何 import 之前）
os.environ["LAYERNORM_TYPE"]  = "openfold"
os.environ["MLU_VISIBLE_DEVICES"] = ""   # 禁用 MLU

sys.path.insert(0, "/root/Protenix")

import torch
from collections.abc import Mapping
from contextlib import nullcontext

# ── Patch ESM（避免导入错误）─────────────────────────────────────────────────
class _FakeEsm:
    class FastaBatchedDataset:
        pass
    @staticmethod
    def pretrained(*args, **kwargs):
        class _M:
            @staticmethod
            def eval():
                pass
        return _M()
sys.modules["esm"] = _FakeEsm()

# ── 配置 ─────────────────────────────────────────────────────────────────────
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from protenix.model.protenix import Protenix

model_name = "protenix-v2"
base_configs = {
    **configs_base,
    **({"data": data_configs} if data_configs else {}),
    **inference_configs,
}

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

deep_update(base_configs, model_configs[model_name])

# ★ 关键覆盖（绕过 CUDA-only 算子）
base_configs["triangle_multiplicative"] = "torch"
base_configs["triangle_attention"]       = "torch"

configs = parse_configs(base_configs, arg_str=None, fill_required_with_null=True)
configs.use_msa      = False
configs.use_template = False
configs.model_name   = model_name

print(f"triangle_multiplicative = {configs.triangle_multiplicative}")
print(f"triangle_attention       = {configs.triangle_attention}")

# ── 设备 ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
DTYPE  = torch.bfloat16

# ── 模型加载 ─────────────────────────────────────────────────────────────────
print("[1] 加载模型（CPU JIT 编译约 5-7 分钟）…")
model = Protenix(configs).to(DEVICE).eval()
ckpt  = torch.load("/root/.cache/protenix/protenix-v2.pt",
                   map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
print(f"[2] 模型就绪，参数量: {sum(p.numel() for p in model.parameters()):,}")

# ── 用官方 featurizer 生成真实输入 ──────────────────────────────────────────
from protenix.data.inference.infer_dataloader import get_inference_dataloader

sample = [{
    "sequences": [{
        "proteinChain": {
            "sequence": "MAEAGGGG",   # 8 个氨基酸
            "count": 1
        }
    }],
    "name": "test_cpu"
}]

tmpdir    = tempfile.mkdtemp()
json_path = os.path.join(tmpdir, "input.json")
with open(json_path, "w") as f:
    json.dump(sample, f)

configs.input_json_path     = json_path
configs.dump_dir            = tmpdir
configs.load_checkpoint_dir = "/root/.cache/protenix"

print("[3] 生成特征输入（官方的 json_to_feature）…")
dataloader = get_inference_dataloader(configs=configs)
result     = next(iter(dataloader))
# DataLoader 返回 list of tuples: [(data, atom_array, error_msg)]
data = result[0][0] if isinstance(result, list) else result[0]
feat = data["input_feature_dict"]
print(f"[4] 特征就绪: msa={feat['msa'].shape}, N_token={feat['msa'].shape[1]}")

# ── Forward ─────────────────────────────────────────────────────────────────
print("[5] Running forward (首次 JIT 编译 + 运行约 3-10 分钟)…")

import time
t0 = time.time()

enable_amp = (
    torch.autocast(device_type="cuda", dtype=DTYPE)
    if torch.cuda.is_available()
    else nullcontext()
)

with enable_amp:
    with torch.no_grad():
        pred, _, _ = model(
            input_feature_dict=feat,
            label_full_dict=None,
            label_dict=None,
            mode="inference",
            mc_dropout_apply_rate=configs.mc_dropout_apply_rate,
        )

t1 = time.time()

# ── 结果 ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("✅ Forward PASSED!")
print(f"  positions  : {pred['positions'].shape}")
print(f"  plDDT      : {pred['plDDT'].shape}")
print(f"  mean_plddt : {pred['mean_plddt'].item():.2f}")
print(f"  耗时       : {t1-t0:.1f}s")
print("=" * 60)

shutil.rmtree(tmpdir)
