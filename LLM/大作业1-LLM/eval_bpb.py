#!/usr/bin/env python3
"""
eval_bpb.py — 助教统一评测脚本

计算学生模型在私有评测集上的 loss 和 Bits Per Byte (BPB)。

用法：
    python eval_bpb.py \
        --checkpoint checkpoints/best.pt \
        --config     config.json \
        --tokenizer  tokenizer.json \
        --model_file model.py \
        --eval_data  data/TinyStories-valid.txt \
        --output     results/result.json

依赖：
    pip install torch numpy tokenizers
"""

import os
import sys
import json
import math
import argparse
import importlib.util
import traceback
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


# ============================================================
# 工具：动态加载学生的 model.py，返回 (GPT, module)
# ============================================================
def load_student_module(model_py_path):
    """从学生的 model.py 里加载模块，返回 (GPT, module)。"""
    if not os.path.exists(model_py_path):
        raise FileNotFoundError(f"model.py 不存在: {model_py_path}")

    spec = importlib.util.spec_from_file_location("student_model", model_py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(model_py_path)))
    spec.loader.exec_module(mod)

    if not hasattr(mod, "GPT"):
        raise AttributeError("model.py 中未找到 GPT 类")
    return mod.GPT, mod


# ============================================================
# 工具：构造模型实例，兼容两种 GPT.__init__ 签名
#   - GPT(config)                      （模板风格）
#   - GPT(block_size=..., n_layer=...) （散列参数风格）
# ============================================================
def build_model(GPT, mod, config_dict):
    import inspect

    sig = inspect.signature(GPT.__init__)
    param_names = [p.name for p in sig.parameters.values() if p.name != "self"]

    # --- 情况 1：GPT.__init__(self, config) ---
    if param_names == ["config"]:
        # 优先用学生自己定义的 GPTConfig 类
        if hasattr(mod, "GPTConfig"):
            gpt_config_cls = getattr(mod, "GPTConfig")
            cfg_sig = inspect.signature(gpt_config_cls.__init__)
            cfg_params = [p.name for p in cfg_sig.parameters.values()
                          if p.name != "self"]
            cfg_kwargs = {k: config_dict[k] for k in cfg_params if k in config_dict}
            print(f"[INFO] 用学生 GPTConfig 构造: {cfg_kwargs}")
            config_obj = gpt_config_cls(**cfg_kwargs)
        else:
            # 兜底：把 config_dict 直接当属性用
            print(f"[INFO] 学生未定义 GPTConfig，使用 SimpleNamespace 兜底")
            config_obj = SimpleNamespace(**config_dict)

        return GPT(config_obj), config_dict

    # --- 情况 2：GPT.__init__(self, block_size, n_layer, ...) ---
    model_kwargs = {k: config_dict[k] for k in param_names if k in config_dict}
    missing = [k for k in param_names if k not in config_dict]
    if missing:
        raise ValueError(f"config.json 缺少 GPT.__init__ 所需的参数: {missing}")

    print(f"[INFO] 用散列参数构造: {model_kwargs}")
    return GPT(**model_kwargs), config_dict


# ============================================================
# 工具：加载学生模型
# ============================================================
def load_student_model(model_py_path, config_path, checkpoint_path, device):
    GPT, mod = load_student_module(model_py_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model, config = build_model(GPT, mod, config)
    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 参数量: {n_params / 1e6:.2f}M")
    return model, config


# ============================================================
# 工具：读取评测集
# ============================================================
def load_eval_stories(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "<|endoftext|>" in content:
        stories = [s.strip() for s in content.split("<|endoftext|>") if s.strip()]
    else:
        stories = [line.strip() for line in content.splitlines() if line.strip()]
    return stories


# ============================================================
# 核心：计算 loss 和 BPB
# ============================================================
@torch.no_grad()
def compute_loss_and_bpb(model, tokenizer, stories, block_size,
                         eos_id, device, amp_dtype=torch.float16):
    all_ids = []
    total_bytes = 0
    for story in stories:
        ids = tokenizer.encode(story).ids
        all_ids.extend(ids)
        all_ids.append(eos_id)
        total_bytes += len(story.encode("utf-8"))

    total_tokens = len(all_ids)
    print(f"[INFO] 故事数: {len(stories)}")
    print(f"[INFO] 总 token 数: {total_tokens}")
    print(f"[INFO] 原始文本 UTF-8 字节数: {total_bytes}")

    if total_tokens <= block_size + 1:
        raise ValueError("评测集太小，无法切出一个完整 block")

    ids_tensor = torch.tensor(all_ids, dtype=torch.long)
    n_blocks = (len(ids_tensor) - 1) // block_size

    total_nats = 0.0
    counted_tokens = 0

    for i in range(n_blocks):
        start = i * block_size
        chunk = ids_tensor[start : start + block_size + 1]
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)

        with torch.cuda.amp.autocast(dtype=amp_dtype,
                                     enabled=(device.type == "cuda")):
            logits, _ = model(x)

        logits_flat = logits.view(-1, logits.size(-1)).float()
        y_flat = y.view(-1)
        loss_sum = F.cross_entropy(logits_flat, y_flat, reduction="sum")

        total_nats += loss_sum.item()
        counted_tokens += y_flat.numel()

        if (i + 1) % 50 == 0:
            cur_loss = total_nats / counted_tokens
            cur_bpb = total_nats / (total_bytes * math.log(2))
            print(f"  block {i+1}/{n_blocks} | "
                  f"avg_loss={cur_loss:.4f} | BPB={cur_bpb:.4f}")

    avg_loss = total_nats / counted_tokens
    bpb = total_nats / (total_bytes * math.log(2))
    ppl = math.exp(avg_loss)

    return {
        "total_nats": total_nats,
        "total_tokens": counted_tokens,
        "total_bytes": total_bytes,
        "avg_loss": avg_loss,
        "perplexity": ppl,
        "bpb": bpb,
        "n_blocks": n_blocks,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", default="fp16",
                        choices=["fp16", "fp32"])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    result = {
        "checkpoint": args.checkpoint,
        "eval_data": args.eval_data,
    }

    try:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 使用设备: {device}")

        amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.float32

        tokenizer = Tokenizer.from_file(args.tokenizer)
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_id is None:
            eos_id = tokenizer.get_vocab_size() - 1
            print(f"[WARN] 未找到 <|endoftext|>，使用兜底 eos_id={eos_id}")
        else:
            print(f"[INFO] eos_id = {eos_id}")

        model, config = load_student_model(
            args.model_file, args.config, args.checkpoint, device
        )
        block_size = config.get("block_size", 512)

        stories = load_eval_stories(args.eval_data)
        print(f"[INFO] 评测集加载完成，共 {len(stories)} 条故事")

        metrics = compute_loss_and_bpb(
            model, tokenizer, stories, block_size, eos_id,
            device, amp_dtype=amp_dtype
        )

        result.update(metrics)
        result["status"] = "ok"

        print("\n===== 评测结果 =====")
        print(f"Avg Loss (nats/token): {metrics['avg_loss']:.4f}")
        print(f"Perplexity:            {metrics['perplexity']:.4f}")
        print(f"BPB:                   {metrics['bpb']:.4f}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"\n[ERROR] {e}")
        print(result["traceback"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 结果已写入 {args.output}")


if __name__ == "__main__":
    main()

# python eval_bpb.py \
#     --checkpoint checkpoints/best.pt \
#     --config config.json \
#     --tokenizer  tokenizer.json \
#     --model_file model.py \
#     --eval_data  data/TinyStories-valid.txt \
#     --output     results/result.json

# python eval_bpb.py --checkpoint checkpoints/best.pt --config config.json --tokenizer  tokenizer.json --model_file model.py --eval_data  data/TinyStories-valid.txt --output results/result.json