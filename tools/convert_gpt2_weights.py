"""
GPT-2 权重转换脚本
将 OpenAI 发布的 GPT-2 权重转换为本项目的二进制格式

使用方法:
    pip install tensorflow numpy
    python convert_gpt2_weights.py --model_size 124M --output_dir ./gpt2-weights

输出文件:
    - gpt2-{size}.bin: 二进制权重文件（可被 Gpt2WeightLoader.cs 加载）
    - encoder.json: GPT-2 词表
    - vocab.bpe: BPE merge 规则
    - hparams.json: 模型超参数
"""

import argparse
import json
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np


def download_file(url, dest, backup_url=None):
    """下载文件，支持备用 URL"""
    if os.path.exists(dest):
        print(f"  已存在: {dest}")
        return
    print(f"  下载: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        if backup_url:
            print(f"  主 URL 失败，尝试备用: {backup_url}")
            urllib.request.urlretrieve(backup_url, dest)
        else:
            raise


def download_gpt2_files(model_size, models_dir):
    """下载 GPT-2 模型文件"""
    base_url = f"https://openaipublic.blob.core.windows.net/gpt-2/models/{model_size}"
    backup_url = f"https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2/{model_size}"

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta",
        "vocab.bpe"
    ]

    for fn in filenames:
        download_file(
            f"{base_url}/{fn}",
            os.path.join(model_dir, fn),
            f"{backup_url}/{fn}"
        )

    return model_dir


def load_gpt2_params(ckpt_path, n_layers):
    """从 TensorFlow checkpoint 加载参数"""
    import tensorflow as tf

    params = {"blocks": [{} for _ in range(n_layers)]}

    for name, _ in tf.train.list_variables(ckpt_path):
        arr = np.squeeze(tf.train.load_variable(ckpt_path, name))
        parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        target = params
        if parts[0].startswith("h"):
            layer_idx = int(parts[0][1:])
            target = params["blocks"][layer_idx]
            parts = parts[1:]

        for key in parts[:-1]:
            target = target.setdefault(key, {})
        target[parts[-1]] = arr

    return params


def save_binary_weights(params, output_path, hparams):
    """
    将权重保存为简单二进制格式:
    [int32: param_count]
    for each param:
        [int32: name_len][utf8: name]
        [int32: ndim][int32 * ndim: shape]
        [float32 * size: data]
    """
    emb_dim = hparams["n_embd"]
    n_layers = hparams["n_layer"]

    # 收集所有命名参数
    named_params = []

    # Token + Position embeddings
    named_params.append(("tok_emb.Weight", params["wte"]))
    named_params.append(("pos_emb.Weight", params["wpe"]))

    # Transformer blocks
    for i in range(n_layers):
        blk = params["blocks"][i]
        prefix = f"trf_blocks.layers.{i}"

        # Attention: GPT-2 合并存储 QKV 为 [emb, 3*emb]，需要拆分
        c_attn_w = blk["attn"]["c_attn"]["w"]  # [emb, 3*emb]
        c_attn_b = blk["attn"]["c_attn"]["b"]  # [3*emb]

        # 拆分 Q, K, V
        wq, wk, wv = np.split(c_attn_w, 3, axis=-1)  # 各 [emb, emb]
        bq, bk, bv = np.split(c_attn_b, 3, axis=-1)   # 各 [emb]

        # 注意: 我们的 Linear 存储 weight 为 [out, in]，GPT-2 存储为 [in, out]
        # 所以需要转置
        named_params.append((f"{prefix}.attn._wQuery.Weight", wq.T))
        named_params.append((f"{prefix}.attn._wQuery.Bias", bq))
        named_params.append((f"{prefix}.attn._wKey.Weight", wk.T))
        named_params.append((f"{prefix}.attn._wKey.Bias", bk))
        named_params.append((f"{prefix}.attn._wValue.Weight", wv.T))
        named_params.append((f"{prefix}.attn._wValue.Bias", bv))

        # Output projection
        named_params.append((f"{prefix}.attn._outProj.Weight",
                           blk["attn"]["c_proj"]["w"].T))
        named_params.append((f"{prefix}.attn._outProj.Bias",
                           blk["attn"]["c_proj"]["b"]))

        # Layer norms
        named_params.append((f"{prefix}._norm1.Scale", blk["ln_1"]["g"]))
        named_params.append((f"{prefix}._norm1.Shift", blk["ln_1"]["b"]))
        named_params.append((f"{prefix}._norm2.Scale", blk["ln_2"]["g"]))
        named_params.append((f"{prefix}._norm2.Shift", blk["ln_2"]["b"]))

        # Feed-forward
        named_params.append((f"{prefix}._ff._layers.layers.0.Weight",
                           blk["mlp"]["c_fc"]["w"].T))
        named_params.append((f"{prefix}._ff._layers.layers.0.Bias",
                           blk["mlp"]["c_fc"]["b"]))
        named_params.append((f"{prefix}._ff._layers.layers.2.Weight",
                           blk["mlp"]["c_proj"]["w"].T))
        named_params.append((f"{prefix}._ff._layers.layers.2.Bias",
                           blk["mlp"]["c_proj"]["b"]))

    # Final layer norm
    named_params.append(("final_norm.Scale", params["ln_f"]["g"]))
    named_params.append(("final_norm.Shift", params["ln_f"]["b"]))

    # Output head: GPT-2 ties weights with tok_emb (wte)
    named_params.append(("out_head.Weight", params["wte"]))  # 共享权重

    # 写入二进制文件
    with open(output_path, "wb") as f:
        f.write(struct.pack("<i", len(named_params)))

        for name, arr in named_params:
            arr = arr.astype(np.float32)
            name_bytes = name.encode("utf-8")

            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<i", len(arr.shape)))
            for dim in arr.shape:
                f.write(struct.pack("<i", dim))
            f.write(arr.tobytes())

    total_params = sum(np.prod(arr.shape) for _, arr in named_params)
    print(f"保存 {len(named_params)} 个参数张量，共 {total_params:,} 个 float32")


def main():
    parser = argparse.ArgumentParser(description="下载并转换 GPT-2 权重")
    parser.add_argument("--model_size", default="124M",
                       choices=["124M", "355M", "774M", "1558M"])
    parser.add_argument("--output_dir", default="./gpt2-weights")
    parser.add_argument("--models_dir", default="./gpt2-download")
    args = parser.parse_args()

    print(f"=== 下载 GPT-2 {args.model_size} ===")
    model_dir = download_gpt2_files(args.model_size, args.models_dir)

    print(f"\n=== 加载超参数 ===")
    with open(os.path.join(model_dir, "hparams.json")) as f:
        hparams = json.load(f)
    print(f"  emb_dim={hparams['n_embd']}, layers={hparams['n_layer']}, heads={hparams['n_head']}")

    print(f"\n=== 加载 TF checkpoint ===")
    ckpt_path = os.path.join(model_dir, "model.ckpt")
    params = load_gpt2_params(ckpt_path, hparams["n_layer"])

    print(f"\n=== 导出二进制权重 ===")
    os.makedirs(args.output_dir, exist_ok=True)
    bin_path = os.path.join(args.output_dir, f"gpt2-{args.model_size}.bin")
    save_binary_weights(params, bin_path, hparams)
    print(f"  输出: {bin_path}")

    # 复制 tokenizer 文件
    import shutil
    for fn in ["encoder.json", "vocab.bpe", "hparams.json"]:
        src = os.path.join(model_dir, fn)
        dst = os.path.join(args.output_dir, fn)
        shutil.copy2(src, dst)
        print(f"  复制: {dst}")

    print(f"\n=== 完成！===")
    print(f"在 C# 中使用:")
    print(f'  var loader = Gpt2WeightLoader.Load("{bin_path}", model);')


if __name__ == "__main__":
    main()
