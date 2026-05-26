#!/usr/bin/env python3
"""
MobileCLIP2-S2 preprocessing contract + FP16 risk test.

This script does NOT export a new model.

It answers two questions:
  1. Is the current TFLite model consistent with Apple/MobileCLIP2-S2 preprocessing?
     Contract: RGB -> resize/center-crop -> ToTensor -> NCHW float32 in [0,1].
  2. How risky is FP16 for embeddings?
     It tests:
       - FP32 PyTorch reference vs FP32 TFLite
       - FP32 TFLite vs output rounded to FP16 and re-normalized
       - FP32 TFLite vs input rounded to FP16 then fed as float32
       - Retrieval ranking drift after FP16 output rounding

Important:
  Existing mobileclip2_s2_image.tflite input is float32.
  This script does NOT prove a future real FP16-delegate / FP16-weight model is safe.
  It gives a conservative embedding-level signal before doing actual FP16 conversion/delegate testing.

Run:
  TF_ENABLE_ONEDNN_OPTS=0 PYTHONFAULTHANDLER=1 python test_mobileclip2_norm_fp16.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


MODEL_NAME = "MobileCLIP2-S2"
IMAGE_TFLITE = "mobileclip2_s2_image.tflite"
TEXT_TFLITE = "mobileclip2_s2_text.tflite"
OUT_JSON = "mobileclip2_norm_fp16_report.json"


class ImageEncoderRef(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model.encode_image(x)
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class TextEncoderOriginal(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        y = self.model.encode_text(text)
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def make_mobileclip2_s2_preprocess(image_size):
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    # MobileCLIP2-S0/S2/B: identity normalization, not OpenAI CLIP mean/std.
    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),  # [0,1], CHW, float32
    ])


def make_openai_clip_preprocess(image_size):
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    # Wrong for MobileCLIP2-S2, included only as contrast.
    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def load_model():
    print("=" * 90)
    print("Load MobileCLIP2-S2")
    print("=" * 90)

    import open_clip
    from huggingface_hub import hf_hub_download
    from mobileclip.modules.common.mobileone import reparameterize_model

    weight_path = hf_hub_download(
        repo_id=f"apple/{MODEL_NAME}",
        filename=f"{MODEL_NAME.lower().replace('-', '_')}.pt",
        cache_dir="./hf_cache",
    )

    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=weight_path)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    model = reparameterize_model(model)
    model.eval()

    image_size = getattr(model.visual, "image_size", 256)
    if isinstance(image_size, (tuple, list)):
        image_size = int(image_size[0])
    else:
        image_size = int(image_size)

    context_length = int(getattr(model, "context_length", 77))

    print("weights:", weight_path)
    print("image_size:", image_size)
    print("context_length:", context_length)

    return model, tokenizer, image_size, context_length


class TFLiteRunner:
    def __init__(self, path, disable_default_delegates=False):
        import tensorflow as tf

        kwargs = dict(model_path=path, num_threads=1)
        if disable_default_delegates:
            kwargs["experimental_op_resolver_type"] = (
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )

        self.path = path
        self.interpreter = tf.lite.Interpreter(**kwargs)
        self.interpreter.allocate_tensors()

        print(f"\nTFLite {path}, disable_default_delegates={disable_default_delegates}")
        print("input:", self.interpreter.get_input_details())
        print("output:", self.interpreter.get_output_details())

    def __call__(self, x):
        x = np.asarray(x)
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        shape = tuple(int(v) for v in inp["shape"])

        if shape and shape[0] == 1 and x.shape[0] != 1:
            return np.concatenate([self(x[i:i+1]) for i in range(x.shape[0])], axis=0)

        if tuple(x.shape) != shape:
            self.interpreter.resize_tensor_input(inp["index"], x.shape, strict=False)
            self.interpreter.allocate_tensors()
            inp = self.interpreter.get_input_details()[0]
            out = self.interpreter.get_output_details()[0]

        x = x.astype(inp["dtype"], copy=False)
        self.interpreter.set_tensor(inp["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(out["index"])


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def run_torch_single(module, x):
    outs = []
    module.eval()
    with torch.no_grad():
        for i in range(x.shape[0]):
            outs.append(module(x[i:i+1]))
    return to_np(torch.cat(outs, dim=0))


def l2_normalize_np(x, eps=1e-12):
    x = np.asarray(x).astype(np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def compare(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise RuntimeError(f"{name}: shape mismatch {a.shape} vs {b.shape}")

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    flat = diff.reshape(diff.shape[0], -1)

    aa = l2_normalize_np(a.reshape(a.shape[0], -1))
    bb = l2_normalize_np(b.reshape(b.shape[0], -1))
    cos = np.sum(aa * bb, axis=-1)

    item = {
        "name": name,
        "shape": list(a.shape),
        "max_abs": float(flat.max()),
        "mean_abs": float(flat.mean()),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
    }

    print(
        f"{name:48s} "
        f"max_abs={item['max_abs']:.8g} "
        f"mean_abs={item['mean_abs']:.8g} "
        f"cos_min={item['cos_min']:.10f} "
        f"cos_mean={item['cos_mean']:.10f}"
    )
    return item


def tensor_stats(name, x):
    arr = to_np(x)
    print(
        f"{name}: shape={arr.shape}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}, "
        f"mean={arr.mean():.6f}, std={arr.std():.6f}"
    )


def synthetic_images(image_size):
    out = []

    img = Image.new("RGB", (image_size, image_size), "red")
    out.append(("synthetic_red", img))

    img = Image.new("RGB", (image_size, image_size), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([(image_size//4, image_size//4), (3*image_size//4, 3*image_size//4)], fill="blue")
    out.append(("synthetic_blue_circle", img))

    img = Image.new("RGB", (image_size, image_size), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([(20, image_size//2), (image_size-20, image_size-25)], fill="yellow")
    out.append(("synthetic_yellow_block", img))

    return out


def cifar10_images(limit=100):
    try:
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(root="./data", train=False, download=True)
        items = []
        for i in range(min(limit, len(ds))):
            img, label = ds[i]
            items.append((f"cifar10_{i}_{ds.classes[label]}", img))
        return items
    except Exception as e:
        print("CIFAR10 unavailable:", repr(e))
        return []


def preprocess_items(preprocess, items):
    names = [n for n, _ in items]
    x = torch.stack([preprocess(img) for _, img in items], dim=0).to(torch.float32)
    return names, x


def ranking_metrics(name, image_emb_fp32, image_emb_alt, text_emb_fp32, text_emb_alt, topk=(1, 5, 10)):
    """
    Compare image-text retrieval ranking drift.
    Rows: text queries, columns: images.
    """
    img0 = l2_normalize_np(image_emb_fp32)
    img1 = l2_normalize_np(image_emb_alt)
    txt0 = l2_normalize_np(text_emb_fp32)
    txt1 = l2_normalize_np(text_emb_alt)

    sim0 = txt0 @ img0.T
    sim1 = txt1 @ img1.T

    order0 = np.argsort(-sim0, axis=1)
    order1 = np.argsort(-sim1, axis=1)

    result = {
        "name": name,
        "queries": int(sim0.shape[0]),
        "images": int(sim0.shape[1]),
        "max_similarity_delta": float(np.abs(sim0 - sim1).max()),
        "mean_similarity_delta": float(np.abs(sim0 - sim1).mean()),
        "top1_same_rate": float(np.mean(order0[:, 0] == order1[:, 0])),
    }

    for k in topk:
        if k > sim0.shape[1]:
            continue
        same = []
        overlap = []
        for i in range(sim0.shape[0]):
            s0 = set(order0[i, :k].tolist())
            s1 = set(order1[i, :k].tolist())
            same.append(order0[i, :k].tolist() == order1[i, :k].tolist())
            overlap.append(len(s0 & s1) / k)
        result[f"top{k}_exact_same_rate"] = float(np.mean(same))
        result[f"top{k}_set_overlap_mean"] = float(np.mean(overlap))

    print(
        f"{name:48s} "
        f"top1_same={result['top1_same_rate']:.3f} "
        f"sim_delta_max={result['max_similarity_delta']:.8g} "
        f"sim_delta_mean={result['mean_similarity_delta']:.8g}"
    )
    return result


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    if not Path(IMAGE_TFLITE).exists():
        raise FileNotFoundError(f"Missing {IMAGE_TFLITE}")
    if not Path(TEXT_TFLITE).exists():
        raise FileNotFoundError(f"Missing {TEXT_TFLITE}")

    model, tokenizer, image_size, context_length = load_model()
    preprocess_right = make_mobileclip2_s2_preprocess(image_size)
    preprocess_openai = make_openai_clip_preprocess(image_size)

    image_ref = ImageEncoderRef(model).eval()
    text_ref = TextEncoderOriginal(model).eval()

    image_tflite = TFLiteRunner(IMAGE_TFLITE, disable_default_delegates=False)
    text_tflite = TFLiteRunner(TEXT_TFLITE, disable_default_delegates=False)

    items = synthetic_images(image_size) + cifar10_images(limit=100)
    names_right, x_right = preprocess_items(preprocess_right, items)
    _, x_openai = preprocess_items(preprocess_openai, items)

    tensor_stats("RIGHT MobileCLIP2-S2 preprocess [0,1]", x_right)
    tensor_stats("WRONG OpenAI/OpenCLIP preprocess", x_openai)

    report = {
        "model": MODEL_NAME,
        "image_size": image_size,
        "context_length": context_length,
        "preprocess_contract": {
            "correct": "RGB -> Resize/CenterCrop -> ToTensor -> float32 NCHW in [0,1]; no OpenAI mean/std Normalize",
            "wrong_for_s2": "OpenAI/OpenCLIP mean/std Normalize",
        },
        "comparisons": [],
        "ranking": [],
    }

    print("\n" + "=" * 90)
    print("FP32 correctness under correct MobileCLIP2-S2 preprocess")
    print("=" * 90)

    img_torch = run_torch_single(image_ref, x_right)
    img_tflite = image_tflite(to_np(x_right))
    report["comparisons"].append(compare("image PyTorch FP32 vs TFLite FP32", img_torch, img_tflite))

    print("\n" + "=" * 90)
    print("Contrast: OpenAI norm is a different external input contract, not Apple S2 contract")
    print("=" * 90)

    img_torch_wrong = run_torch_single(image_ref, x_openai)
    img_tflite_wrong = image_tflite(to_np(x_openai))
    report["comparisons"].append(compare("wrong-norm PyTorch vs TFLite", img_torch_wrong, img_tflite_wrong))
    report["comparisons"].append(compare("right-norm embedding vs wrong-norm embedding", img_tflite, img_tflite_wrong))

    print("\n" + "=" * 90)
    print("FP16 approximation tests on existing FP32 TFLite outputs")
    print("=" * 90)

    # 1. Simulate input precision loss only.
    x_right_fp16_rounded = to_np(x_right).astype(np.float16).astype(np.float32)
    img_tflite_input_fp16_rounded = image_tflite(x_right_fp16_rounded)
    report["comparisons"].append(compare("image TFLite FP32 input vs input rounded fp16", img_tflite, img_tflite_input_fp16_rounded))

    # 2. Simulate embedding storage/output precision loss.
    img_tflite_output_fp16 = l2_normalize_np(img_tflite.astype(np.float16).astype(np.float32)).astype(np.float32)
    report["comparisons"].append(compare("image TFLite FP32 output vs output fp16+renorm", img_tflite, img_tflite_output_fp16))

    prompts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a horse",
        "a photo of a ship",
        "a photo of a truck",
        "a photo of an airplane",
        "a photo of an automobile",
        "a red square",
        "a blue circle",
        "a yellow object",
    ]
    tokens = torch.as_tensor(tokenizer(prompts), dtype=torch.long)
    txt_torch = run_torch_single(text_ref, tokens)
    txt_tflite = text_tflite(to_np(tokens))
    txt_tflite_output_fp16 = l2_normalize_np(txt_tflite.astype(np.float16).astype(np.float32)).astype(np.float32)

    report["comparisons"].append(compare("text PyTorch FP32 vs TFLite FP32", txt_torch, txt_tflite))
    report["comparisons"].append(compare("text TFLite FP32 output vs output fp16+renorm", txt_tflite, txt_tflite_output_fp16))

    print("\n" + "=" * 90)
    print("Retrieval ranking drift from FP16 output storage")
    print("=" * 90)

    report["ranking"].append(ranking_metrics(
        "retrieval FP32 vs image-output-fp16",
        img_tflite,
        img_tflite_output_fp16,
        txt_tflite,
        txt_tflite,
    ))

    report["ranking"].append(ranking_metrics(
        "retrieval FP32 vs image+text-output-fp16",
        img_tflite,
        img_tflite_output_fp16,
        txt_tflite,
        txt_tflite_output_fp16,
    ))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nreport:", OUT_JSON)
    print("\nDecision hints:")
    print("- If 'right-norm embedding vs wrong-norm embedding' is large, that confirms OpenAI norm is a different semantic input, not a harmless equivalent.")
    print("- FP16 output storage is usually acceptable if cosine remains very close to 1 and retrieval top-k overlap stays high.")
    print("- This script does not validate FP16 weights/delegate. For that, test the actual FP16 TFLite/delegate on-device.")


if __name__ == "__main__":
    main()
