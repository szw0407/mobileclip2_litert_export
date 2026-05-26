#!/usr/bin/env python3
"""
Minimal MobileCLIP2-S2 LiteRT/TFLite export + test script.

This version explicitly fixes MobileCLIP2-S2 image preprocessing:

  MobileCLIP2-S0/S2/B variants should use:
      mean=(0,0,0), std=(1,1,1)
  i.e. Resize/Crop/ToTensor only, value range [0, 1].

The exported image model still expects:
      float32 NCHW [1, 3, 256, 256]
The preprocessing is NOT inside the model; it must be done outside.

Three steps:
  1. Load HuggingFace model
  2. Convert image/text encoders
  3. Test exported TFLite files against original PyTorch outputs

Run:
  TF_ENABLE_ONEDNN_OPTS=0 PYTHONFAULTHANDLER=1 python main.py

Optional:
  python main.py --skip-export
  python main.py --only-export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


MODEL_NAME = "MobileCLIP2-S2"
IMAGE_TFLITE = "mobileclip2_s2_image.tflite"
TEXT_TFLITE = "mobileclip2_s2_text.tflite"
THRESHOLD = 10e-4  # 1e-3


# =============================================================================
# Export wrappers
# =============================================================================

class ImageEncoderForExport(nn.Module):
    """
    Equivalent to normalized model.encode_image(x).

    Input contract:
      x: float32 NCHW, already preprocessed, for MobileCLIP2-S2 usually [0, 1].
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.encode_image(x)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x


class TextEncoderForExport(nn.Module):
    """
    Equivalent to normalized model.encode_text(text), but avoids torch.argmax.

    Original open_clip text pooling is logically:
      x = x[batch_indices, text.argmax(dim=-1)]

    LiteRT may fail to legalize ArgMax, so we create an equivalent one-hot mask:
      score = token_id * (context_length + 1) + reverse_position
      mask = score == reduce_max(score)
      pooled = reduce_sum(x * mask)

    This matches torch.argmax tie-breaking: earliest maximum wins.
    """

    def __init__(self, model, context_length=77):
        super().__init__()
        self.text = model.text
        self.ln_final = model.text.ln_final
        self.text_projection = getattr(model.text, "text_projection", None)
        self.context_length = int(context_length)

        reverse_position = torch.arange(
            self.context_length - 1,
            -1,
            -1,
            dtype=torch.float32,
        )
        self.register_buffer("reverse_position", reverse_position, persistent=False)

    def forward(self, text):
        x, attn_mask = self.text._embeds(text)
        x = self.text.transformer(x, attn_mask=attn_mask)
        x = self.ln_final(x)

        score = text.to(torch.float32) * float(self.context_length + 1)
        score = score + self.reverse_position.unsqueeze(0)
        max_score = torch.amax(score, dim=1, keepdim=True)
        mask = (score == max_score).to(dtype=x.dtype).unsqueeze(-1)

        x = torch.sum(x * mask, dim=1)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x


class TextEncoderOriginal(nn.Module):
    """Reference: use original model.encode_text + normalization."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        x = self.model.encode_text(text)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x


# =============================================================================
# Step 1: load HF model
# =============================================================================

def make_mobileclip2_s2_preprocess(image_size):
    """
    IMPORTANT:
    MobileCLIP2-S2 image normalization is external and should be identity:
      mean=(0,0,0), std=(1,1,1)

    Therefore this returns only Resize/CenterCrop/ToTensor.
    """
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),  # RGB -> float32 CHW in [0, 1], no Normalize
    ])


def load_hf_model():
    print("=" * 90)
    print("Step 1/3: load HuggingFace model")
    print("=" * 90)

    import open_clip
    from huggingface_hub import hf_hub_download
    from mobileclip.modules.common.mobileone import reparameterize_model

    weight_file = f"{MODEL_NAME.lower().replace('-', '_')}.pt"
    weight_path = hf_hub_download(
        repo_id=f"apple/{MODEL_NAME}",
        filename=weight_file,
        cache_dir="./hf_cache",
    )
    print(f"weights: {weight_path}")

    # We intentionally ignore the returned preprocess because local checkpoint loading
    # can fall back to OpenCLIP default normalization, which is wrong for MobileCLIP2-S2.
    model, _, openclip_preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=weight_path,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model.eval()
    model = reparameterize_model(model)
    model.eval()

    image_size = getattr(model.visual, "image_size", 224)
    if isinstance(image_size, (tuple, list)):
        image_size = int(image_size[0])
    else:
        image_size = int(image_size)

    context_length = int(getattr(model, "context_length", 77))

    preprocess = make_mobileclip2_s2_preprocess(image_size)

    print(f"image_size: {image_size}")
    print(f"context_length: {context_length}")
    print("preprocess: Resize + CenterCrop + ToTensor, NO Normalize")
    print("input contract: RGB float32 NCHW in [0, 1]")

    return model, tokenizer, preprocess, openclip_preprocess, image_size, context_length


# =============================================================================
# Step 2: convert
# =============================================================================

def convert_models(model, image_size, context_length):
    print("\n" + "=" * 90)
    print("Step 2/3: convert PyTorch wrappers to LiteRT/TFLite")
    print("=" * 90)

    from litert_torch import convert

    image_encoder = ImageEncoderForExport(model).eval()
    text_encoder = TextEncoderForExport(model, context_length).eval()

    # Use realistic MobileCLIP2-S2 image input range [0, 1], not torch.randn.
    image_sample = torch.rand(1, 3, image_size, image_size, dtype=torch.float32)
    text_sample = torch.zeros(1, context_length, dtype=torch.long)

    print(f"export image encoder -> {IMAGE_TFLITE}")
    image_litert = convert(
        module=image_encoder,
        sample_args=(image_sample,),
    )
    image_litert.export(IMAGE_TFLITE)

    print(f"export text encoder -> {TEXT_TFLITE}")
    text_litert = convert(
        module=text_encoder,
        sample_args=(text_sample,),
    )
    text_litert.export(TEXT_TFLITE)

    print("conversion done")


# =============================================================================
# Step 3: test
# =============================================================================

def make_synthetic_images(image_size):
    images = []

    img = Image.new("RGB", (image_size, image_size), "red")
    images.append(("red_square", img))

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [(image_size // 4, image_size // 4), (3 * image_size // 4, 3 * image_size // 4)],
        fill="blue",
    )
    images.append(("blue_circle", img))

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [(20, image_size // 2), (image_size - 20, image_size - 25)],
        fill="yellow",
    )
    for i in range(5):
        x = 40 + i * max(15, image_size // 8)
        draw.rectangle([(x, image_size // 2 - 25), (x + 5, image_size // 2)], fill="red")
    images.append(("birthday_cake", img))

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [(image_size // 4, image_size // 3), (3 * image_size // 4, 2 * image_size // 3)],
        outline="black",
        width=3,
    )
    draw.rectangle(
        [(image_size // 4 - 10, 2 * image_size // 3), (3 * image_size // 4 + 10, 2 * image_size // 3 + 18)],
        fill="gray",
    )
    images.append(("laptop", img))

    return images


def make_cifar10_images(limit=40):
    try:
        from torchvision.datasets import CIFAR10

        dataset = CIFAR10(root="./data", train=False, download=True)
        result = []
        seen = {}
        per_class = max(1, limit // 10)

        for img, label in dataset:
            if seen.get(label, 0) >= per_class:
                continue
            seen[label] = seen.get(label, 0) + 1
            result.append((f"cifar10_{dataset.classes[label]}_{seen[label]}", img))
            if len(result) >= limit:
                break

        return result

    except Exception as e:
        print(f"skip CIFAR10: {e}")
        return []


def make_random_valid_tokens(batch, context_length):
    tokens = torch.zeros(batch, context_length, dtype=torch.long)
    sot = 49406
    eot = 49407

    for i in range(batch):
        eot_pos = min(context_length - 1, 2 + i % max(1, context_length - 2))
        tokens[i, 0] = sot
        if eot_pos > 1:
            tokens[i, 1:eot_pos] = torch.randint(1, 32000, (eot_pos - 1,), dtype=torch.long)
        tokens[i, eot_pos] = eot

    return tokens


def tensor_stats(name, x):
    arr = to_numpy(x)
    print(
        f"{name}: shape={arr.shape} "
        f"min={arr.min():.6f} max={arr.max():.6f} "
        f"mean={arr.mean():.6f} std={arr.std():.6f}"
    )
    if arr.ndim == 4 and arr.shape[1] == 3:
        print("  channel_mean:", arr.mean(axis=(0, 2, 3)))
        print("  channel_std: ", arr.std(axis=(0, 2, 3)))


class TFLiteRunner:
    def __init__(self, path, disable_default_delegates=False):
        import tensorflow as tf

        kwargs = dict(model_path=path, num_threads=1)
        if disable_default_delegates:
            kwargs["experimental_op_resolver_type"] = (
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )

        self.tf = tf
        self.path = path
        self.disable_default_delegates = disable_default_delegates
        self.interpreter = tf.lite.Interpreter(**kwargs)
        self.interpreter.allocate_tensors()

        print(f"\nTFLite {path}, disable_default_delegates={disable_default_delegates}")
        print("input:", self.interpreter.get_input_details())
        print("output:", self.interpreter.get_output_details())

    def __call__(self, x):
        x = np.asarray(x)
        in_info = self.interpreter.get_input_details()[0]
        out_info = self.interpreter.get_output_details()[0]

        model_shape = tuple(int(v) for v in in_info["shape"])
        if model_shape and model_shape[0] == 1 and x.shape[0] != 1:
            outputs = [self(x[i:i + 1]) for i in range(x.shape[0])]
            return np.concatenate(outputs, axis=0)

        if tuple(x.shape) != model_shape:
            self.interpreter.resize_tensor_input(in_info["index"], x.shape, strict=False)
            self.interpreter.allocate_tensors()
            in_info = self.interpreter.get_input_details()[0]
            out_info = self.interpreter.get_output_details()[0]

        x = x.astype(in_info["dtype"], copy=False)
        self.interpreter.set_tensor(in_info["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(out_info["index"])


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def run_torch(module, x):
    module.eval()
    with torch.no_grad():
        return to_numpy(module(x))


def run_torch_single(module, x):
    module.eval()
    outs = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            outs.append(module(x[i:i + 1]))
    return to_numpy(torch.cat(outs, dim=0))


def compare(name, ref, got, threshold=THRESHOLD):
    ref = np.asarray(ref)
    got = np.asarray(got)

    if ref.shape != got.shape:
        raise RuntimeError(f"{name}: shape mismatch: ref={ref.shape}, got={got.shape}")

    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    flat = diff.reshape(diff.shape[0], -1)
    max_abs = float(flat.max())
    mean_abs = float(flat.mean())

    ref_flat = ref.reshape(ref.shape[0], -1).astype(np.float64)
    got_flat = got.reshape(got.shape[0], -1).astype(np.float64)
    ref_flat /= np.maximum(np.linalg.norm(ref_flat, axis=-1, keepdims=True), 1e-12)
    got_flat /= np.maximum(np.linalg.norm(got_flat, axis=-1, keepdims=True), 1e-12)
    cosine = np.sum(ref_flat * got_flat, axis=-1)

    passed = max_abs <= threshold

    print(
        f"{'PASS' if passed else 'FAIL'} | {name:44s} "
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} "
        f"cos_min={float(cosine.min()):.10f} cos_mean={float(cosine.mean()):.10f}"
    )

    return {
        "name": name,
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cos_min": float(cosine.min()),
        "cos_mean": float(cosine.mean()),
        "shape": list(ref.shape),
    }


def compare_per_sample(name, names, ref, got, top_k=12):
    ref = np.asarray(ref)
    got = np.asarray(got)
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    flat = diff.reshape(diff.shape[0], -1)

    ref_flat = ref.reshape(ref.shape[0], -1).astype(np.float64)
    got_flat = got.reshape(got.shape[0], -1).astype(np.float64)
    ref_flat /= np.maximum(np.linalg.norm(ref_flat, axis=-1, keepdims=True), 1e-12)
    got_flat /= np.maximum(np.linalg.norm(got_flat, axis=-1, keepdims=True), 1e-12)
    cos = np.sum(ref_flat * got_flat, axis=-1)

    order = np.argsort(cos)
    print(f"\nWorst samples for {name}:")
    for rank, i in enumerate(order[:top_k]):
        print(
            f"  #{rank+1:02d} idx={int(i):02d} {names[i]:28s} "
            f"cos={cos[i]:.10f} max_abs={flat[i].max():.8g} mean_abs={flat[i].mean():.8g}"
        )


def preprocess_items(preprocess, items):
    names = [name for name, _ in items]
    x = torch.stack([preprocess(img) for _, img in items], dim=0).to(torch.float32)
    return names, x


def test_exported_models(model, tokenizer, preprocess, openclip_preprocess, image_size, context_length, compare_wrong_preprocess=False):
    print("\n" + "=" * 90)
    print("Step 3/3: test exported TFLite models against original PyTorch")
    print("=" * 90)

    image_ref = ImageEncoderForExport(model).eval()
    text_ref_original = TextEncoderOriginal(model).eval()
    text_ref_no_argmax = TextEncoderForExport(model, context_length).eval()

    image_tflite_default = TFLiteRunner(IMAGE_TFLITE, disable_default_delegates=False)
    image_tflite_builtin = TFLiteRunner(IMAGE_TFLITE, disable_default_delegates=True)
    text_tflite_default = TFLiteRunner(TEXT_TFLITE, disable_default_delegates=False)

    results = []

    # Text equivalence proof.
    prompts = [
        "a red square",
        "a blue circle",
        "a birthday cake",
        "a laptop computer",
        "a photo of a dog",
        "a photo of a truck",
        "an indoor scene",
    ]
    prompt_tokens = torch.as_tensor(tokenizer(prompts), dtype=torch.long)

    results.append(compare(
        "text original vs no-argmax wrapper",
        run_torch(text_ref_original, prompt_tokens),
        run_torch(text_ref_no_argmax, prompt_tokens),
    ))

    random_valid_tokens = make_random_valid_tokens(12, context_length)
    results.append(compare(
        "text original vs no-argmax random-eot",
        run_torch(text_ref_original, random_valid_tokens),
        run_torch(text_ref_no_argmax, random_valid_tokens),
    ))

    # Dataset 1: synthetic.
    synthetic = make_synthetic_images(image_size)
    synthetic_names, synthetic_images = preprocess_items(preprocess, synthetic)
    tensor_stats("synthetic MobileCLIP2-S2 preprocess", synthetic_images)

    ref_syn_single = run_torch_single(image_ref, synthetic_images)
    tf_syn_default = image_tflite_default(to_numpy(synthetic_images))
    tf_syn_builtin = image_tflite_builtin(to_numpy(synthetic_images))

    results.append(compare(
        "synthetic torch single vs tflite default",
        ref_syn_single,
        tf_syn_default,
    ))
    results.append(compare(
        "synthetic torch single vs tflite builtin",
        ref_syn_single,
        tf_syn_builtin,
    ))

    # Dataset 2: CIFAR10.
    cifar10 = make_cifar10_images(limit=40)
    if cifar10:
        cifar_names, cifar_images = preprocess_items(preprocess, cifar10)
        tensor_stats("cifar10 MobileCLIP2-S2 preprocess", cifar_images)

        ref_cifar_batch = run_torch(image_ref, cifar_images)
        ref_cifar_single = run_torch_single(image_ref, cifar_images)
        tf_cifar_default = image_tflite_default(to_numpy(cifar_images))
        tf_cifar_builtin = image_tflite_builtin(to_numpy(cifar_images))

        results.append(compare(
            "cifar10 torch batch vs torch single",
            ref_cifar_batch,
            ref_cifar_single,
        ))
        results.append(compare(
            "cifar10 torch single vs tflite default",
            ref_cifar_single,
            tf_cifar_default,
        ))
        results.append(compare(
            "cifar10 torch single vs tflite builtin",
            ref_cifar_single,
            tf_cifar_builtin,
        ))
        results.append(compare(
            "cifar10 tflite default vs builtin",
            tf_cifar_default,
            tf_cifar_builtin,
        ))

        compare_per_sample(
            "cifar10 torch single vs tflite default",
            cifar_names,
            ref_cifar_single,
            tf_cifar_default,
        )

        if compare_wrong_preprocess:
            wrong_names, wrong_images = preprocess_items(openclip_preprocess, cifar10)
            tensor_stats("cifar10 OpenCLIP returned preprocess, for contrast", wrong_images)

            ref_wrong_single = run_torch_single(image_ref, wrong_images)
            tf_wrong_default = image_tflite_default(to_numpy(wrong_images))

            results.append(compare(
                "CONTRAST wrong-preprocess torch single vs tflite",
                ref_wrong_single,
                tf_wrong_default,
            ))
            compare_per_sample(
                "CONTRAST wrong-preprocess torch single vs tflite",
                wrong_names,
                ref_wrong_single,
                tf_wrong_default,
            )

    # Dataset 3: random in correct input range [0, 1].
    random_images = torch.rand(16, 3, image_size, image_size, dtype=torch.float32)
    tensor_stats("random uniform [0,1]", random_images)

    ref_random_single = run_torch_single(image_ref, random_images)
    tf_random_default = image_tflite_default(to_numpy(random_images))
    tf_random_builtin = image_tflite_builtin(to_numpy(random_images))

    results.append(compare(
        "random[0,1] torch single vs tflite default",
        ref_random_single,
        tf_random_default,
    ))
    results.append(compare(
        "random[0,1] torch single vs tflite builtin",
        ref_random_single,
        tf_random_builtin,
    ))
    results.append(compare(
        "random[0,1] tflite default vs builtin",
        tf_random_default,
        tf_random_builtin,
    ))

    # Text TFLite.
    synthetic_tokens = torch.as_tensor(tokenizer([
        "a red square",
        "a blue circle",
        "a birthday cake",
        "a laptop computer",
    ]), dtype=torch.long)

    results.append(compare(
        "synthetic text torch original vs tflite",
        run_torch(text_ref_original, synthetic_tokens),
        text_tflite_default(to_numpy(synthetic_tokens)),
    ))

    report = {
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "image_size": image_size,
        "context_length": context_length,
        "image_preprocess": {
            "resize": image_size,
            "center_crop": image_size,
            "to_tensor": True,
            "normalize_mean": [0.0, 0.0, 0.0],
            "normalize_std": [1.0, 1.0, 1.0],
            "input_range": "[0, 1]",
            "layout": "NCHW",
            "dtype": "float32",
        },
        "passed": all(r["passed"] for r in results),
        "results": results,
        "files": {
            IMAGE_TFLITE: Path(IMAGE_TFLITE).stat().st_size if Path(IMAGE_TFLITE).exists() else None,
            TEXT_TFLITE: Path(TEXT_TFLITE).stat().st_size if Path(TEXT_TFLITE).exists() else None,
        },
    }

    with open("equivalence_report_mobileclip2_s2_preprocess_fixed.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nreport: equivalence_report_mobileclip2_s2_preprocess_fixed.json")
    print("overall:", "PASS" if report["passed"] else "FAIL")

    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true", help="Do not export; only test existing .tflite files.")
    parser.add_argument("--only-export", action="store_true", help="Export only; do not test.")
    parser.add_argument("--compare-wrong-preprocess", action="store_true", help="Also test OpenCLIP returned preprocess as a contrast.")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    model, tokenizer, preprocess, openclip_preprocess, image_size, context_length = load_hf_model()

    if not args.skip_export:
        convert_models(model, image_size, context_length)

    if not args.only_export:
        test_exported_models(
            model,
            tokenizer,
            preprocess,
            openclip_preprocess,
            image_size,
            context_length,
            compare_wrong_preprocess=args.compare_wrong_preprocess,
        )


if __name__ == "__main__":
    main()
