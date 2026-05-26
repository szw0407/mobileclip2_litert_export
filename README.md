# MobileCLIP2-S2 → LiteRT/TFLite Export

Automatically export Apple's [MobileCLIP2-S2](https://huggingface.co/apple/MobileCLIP2-S2) image and text encoders into standalone TFLite models for on-device retrieval on Android/Flutter.

## Quick Start

```bash
# Install Apple's ml-mobileclip (official)
git clone https://github.com/apple/ml-mobileclip.git
pip install -e ./ml-mobileclip

# One-click export
python scripts/run_all.py
```

Or step by step:

```bash
python scripts/00_check_env.py
python scripts/01_download_model.py
python scripts/02_export_litert_torch.py           # preferred path
python scripts/03_export_onnx_fallback.py           # if litert-torch fails
python scripts/08_inspect_tflite_io.py
python scripts/04_validate_numerical.py             # fp32 validation
python scripts/07_export_fp16.py                    # optional
python scripts/04_validate_numerical.py --fp16      # fp16 validation
python scripts/05_make_test_dataset.py
python scripts/06_run_retrieval_smoke_test.py
```

## Artifacts

```
artifacts/
  models/
    mobileclip2_s2_image_fp32.tflite     # Image encoder FP32
    mobileclip2_s2_text_fp32.tflite      # Text encoder FP32
    mobileclip2_s2_image_fp16.tflite     # Image encoder FP16 (optional)
    mobileclip2_s2_text_fp16.tflite      # Text encoder FP16 (optional)
    tokenizer_config.json                # Tokenizer config for Android/Flutter
    model_metadata.json                  # Full model metadata
  test_data/
    cifar10_subset/                      # CIFAR10 subset for smoke test
    synthetic/                           # Synthetic images for smoke test
  reports/
    conversion_report.json               # Full conversion report
    numerical_validation_fp32.json       # Numerical validation results
    numerical_validation_fp16.json       # FP16 numerical validation
    tflite_io_report.json                # TFLite IO shape/dtype inspection
    retrieval_smoke_test_report.md       # Retrieval smoke test report
```

## Model Details

| Property | Value |
|----------|-------|
| Image input shape | `[1, 3, 256, 256]` (NCHW) float32 |
| Text input shape | `[1, 77]` int64 (or int32) |
| Embedding dimension | 512 |
| Output | L2-normalized embedding (norm ≈ 1.0) |

## Export Method

- **Preferred**: `litert_torch.convert()` (litert-torch / ai-edge-torch)
- **Fallback**: `PyTorch → ONNX → onnx2tf → TFLite`

The actual method used is recorded in `conversion_report.json`.

## Flutter / Android Usage

### Image Encoder
1. Resize to 256×256 (BICUBIC), CenterCrop to 224×224
2. Normalize: mean=[0.48145, 0.45783, 0.40821], std=[0.26863, 0.26130, 0.27578]
3. Convert to float32 NCHW tensor (1×3×224×224)
4. Run through `mobileclip2_s2_image_fp32.tflite`
5. Output is a 512-dim L2-normalized embedding

### Text Encoder
1. Tokenize with CLIP tokenizer (context_length=77, SOT=49406, EOT=49407)
2. Convert to int64 tensor (1×77)
3. 77 tokens: `<|startoftext|> token1 token2 ... <|endoftext|> <pad>...`
4. Run through `mobileclip2_s2_text_fp32.tflite`
5. Output is a 512-dim L2-normalized embedding

### Similarity
```dart
double cosineSimilarity(List<double> a, List<double> b) {
  double dot = 0, na = 0, nb = 0;
  for (int i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (sqrt(na) * sqrt(nb) + 1e-12);
}
```

## Requirements

- Python 3.10 or 3.11
- Apple's [ml-mobileclip](https://github.com/apple/ml-mobileclip) (`pip install -e ./ml-mobileclip`)
- See `requirements.txt` for full list

## Known Limitations

- **CIFAR10 smoke test** does NOT represent real gallery retrieval quality (low 32×32 resolution).
- **Synthetic images** are simple geometric shapes—only verify the pipeline, not model capability.
- **INT8 quantization** is not recommended by default; CLIP embedding retrieval is sensitive to INT8 sorting errors. Use `--int8-experimental` only if you validate recall yourself.
- **Token dtype**: TFLite may prefer int32 inputs. The pipeline automatically tries int32 fallback if int64 fails.
- **Layout**: PyTorch uses NCHW. The exported TFLite preserves NCHW layout. Android/Flutter side must feed NCHW data.
