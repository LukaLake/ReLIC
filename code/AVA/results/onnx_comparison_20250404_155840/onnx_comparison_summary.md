# PyTorch vs ONNX Model Comparison

Date: 2025-04-04

Device for PyTorch: CPU
Device for ONNX: CPU

## Inference Performance

| Model | PyTorch (ms) | ONNX (ms) | Speedup |
|-------|-------------|-----------|--------|
| NIMA Teacher | 33.12 | 2.76 | 12.05x |
| LightNIMA | 10.49 | 0.83 | 12.86x |

## Prediction Consistency

| Model | PyTorch vs ONNX Score Difference |
|-------|----------------------------------|
| NIMA Teacher | 0.0000 |
| LightNIMA | 0.0000 |

## Model File Size

| Model | PyTorch (MB) | ONNX (MB) | Size Change |
|-------|-------------|-----------|-------------|
| NIMA Teacher | 17.51 | 17.09 | -2.4% |
| LightNIMA | 1.23 | 1.17 | -5.2% |

## Conclusion

- ✅ **Significant performance improvement**: ONNX models are substantially faster than PyTorch models
- ✅ **High prediction consistency**: ONNX models produce almost identical results to PyTorch models
- ✅ **Significant model compression**: LightNIMA is less than half the size of NIMA Teacher in both formats
