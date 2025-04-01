# Drone Image Aesthetics Assessment Results

Date: 2025-03-28 15:27:08

## Model Comparison

| Metric | NIMA | LightNIMA | Improvement |
|--------|------|-----------|------------|
| Model Size | 17.38 MB | 1.18 MB | 93.2% reduction |
| Inference Time | 10.11 ms | 3.19 ms | 3.17x faster |
| Pearson Correlation | - | 0.9461 | - |
| Spearman Correlation | - | 0.9030 | - |
| Mean Absolute Error | - | 0.2786 | - |
| Root Mean Square Error | - | 0.2952 | - |

## Visualization Files

- `score_correlation.png`: Scatter plot of NIMA vs. LightNIMA scores
- `error_distribution.png`: Histogram of prediction errors
- `time_comparison.png`: Comparison of inference times
- `model_size_comparison.png`: Comparison of model sizes
- `average_distribution.png`: Average score distribution comparison
- `distributions/`: 10 individual distribution plots

## Data Files

- `detailed_results.csv`: Detailed metrics for each image
- `summary_statistics.csv`: Summary statistics for model comparison
