# Drone Image Aesthetics Assessment Results

Date: 2025-04-04 10:43:08

## Model Comparison

| Metric | NIMA | LightNIMA | Improvement |
|--------|------|-----------|------------|
| Model Size | 17.38 MB | 1.18 MB | 93.2% reduction |
| Inference Time | 11.33 ms | 3.23 ms | 3.50x faster |
| Pearson Correlation | - | 0.9140 | - |
| Spearman Correlation | - | 0.8061 | - |
| Mean Absolute Error | - | 0.2164 | - |
| Root Mean Square Error | - | 0.2412 | - |

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
