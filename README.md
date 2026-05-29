# pandasclean [![Downloads](https://pepy.tech/badge/pandasclean)](https://pepy.tech/project/pandasclean)


A lightweight Python library for cleaning and optimizing pandas DataFrames. Built for data analysts and data scientists who want practical, no-fuss data cleaning with sensible defaults.

---
## Quick Start

```python
import pandas as pd
from pandasclean import auto_clean

df = pd.read_csv('your_data.csv')
df_clean, report = auto_clean(df)
```

That's it. One line cleans your entire DataFrame.

---

## Features

- **Outlier Detection & Handling (IQR)** — Detect outliers using the IQR method and choose to report, drop, or cap them
- **Outlier Detection & Handling (Z-score)** — Detect outliers using the Z-score method and choose to report, drop, or cap them
- **Memory Reduction** — Automatically downcast numeric dtypes and convert low cardinality string columns to save memory
- **NaN Handling** — Drop, fill with mean/median, or supply custom fill values per column
- **Auto Clean** — One function that runs everything with sensible defaults

---

## Installation

```bash
pip install pandasclean
```

---

## Usage

### Auto Clean

Runs all cleaning functions in the correct order with sensible defaults.

```python
from pandasclean import auto_clean

df_clean, report = auto_clean(df)
```

For custom behaviour, use the individual functions directly.

---

### Outlier Detection & Handling

```python
from pandasclean import find_outliers

# Report outlier bounds without changing data
df, bounds = find_outliers(df, strategy='report')

# Drop rows containing outliers
df_clean, bounds = find_outliers(df, strategy='drop')

# Cap outliers to the nearest bound (Winsorization)
df_clean, bounds = find_outliers(df, strategy='cap')

# Target specific columns with a custom multiplier
df_clean, bounds = find_outliers(df, columns=['age', 'salary'], multiplier=3.0, strategy='cap')
```

---

### Z-score Outlier Detection & Handling

```python
from pandasclean import find_outliers_zscore

# Report outlier info without changing data
df, info = find_outliers_zscore(df, strategy='report')

# Drop rows containing outliers
df_clean, info = find_outliers_zscore(df, strategy='drop')

# Cap outliers to mean ± (threshold × std)
df_clean, info = find_outliers_zscore(df, strategy='cap')

# Target specific columns with a custom threshold
df_clean, info = find_outliers_zscore(df, columns=['age', 'salary'], threshold=2.0, strategy='cap')
```

---

### NaN Handling

```python
from pandasclean import handle_nan

# Report null counts and percentages without making changes
df, report = handle_nan(df, strategy='report')

# Drop rows with any NaN
df_clean, report = handle_nan(df, strategy='drop', axis='rows', how='any')

# Drop columns where more than 50% of values are NaN
df_clean, report = handle_nan(df, strategy='drop', axis='columns', threshold=50)

# Fill NaN with column mean (numeric columns only)
df_clean, report = handle_nan(df, strategy='mean')

# Fill NaN with column median (numeric columns only)
df_clean, report = handle_nan(df, strategy='median')

# Fill all NaN with a single custom value
df_clean, report = handle_nan(df, strategy='custom', fill_value=0)

# Fill NaN with different values per column
df_clean, report = handle_nan(df, strategy='custom', fill_value={
    'age': 0,
    'name': 'unknown',
    'salary': 50000
})
```

---

### Memory Reduction

```python
from pandasclean import reduce_memory

# Optimize all columns with default settings
df_optimized, report = reduce_memory(df)

# Disable category conversion for string columns
df_optimized, report = reduce_memory(df, convert_category=False)

# Custom cardinality threshold for category conversion
df_optimized, report = reduce_memory(df, cardinality_threshold=0.3)
```

#### Benchmarks

Tested on a 1.5 million row DataFrame with mixed dtypes:

| Column | Before | After | Notes |
|--------|--------|-------|-------|
| `Account_ID` | `int64` | `int32` | Downcast |
| `Customer_Age` | `int64` | `int8` | Downcast |
| `Account_Balance` | `float64` | `float32` | Downcast |
| `Monthly_Spend` | `float64` | `float32` | Downcast |
| `Credit_Score` | `float64` | `float32` | Downcast |
| `Activity_Metric_1` | `float64` | `float32` | Downcast |
| `Activity_Metric_2` | `float64` | `float32` | Downcast |
| `Tier` | `object` | `category` | Low cardinality |

```python
before = df.memory_usage(deep=True).sum() / (1024 * 1024)
df_optimized, report = reduce_memory(df)
after = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)

# Before: 152.71 MB
# After:  37.19 MB
# Reduction: 75.6%
```

> Always use `memory_usage(deep=True)` for accurate measurement — pandas default undercounts string column memory.

---

## How It Works

### Memory Reduction

| dtype | Action |
|-------|--------|
| `int64` | Downcast to smallest safe type (`int8` → `int16` → `int32`) |
| `float64` | Downcast to `float32` where possible |
| `object` / `str` | Convert to `category` if cardinality ratio is below threshold |

### Outlier Detection

Uses the IQR method to compute bounds:
- `lower_bound = Q1 - (multiplier × IQR)`
- `upper_bound = Q3 + (multiplier × IQR)`

Standard multiplier values:
- `1.5` — mild outliers (default)
- `3.0` — extreme outliers only

### Z-score Outlier Detection

Uses the Z-score method to compute bounds:
- `z_score = (value - mean) / std`
- `lower_bound = mean - (threshold × std)`
- `upper_bound = mean + (threshold × std)`

Standard threshold values:
- `2.0` — aggressive (catches more outliers)
- `3.0` — conservative (default, catches extreme outliers only)

---

## Parameters

### `auto_clean(df)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `df` | required | Input DataFrame |

---

### `find_outliers(df, columns, multiplier, strategy)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `df` | required | Input DataFrame |
| `columns` | `None` | Columns to check. Defaults to all numeric columns |
| `multiplier` | `1.5` | IQR multiplier. Use `3.0` for extreme outliers only |
| `strategy` | `'report'` | One of `'report'`, `'drop'`, `'cap'` |

---

### `find_outliers_zscore(df, columns, threshold, strategy)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `df` | required | Input DataFrame |
| `columns` | `None` | Columns to check. Defaults to all numeric columns |
| `threshold` | `3.0` | Z-score threshold. Use `2.0` for more aggressive detection |
| `strategy` | `'report'` | One of `'report'`, `'drop'`, `'cap'` |

---

### `handle_nan(df, columns, strategy, fill_value, axis, how, threshold)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `df` | required | Input DataFrame |
| `columns` | `None` | Columns to process. Defaults to all columns |
| `strategy` | `'report'` | One of `'report'`, `'drop'`, `'mean'`, `'median'`, `'custom'` |
| `fill_value` | `None` | Scalar or dict. Required when strategy is `'custom'` |
| `axis` | `'rows'` | `'rows'` or `'columns'`. Only used with `'drop'` strategy |
| `how` | `'any'` | `'any'` or `'all'`. Only used with `'drop'` strategy |
| `threshold` | `None` | NaN percentage threshold for dropping. Overrides `how` when set |

---

### `reduce_memory(df, columns, convert_category, cardinality_threshold)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `df` | required | Input DataFrame |
| `columns` | `None` | Columns to process. Defaults to all columns |
| `convert_category` | `True` | Whether to convert low cardinality strings to `category` |
| `cardinality_threshold` | `0.5` | Max unique ratio to trigger category conversion |

---

## Use Cases

- **Data cleaning before analysis** — handle outliers and NaN values in one step
- **Memory optimization before ML training** — reduce DataFrame size before fitting models, especially useful for GPU training where float32 is standard
- **CSV/Parquet compression** — smaller dtypes mean smaller files on disk. Saving to parquet after running `reduce_memory` can reduce file sizes significantly

---

## Roadmap

- [x] Outlier detection and handling (IQR method)
- [x] Memory reduction (dtype downcasting + category conversion)
- [x] NaN handling (drop, mean, median, custom)
- [x] Auto clean
- [x] Z-score based outlier detection
- [ ] Skewness detection and fixing
- [ ] Duplicate detection and removal
- [ ] HTML report generator

---

## Contributing

We welcome contributions! If you want to help improve `pandasclean`, please consider the following:

1. **Reporting Bugs**: Open an issue on GitHub to report any bugs or any unexpected behavior.
2. **Running Tests**: Ensure that your changes are covered by tests. You can run them using:
   ```bash
   python -m unittest tests.py
   python -m unittest discover
   ```
3. **Pull Requests**: When submitting a PR, please provide a clear description of your changes and ensure that the tests pass.
4. **Documentation**: If you add a new feature, please also update the documentation.

## License

MIT License
