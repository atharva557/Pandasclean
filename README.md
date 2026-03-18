# pandasclean

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

- **Outlier Detection & Handling** — Detect outliers using the IQR method and choose to report, drop, or cap them
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

Runs all cleaning functions in the correct order with default settings.

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

> On a 1.5 million row DataFrame, memory reduction of 75%+ is typical.

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

## Roadmap

- [x] Outlier detection and handling (IQR method)
- [x] Memory reduction (dtype downcasting + category conversion)
- [x] NaN handling (drop, mean, median, custom)
- [x] Auto clean
- [ ] Z-score based outlier detection
- [ ] Consistency improvements across functions

---

## License

MIT License