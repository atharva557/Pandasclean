# pandasclean

A lightweight Python library for cleaning and optimizing pandas DataFrames. Built for data analysts and data scientists who want practical, no-fuss data cleaning with sensible defaults.

---

## Features

- **Outlier Detection & Handling** — Detect outliers using the IQR method and choose to report, drop, or cap them
- **Memory Reduction** — Automatically downcast numeric dtypes and convert low cardinality string columns to save memory
- **NaN Handling** — *(Coming soon)*
- **Auto Clean** — *(Coming soon)* One function to rule them all

---

## Installation

```bash
pip install pandasclean
```

> Not yet on PyPI — coming soon!

---

## Usage

### Outlier Detection

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

### Memory Reduction

```python
from pandasclean import reduce_memory

df_optimized, report = reduce_memory(df)

# Disable category conversion
df_optimized, report = reduce_memory(df, convert_category=False)

# Custom cardinality threshold
df_optimized, report = reduce_memory(df, cardinality_threshold=0.3)
```

---

## How Memory Reduction Works

| dtype | Action |
|-------|--------|
| `int64` | Downcast to smallest safe int (`int8` → `int16` → `int32`) |
| `float64` | Downcast to `float32` where possible |
| `object` / `str` | Convert to `category` if cardinality ratio is below threshold |

> On a 100,000 row DataFrame, memory reduction of 50%+ is typical.

---

## Parameters

### `find_outliers(df, columns, multiplier, strategy)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `df` | required | Input DataFrame |
| `columns` | `None` | Columns to check. Defaults to all numeric columns |
| `multiplier` | `1.5` | IQR multiplier. Use `3.0` for extreme outliers only |
| `strategy` | `'report'` | One of `'report'`, `'drop'`, `'cap'` |

### `reduce_memory(df, columns, convert_category, cardinality_threshold)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `df` | required | Input DataFrame |
| `columns` | `None` | Columns to process. Defaults to all columns |
| `convert_category` | `True` | Whether to convert low cardinality strings to `category` |
| `cardinality_threshold` | `0.5` | Max unique ratio to trigger category conversion |

---

## Roadmap

- [x] Outlier detection and handling
- [x] Memory reduction
- [ ] NaN handling (drop, fill with mean/median/custom)
- [ ] Auto clean (runs all functions with defaults)
- [ ] Z-score based outlier detection

---

## License

MIT License
