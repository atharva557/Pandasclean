import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from pandasclean import find_outliers, reduce_memory, handle_nan, auto_clean

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_df():
    """Base DataFrame used by most tests."""
    return pd.DataFrame({
        'age':    [25, 30, 35, 200, 22, 28],   # 200 is outlier
        'salary': [50000, 60000, 55000, 52000, 1000000, 58000],  # 1000000 is outlier
        'name':   ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve', 'Frank'],
    })

def make_nan_df():
    """DataFrame with missing values."""
    return pd.DataFrame({
        'age':    [25, None, 35, None, 22, 28],
        'salary': [50000, 60000, None, 52000, None, 58000],
        'name':   ['Alice', None, 'Charlie', 'Dave', None, 'Frank'],
    })

def make_memory_df():
    """DataFrame suited for memory reduction testing."""
    return pd.DataFrame({
        'small_int':  np.array([1, 2, 3, 4, 5], dtype=np.int64),
        'big_float':  np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
        'category':   ['a', 'b', 'a', 'b', 'a'],
        'unique_str': ['x1', 'x2', 'x3', 'x4', 'x5'],
    })

passed = 0
failed = 0

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL  {name}")
        print(f"          {type(e).__name__}: {e}")
        failed += 1


# ── find_outliers tests ────────────────────────────────────────────────────────

print("\n── find_outliers ──────────────────────────────────────────────────────")

def test_outliers_report_returns_df_and_dict():
    df = make_df()
    result, bounds = find_outliers(df, strategy='report')
    assert isinstance(result, pd.DataFrame)
    assert isinstance(bounds, dict)
run_test("report strategy returns (DataFrame, dict)", test_outliers_report_returns_df_and_dict)

def test_outliers_report_does_not_modify_df():
    df = make_df()
    result, _ = find_outliers(df, strategy='report')
    assert len(result) == len(df)
run_test("report strategy does not modify DataFrame", test_outliers_report_does_not_modify_df)

def test_outliers_drop_removes_outlier_rows():
    df = make_df()
    result, _ = find_outliers(df, strategy='drop')
    assert len(result) < len(df)
run_test("drop strategy removes outlier rows", test_outliers_drop_removes_outlier_rows)

def test_outliers_drop_does_not_mutate_original():
    df = make_df()
    original_len = len(df)
    find_outliers(df, strategy='drop')
    assert len(df) == original_len
run_test("drop strategy does not mutate original DataFrame", test_outliers_drop_does_not_mutate_original)

def test_outliers_cap_keeps_same_row_count():
    df = make_df()
    result, _ = find_outliers(df, strategy='cap')
    assert len(result) == len(df)
run_test("cap strategy keeps same number of rows", test_outliers_cap_keeps_same_row_count)

def test_outliers_cap_clamps_values():
    df = make_df()
    result, bounds = find_outliers(df, columns=['age'], strategy='cap')
    lower, upper = bounds['age']
    assert result['age'].max() <= upper
    assert result['age'].min() >= lower
run_test("cap strategy clamps values within bounds", test_outliers_cap_clamps_values)

def test_outliers_negative_multiplier_converted():
    df = make_df()
    result, _ = find_outliers(df, multiplier=-1.5, strategy='report')
    assert isinstance(result, pd.DataFrame)
run_test("negative multiplier is converted to positive", test_outliers_negative_multiplier_converted)

def test_outliers_zero_iqr_handled():
    df = pd.DataFrame({'const': [5, 5, 5, 5, 5]})
    _, bounds = find_outliers(df, strategy='report')
    assert isinstance(bounds['const'], str)
run_test("zero IQR column is handled gracefully", test_outliers_zero_iqr_handled)

def test_outliers_non_numeric_column_skipped():
    df = make_df()
    _, bounds = find_outliers(df, columns=['name'], strategy='report')
    assert isinstance(bounds['name'], str)
run_test("non-numeric column is skipped with message", test_outliers_non_numeric_column_skipped)

def test_outliers_invalid_strategy_defaults():
    df = make_df()
    result, _ = find_outliers(df, strategy='invalid')
    assert isinstance(result, pd.DataFrame)
run_test("invalid strategy defaults to report", test_outliers_invalid_strategy_defaults)


# ── reduce_memory tests ────────────────────────────────────────────────────────

print("\n── reduce_memory ──────────────────────────────────────────────────────")

def test_memory_returns_df_and_report():
    df = make_memory_df()
    result, report = reduce_memory(df)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(report, dict)
run_test("returns (DataFrame, dict)", test_memory_returns_df_and_report)

def test_memory_int_downcast():
    df = pd.DataFrame({'x': np.array([1, 2, 3], dtype=np.int64)})
    result, _ = reduce_memory(df)
    assert result['x'].dtype == np.int8
run_test("int64 downcast to int8 for small values", test_memory_int_downcast)

def test_memory_float_downcast():
    df = pd.DataFrame({'x': np.array([1.1, 2.2, 3.3], dtype=np.float64)})
    result, _ = reduce_memory(df)
    assert result['x'].dtype == np.float32
run_test("float64 downcast to float32", test_memory_float_downcast)

def test_memory_category_conversion():
    df = pd.DataFrame({'tier': ['a', 'b', 'a', 'b', 'a'] * 100})
    result, _ = reduce_memory(df)
    assert str(result['tier'].dtype) == 'category'
run_test("low cardinality string converted to category", test_memory_category_conversion)

def test_memory_high_cardinality_not_converted():
    df = pd.DataFrame({'uid': [f'user_{i}' for i in range(1000)]})
    result, _ = reduce_memory(df)
    assert str(result['uid'].dtype) != 'category'
run_test("high cardinality string not converted to category", test_memory_high_cardinality_not_converted)

def test_memory_convert_category_false():
    df = pd.DataFrame({'tier': ['a', 'b', 'a', 'b', 'a'] * 100})
    result, _ = reduce_memory(df, convert_category=False)
    assert str(result['tier'].dtype) != 'category'
run_test("convert_category=False skips string columns", test_memory_convert_category_false)

def test_memory_does_not_mutate_original():
    df = make_memory_df()
    original_dtypes = df.dtypes.copy()
    reduce_memory(df)
    assert df.dtypes.equals(original_dtypes)
run_test("does not mutate original DataFrame", test_memory_does_not_mutate_original)

def test_memory_invalid_input_raises():
    try:
        reduce_memory([1, 2, 3])
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
run_test("non-DataFrame input raises TypeError", test_memory_invalid_input_raises)

def test_memory_report_has_summary():
    df = make_memory_df()
    _, report = reduce_memory(df)
    assert 'summary' in report
    assert 'memory_saved_mb' in report['summary']
run_test("report contains summary with memory_saved_mb", test_memory_report_has_summary)


# ── handle_nan tests ───────────────────────────────────────────────────────────

print("\n── handle_nan ─────────────────────────────────────────────────────────")

def test_nan_report_returns_counts():
    df = make_nan_df()
    _, report = handle_nan(df, strategy='report')
    assert report['columns']['age']['null count'] == 2
run_test("report strategy returns correct null counts", test_nan_report_returns_counts)

def test_nan_report_does_not_modify_df():
    df = make_nan_df()
    result, _ = handle_nan(df, strategy='report')
    assert result.isnull().sum().sum() == df.isnull().sum().sum()
run_test("report strategy does not modify DataFrame", test_nan_report_does_not_modify_df)

def test_nan_drop_rows():
    df = make_nan_df()
    result, _ = handle_nan(df, strategy='drop', axis='rows')
    assert result.isnull().sum().sum() == 0
run_test("drop rows removes all NaN rows", test_nan_drop_rows)

def test_nan_drop_columns():
    df = make_nan_df()
    result, _ = handle_nan(df, strategy='drop', axis='columns')
    assert result.isnull().sum().sum() == 0
run_test("drop columns removes all NaN columns", test_nan_drop_columns)

def test_nan_mean_fills_numeric():
    df = make_nan_df()
    result, _ = handle_nan(df, columns=['age', 'salary'], strategy='mean')
    assert result['age'].isnull().sum() == 0
    assert result['salary'].isnull().sum() == 0
run_test("mean strategy fills numeric NaN values", test_nan_mean_fills_numeric)

def test_nan_mean_skips_non_numeric():
    df = make_nan_df()
    result, report = handle_nan(df, columns=['name'], strategy='mean')
    assert report['columns']['name']['action'] == 'skipped - not numeric'
run_test("mean strategy skips non-numeric columns", test_nan_mean_skips_non_numeric)

def test_nan_median_fills_numeric():
    df = make_nan_df()
    result, _ = handle_nan(df, columns=['age', 'salary'], strategy='median')
    assert result['age'].isnull().sum() == 0
run_test("median strategy fills numeric NaN values", test_nan_median_fills_numeric)

def test_nan_custom_scalar():
    df = make_nan_df()
    result, _ = handle_nan(df, columns=['age'], strategy='custom', fill_value=0)
    assert result['age'].isnull().sum() == 0
    assert (result['age'] == 0).any()
run_test("custom scalar fills all NaN with given value", test_nan_custom_scalar)

def test_nan_custom_dict():
    df = make_nan_df()
    result, _ = handle_nan(df, strategy='custom', fill_value={'age': 0, 'salary': 99999})
    assert result['age'].isnull().sum() == 0
    assert result['salary'].isnull().sum() == 0
run_test("custom dict fills columns with individual values", test_nan_custom_dict)

def test_nan_custom_none_raises():
    df = make_nan_df()
    try:
        handle_nan(df, strategy='custom', fill_value=None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
run_test("custom strategy with no fill_value raises ValueError", test_nan_custom_none_raises)

def test_nan_invalid_strategy_raises():
    df = make_nan_df()
    try:
        handle_nan(df, strategy='invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
run_test("invalid strategy raises ValueError", test_nan_invalid_strategy_raises)

def test_nan_invalid_input_raises():
    try:
        handle_nan([1, 2, 3])
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
run_test("non-DataFrame input raises TypeError", test_nan_invalid_input_raises)


# ── auto_clean tests ───────────────────────────────────────────────────────────

print("\n── auto_clean ─────────────────────────────────────────────────────────")

def test_auto_clean_returns_df_and_report():
    df = make_df()
    result, report = auto_clean(df)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(report, dict)
run_test("returns (DataFrame, dict)", test_auto_clean_returns_df_and_report)

def test_auto_clean_report_has_all_keys():
    df = make_df()
    _, report = auto_clean(df)
    assert 'NaN handling' in report
    assert 'Memory info' in report
    assert 'Outliers info' in report
run_test("report contains all three function keys", test_auto_clean_report_has_all_keys)

def test_auto_clean_invalid_input_raises():
    try:
        auto_clean([1, 2, 3])
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
run_test("non-DataFrame input raises TypeError", test_auto_clean_invalid_input_raises)

def test_auto_clean_does_not_mutate_original():
    df = make_df()
    original_shape = df.shape
    auto_clean(df)
    assert df.shape == original_shape
run_test("does not mutate original DataFrame", test_auto_clean_does_not_mutate_original)


# ── Summary ────────────────────────────────────────────────────────────────────

total = passed + failed
print(f"\n{'─' * 60}")
print(f"  Results: {passed}/{total} passed", "🎉" if failed == 0 else "⚠️")
if failed > 0:
    print(f"  {failed} test(s) failed — fix before publishing to PyPI")
print(f"{'─' * 60}\n")