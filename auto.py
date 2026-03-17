import pandas as pd
import numpy as np
from numpy import dtype
from numpy.ma.extras import unique


def clean():
    pass



def find_outliers(df, columns=None, multiplier=1.5, strategy='report'):
    """
        Detect and handle outliers in a DataFrame using the IQR method.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to analyze.

        columns : list of str, optional
            Column names to check for outliers. If None, all numeric
            columns are selected automatically.

        multiplier : float, optional (default=1.5)
            The IQR multiplier used to define outlier bounds.
            Standard values are 1.5 (mild outliers) and 3.0 (extreme outliers).
            Negative values are automatically converted to their absolute value.

        strategy : str, optional (default='report')
            How to handle detected outliers. Options:
            - 'report' : Return the original DataFrame and the outlier bounds
                         dict without making any changes.
            - 'drop'   : Remove rows where any value falls outside the computed
                         bounds. Uses a single combined mask for efficiency.
            - 'cap'    : Clip outlier values to the nearest bound (Winsorization)
                         instead of removing the row.
            Any unrecognized strategy defaults to 'report' with a printed warning.

        Returns
        -------
        df_result : pd.DataFrame
            - 'report' : Original DataFrame (unmodified).
            - 'drop'   : Cleaned DataFrame with outlier rows removed.
            - 'cap'    : DataFrame with outlier values clipped to bounds.

        outliers : dict
            Maps each column name to one of:
            - tuple (lower_bound, upper_bound) : valid numeric bounds.
            - str : reason the column was skipped (non-numeric, zero IQR, etc.)

        Examples
        --------
        >>> df_out, bounds = find_outliers(df, strategy='report')
        >>> df_clean, bounds = find_outliers(df, columns=['age', 'salary'], strategy='drop')
        >>> df_capped, bounds = find_outliers(df, multiplier=3.0, strategy='cap')
        """
    outliers = {}
    if multiplier<0:
        print('multiplier should be positive defaulting to abs of your value')
        multiplier=abs(multiplier)
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns

    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                outliers[column] = "IQR is 0 (Constant or near-constant data)"
                continue
            upper_bound = Q3 + (multiplier * IQR)
            lower_bound = Q1 - (multiplier * IQR)
            outliers[column] = (lower_bound, upper_bound)
        else:
            outliers[column] = "Is not a numeric data type"

    if strategy == 'report':
        return df,outliers

    elif strategy == 'drop':
        mask = pd.Series(True, index=df.index)

        for column in columns:
            if isinstance(outliers[column], tuple):
                lower_bound = outliers[column][0]
                upper_bound = outliers[column][1]

                # Update the mask using '&' (bitwise AND)
                # A row stays True ONLY if it is currently True AND passes this column's check
                mask = mask & (df[column] >= lower_bound) & (df[column] <= upper_bound)

        # Apply the mask all at once to create the cleaned dataframe
        df_cleaned = df[mask].copy()
        return df_cleaned, outliers

    elif strategy == 'cap':
        df_cleaned = df.copy()

        for column in columns:
            if isinstance(outliers[column], tuple):
                lower_bound = outliers[column][0]
                upper_bound = outliers[column][1]
                df_cleaned[column] = df_cleaned[column].astype(float)

                df_cleaned.loc[df_cleaned[column] > upper_bound, column] = upper_bound
                df_cleaned.loc[df_cleaned[column] < lower_bound, column] = lower_bound

        return df_cleaned, outliers
    else:
        print('No valid strategy found defaulting to report')
        return df,outliers



def reduce_memory(df, columns=None, convert_category=True,cardinality_threshold=0.5):
    """
    Reduce memory usage of a DataFrame by downcasting numeric dtypes
    and converting low cardinality string columns to category dtype.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to optimize.

    columns : list of str, optional
        Column names to process. If None, all columns are selected automatically.

    convert_category : bool, optional (default=True)
        Whether to convert low cardinality string/object columns to category dtype.
        Set to False if you want to skip string columns entirely.

    cardinality_threshold : float, optional (default=0.5)
        The maximum ratio of unique values to total rows for a string column
        to be converted to category. For example, 0.5 means a column must have
        less than 50% unique values to be converted.
        Lower values are more conservative, higher values are more aggressive.

    Returns
    -------
    df_clean : pd.DataFrame
        Memory optimized copy of the input DataFrame.

    report : dict
        Maps each column name to a dict with:
        - 'before' : dtype before optimization
        - 'after'  : dtype after optimization

    Notes
    -----
    - Integer columns are downcast to the smallest safe type (int8 → int16 → int32 → int64)
    - Float columns are downcast from float64 to float32 where possible. Minor precision loss may occur.
    - String columns with low cardinality are converted to category dtype which
      can yield significant memory savings for repetitive string data.
    - High cardinality string columns (like IDs or emails) are left unchanged
      as converting them to category would increase memory usage.

    Examples
    --------
    >>> df_optimized, report = reduce_memory(df)
    >>> df_optimized, report = reduce_memory(df, convert_category=False)
    >>> df_optimized, report = reduce_memory(df, cardinality_threshold=0.3)
    """
    if isinstance(df, pd.DataFrame):
        df_clean = df.copy()
    else:
        raise TypeError("Not a dataframe")

    if columns is None:
        columns = df_clean.columns.tolist()

    report_dict={}
    total_memory=df_clean.memory_usage(deep=True).sum()
    total_memory_mb = total_memory / (1024 * 1024)
    for column in columns:
        report_dict[column]={'before':df_clean[column].dtype}
        if pd.api.types.is_integer_dtype(df_clean[column]):
            col_min=df_clean[column].min()
            col_max=df_clean[column].max()
            if col_min>=np.iinfo(np.int8).min and col_max<=np.iinfo(np.int8).max:
                df_clean[column]=df_clean[column].astype('int8')
            elif col_min>=np.iinfo(np.int16).min and col_max<=np.iinfo(np.int16).max:
                df_clean[column]=df_clean[column].astype('int16')
            elif col_min>=np.iinfo(np.int32).min and col_max<=np.iinfo(np.int32).max:
                df_clean[column]=df_clean[column].astype('int32')

        elif pd.api.types.is_float_dtype(df_clean[column]):
            col_min = df_clean[column].min()
            col_max = df_clean[column].max()
            if col_min>=np.finfo(np.float32).min and col_max<=np.finfo(np.float32).max:
                df_clean[column] = df_clean[column].astype('float32')



        elif pd.api.types.is_object_dtype(df_clean[column]) or pd.api.types.is_string_dtype(df_clean[column]):
            if convert_category:
                unique_values=df_clean[column].nunique()
                total_rows=len(df_clean[column])
                cardinality_ratio=unique_values/total_rows
                if cardinality_ratio<cardinality_threshold:
                    df_clean[column]=df_clean[column].astype('category')

        report_dict[column]['after']=df_clean[column].dtype


    final_total_memory = df_clean.memory_usage(deep=True).sum()
    final_total_memory_mb = final_total_memory / (1024 * 1024)
    memory_saved=total_memory_mb-final_total_memory_mb
    report_dict['summary'] = {
        'memory_before_mb': round(total_memory_mb, 2),
        'memory_after_mb': round(final_total_memory_mb, 2),
        'memory_saved_mb': round(memory_saved, 2)
    }
    print(f'{total_memory_mb:.2f}-->{final_total_memory_mb:.2f} ({memory_saved:.2f}mb saved)')
    return df_clean,report_dict