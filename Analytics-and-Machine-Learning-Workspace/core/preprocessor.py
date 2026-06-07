"""Data preprocessing utilities for the analytics pipeline.

This module contains pure data transformation functions used by the Streamlit
cleaning interface. The functions operate on pandas DataFrames and return
transformed copies, preserving the original input data unless explicitly handled
by the caller.

The module covers structural cleaning, temporal feature engineering, text
standardization, cardinality reduction, memory optimization, missing-value
imputation, outlier treatment, skewness correction, numerical scaling, and
categorical encoding.
"""

import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

# =============================================================================
# Column and row management
# =============================================================================


def drop_selected_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Remove selected columns from a DataFrame.

    Missing column names are ignored, allowing the function to be used safely
    when user-selected columns may no longer exist after previous transformations.

    Args:
        df: Input DataFrame.
        columns_to_drop: Column names to remove.

    Returns:
        A new DataFrame with the selected columns removed.
    """
    return df.drop(columns=columns_to_drop, errors="ignore").reset_index(drop=True)


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        A new DataFrame containing only unique rows with a reset integer index.
    """
    return df.drop_duplicates().reset_index(drop=True)


def drop_missing_targets(df: pd.DataFrame, target_columns: list) -> pd.DataFrame:
    """Remove rows with missing values in selected columns.

    This is typically used for target variables, required fields, or critical
    columns such as geospatial coordinates.

    Args:
        df: Input DataFrame.
        target_columns: Column names that must not contain missing values.

    Returns:
        A new DataFrame with incomplete rows removed and the index reset.
    """
    return df.dropna(subset=target_columns).reset_index(drop=True)


# =============================================================================
# Temporal engineering
# =============================================================================


def engineer_temporal_features(
    df: pd.DataFrame,
    date_column: str,
    features_to_extract: list,
    datetime_format: str = "mixed",
    apply_cyclical: bool = False,
    sort_chronological: bool = False,
    set_as_index: bool = False,
) -> pd.DataFrame:
    """Parse a datetime column and engineer temporal features.

    Converts the selected column to pandas datetime format, extracts requested
    calendar-based features, optionally creates cyclical sine/cosine encodings,
    and optionally sorts or indexes the DataFrame chronologically.

    Args:
        df: Input DataFrame.
        date_column: Name of the column containing datetime values.
        features_to_extract: Temporal features to create, such as ``"Month"``,
            ``"Hour"``, or ``"Day of Week"``.
        datetime_format: Datetime parsing format. Supported values include
            ``"mixed"``, ``"ISO8601"``, or a valid pandas datetime format string.
        apply_cyclical: Whether to add sine/cosine encodings for cyclical
            temporal variables.
        sort_chronological: Whether to sort rows by the parsed datetime column.
        set_as_index: Whether to set the datetime column as the DataFrame index.

    Returns:
        A new DataFrame with the selected temporal features added.

    Raises:
        ValueError: If the selected datetime column cannot be parsed into any
            valid datetime values.
    """
    df_clean = df.copy()

    if date_column in df_clean.columns:
        initial_valid_count = df_clean[date_column].notna().sum()

        if datetime_format == "mixed":
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column],
                format="mixed",
                errors="coerce",
            )
        elif datetime_format == "ISO8601":
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column],
                format="ISO8601",
                errors="coerce",
            )
        else:
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column],
                format=datetime_format,
                errors="coerce",
            )

        final_valid_count = df_clean[date_column].notna().sum()

        if initial_valid_count > 0 and final_valid_count == 0:
            if datetime_format == "mixed":
                raise ValueError(
                    f"Validation failed: Could not detect any valid dates in "
                    f"'{date_column}'. Please ensure you selected a column that "
                    "actually contains datetime information."
                )

            raise ValueError(
                f"Validation failed: The format '{datetime_format}' destroyed "
                f"all data in '{date_column}'. Ensure you selected a valid "
                "datetime column and the correct format, or try 'Auto Detect'."
            )

        dt_col = df_clean[date_column].dt

        if "Year" in features_to_extract:
            df_clean["Year"] = dt_col.year

        if "Quarter" in features_to_extract:
            df_clean["Quarter"] = dt_col.quarter

        if "Month" in features_to_extract:
            df_clean["Month"] = dt_col.month

        if "Day" in features_to_extract:
            df_clean["Day"] = dt_col.day

        if "Hour" in features_to_extract:
            df_clean["Hour"] = dt_col.hour

        if "Day of Week" in features_to_extract:
            df_clean["Day_of_Week"] = dt_col.dayofweek

        if "Is Weekend" in features_to_extract:
            df_clean["Is_Weekend"] = (dt_col.dayofweek >= 5).astype(int)

        if "Is Month Start/End" in features_to_extract:
            df_clean["Is_Month_Start"] = dt_col.is_month_start.astype(int)
            df_clean["Is_Month_End"] = dt_col.is_month_end.astype(int)

        if apply_cyclical:
            valid_cyclical_features = {"Month", "Hour", "Day of Week"}
            if not valid_cyclical_features.intersection(features_to_extract):
                raise ValueError(
                    "To apply cyclical encoding, you must extract at least one cyclical feature: 'Month', 'Hour', or 'Day of Week'."
                )

            if "Month" in features_to_extract:
                df_clean["Month_Sin"] = np.sin(2 * np.pi * df_clean["Month"] / 12)
                df_clean["Month_Cos"] = np.cos(2 * np.pi * df_clean["Month"] / 12)

            if "Hour" in features_to_extract:
                df_clean["Hour_Sin"] = np.sin(2 * np.pi * df_clean["Hour"] / 24)
                df_clean["Hour_Cos"] = np.cos(2 * np.pi * df_clean["Hour"] / 24)

            if "Day of Week" in features_to_extract:
                df_clean["DayOfWeek_Sin"] = np.sin(
                    2 * np.pi * df_clean["Day_of_Week"] / 7
                )
                df_clean["DayOfWeek_Cos"] = np.cos(
                    2 * np.pi * df_clean["Day_of_Week"] / 7
                )

        if sort_chronological:
            df_clean = df_clean.sort_values(by=date_column).reset_index(drop=True)

        if set_as_index:
            df_clean = df_clean.set_index(date_column, drop=False)

    return df_clean


# =============================================================================
# Text standardization
# =============================================================================


def clean_text_features(
    df: pd.DataFrame,
    columns: list,
    case_mode: str = "lower",
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    strip_whitespace: bool = True,
    collapse_spaces: bool = True,
) -> pd.DataFrame:
    """Standardize selected text columns.

    Applies vectorized pandas string operations while preserving missing values
    as missing values instead of converting them to the string ``"nan"``.

    Args:
        df: Input DataFrame.
        columns: Text column names to clean.
        case_mode: Case transformation mode. Supported values are ``"lower"``,
            ``"upper"``, ``"title"``, and ``"none"``.
        remove_punctuation: Whether to remove punctuation characters.
        remove_numbers: Whether to remove numeric characters.
        strip_whitespace: Whether to remove leading and trailing whitespace.
        collapse_spaces: Whether to replace repeated whitespace with one space.

    Returns:
        A new DataFrame with standardized text columns.
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns or not pd.api.types.is_object_dtype(
            df_clean[col]
        ):
            continue

        not_null_mask = df_clean[col].notna()

        if not not_null_mask.any():
            continue

        series = df_clean.loc[not_null_mask, col].astype(str)

        if case_mode == "lower":
            series = series.str.lower()
        elif case_mode == "upper":
            series = series.str.upper()
        elif case_mode == "title":
            series = series.str.title()

        if remove_punctuation:
            series = series.str.replace(r"[^\w\s]", "", regex=True)

        if remove_numbers:
            series = series.str.replace(r"\d+", "", regex=True)

        if collapse_spaces:
            series = series.str.replace(r"\s{2,}", " ", regex=True)

        if strip_whitespace:
            series = series.str.strip()

        df_clean.loc[not_null_mask, col] = series

    return df_clean


# =============================================================================
# Cardinality reduction
# =============================================================================


def summarize_cardinality(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Summarize the number of unique categories in selected columns.

    Args:
        df: Input DataFrame.
        columns: Column names to analyze.

    Returns:
        A summary DataFrame containing each feature name and its number of
        unique categories, sorted from highest to lowest cardinality.
    """
    summary = []

    for col in columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            summary.append(
                {
                    "Feature": col,
                    "Unique Categories": unique_count,
                }
            )

    return pd.DataFrame(summary).sort_values(
        by="Unique Categories",
        ascending=False,
    )


def reduce_cardinality(
    df: pd.DataFrame,
    columns: list,
    method: str = "frequency",
    threshold_percent: float = 0.01,
    top_n: int = 10,
    substring: str = "",
    replacement_label: str = "Other",
) -> pd.DataFrame:
    """Group categories in high-cardinality columns.

    Supports frequency-based grouping, top-N category retention, and
    case-insensitive substring matching.

    Args:
        df: Input DataFrame.
        columns: Categorical column names to reduce.
        method: Reduction method. Supported values are ``"frequency"``,
            ``"top_n"``, and ``"substring"``.
        threshold_percent: Minimum relative frequency required to keep a label
            when using frequency-based reduction.
        top_n: Number of most frequent categories to keep when using top-N
            reduction.
        substring: Text pattern to match when using substring-based grouping.
        replacement_label: Replacement label assigned to grouped categories.

    Returns:
        A new DataFrame with reduced-cardinality categorical columns.
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns:
            continue

        not_null_mask = df_clean[col].notna()

        if not not_null_mask.any():
            continue

        if method == "frequency":
            frequencies = df_clean.loc[not_null_mask, col].value_counts(normalize=True)
            rare_labels = frequencies[frequencies < threshold_percent].index
            df_clean[col] = df_clean[col].replace(rare_labels, replacement_label)

        elif method == "top_n":
            top_labels = (
                df_clean.loc[not_null_mask, col].value_counts().nlargest(top_n).index
            )
            replace_mask = not_null_mask & ~df_clean[col].isin(top_labels)
            df_clean.loc[replace_mask, col] = replacement_label

        elif method == "substring":
            if substring:
                contains_mask = (
                    df_clean[col]
                    .astype(str)
                    .str.contains(substring, case=False, na=False)
                )
                df_clean.loc[contains_mask, col] = replacement_label

    return df_clean


# =============================================================================
# Memory optimization
# =============================================================================


def optimize_memory_usage(
    df: pd.DataFrame,
    downcast_integers: bool = True,
    downcast_floats: bool = True,
    categorize_strings: bool = True,
) -> pd.DataFrame:
    """Reduce DataFrame memory usage where safe.

    Downcasts integer and floating-point columns to smaller numeric types when
    their value ranges allow it. Optionally converts object columns to pandas
    categorical dtype when the unique-value ratio is below 50%.

    Args:
        df: Input DataFrame.
        downcast_integers: Whether to downcast integer columns.
        downcast_floats: Whether to downcast floating-point columns.
        categorize_strings: Whether to convert suitable object columns to
            categorical dtype.

    Returns:
        A memory-optimized copy of the input DataFrame.
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        col_type = df_clean[col].dtype

        if downcast_integers and pd.api.types.is_integer_dtype(col_type):
            c_min = df_clean[col].min()
            c_max = df_clean[col].max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_clean[col] = df_clean[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_clean[col] = df_clean[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_clean[col] = df_clean[col].astype(np.int32)

        elif downcast_floats and pd.api.types.is_float_dtype(col_type):
            c_min = df_clean[col].min()
            c_max = df_clean[col].max()

            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_clean[col] = df_clean[col].astype(np.float32)

        elif categorize_strings and pd.api.types.is_object_dtype(col_type):
            num_unique = df_clean[col].nunique()
            num_total = len(df_clean[col])

            if num_unique / num_total < 0.5:
                df_clean[col] = df_clean[col].astype("category")

    return df_clean


# =============================================================================
# Data imputation
# =============================================================================


def impute_numerical_features(
    df: pd.DataFrame,
    columns: list,
    strategy: str = "median",
) -> pd.DataFrame:
    """Impute missing values in selected numerical columns.

    Args:
        df: Input DataFrame.
        columns: Numeric column names to impute.
        strategy: Imputation strategy. Supported values are ``"median"``,
            ``"mean"``, ``"mode"``, and ``"zero"``.

    Returns:
        A new DataFrame with selected numerical columns imputed.
    """
    df_clean = df.copy()

    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            if strategy == "median":
                fill_val = df_clean[col].median()
            elif strategy == "mean":
                fill_val = df_clean[col].mean()
            elif strategy == "mode":
                mode_series = df_clean[col].mode()
                fill_val = mode_series[0] if not mode_series.empty else 0
            else:
                fill_val = 0

            df_clean[col] = df_clean[col].fillna(fill_val)

    return df_clean


def impute_categorical_features(
    df: pd.DataFrame,
    columns: list,
    strategy: str = "Unknown",
) -> pd.DataFrame:
    """Impute missing values in selected categorical columns.

    Args:
        df: Input DataFrame.
        columns: Categorical or text column names to impute.
        strategy: Imputation strategy. Supported values are ``"unknown"``,
            ``"mode"``, and ``"forward fill"``.

    Returns:
        A new DataFrame with selected categorical columns imputed.
    """
    df_clean = df.copy()

    for col in columns:
        if col in df_clean.columns:
            if strategy == "mode":
                fill_val = (
                    df_clean[col].mode()[0]
                    if not df_clean[col].mode().empty
                    else "Unknown"
                )
                df_clean[col] = df_clean[col].fillna(fill_val)

            elif strategy == "forward fill":
                df_clean[col] = df_clean[col].ffill().fillna("Unknown")

            else:
                df_clean[col] = df_clean[col].fillna("Unknown")

    return df_clean


# =============================================================================
# Outlier treatment
# =============================================================================


def handle_outliers(
    df: pd.DataFrame,
    columns: list,
    method: str = "iqr",
    action: str = "cap",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    percentile_range: tuple = (0.01, 0.99),
) -> pd.DataFrame:
    """Detect and treat outliers in selected numerical columns.

    Supports IQR bounds, Z-score bounds, and percentile-based bounds. Outliers
    can be capped, replaced with missing values, or removed by dropping rows.

    Args:
        df: Input DataFrame.
        columns: Numeric column names to process.
        method: Detection method. Supported values are ``"iqr"``, ``"zscore"``,
            and ``"percentile"``.
        action: Treatment method. Supported values are ``"drop"``, ``"cap"``,
            and ``"nan"``.
        iqr_multiplier: Multiplier applied to the interquartile range.
        zscore_threshold: Number of standard deviations used for Z-score bounds.
        percentile_range: Lower and upper quantiles used for percentile bounds.

    Returns:
        A new DataFrame with selected outliers treated.
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(
            df_clean[col]
        ):
            continue

        if method == "iqr":
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (iqr_multiplier * iqr)
            upper_bound = q3 + (iqr_multiplier * iqr)

        elif method == "zscore":
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower_bound = mean - (zscore_threshold * std)
            upper_bound = mean + (zscore_threshold * std)

        elif method == "percentile":
            lower_bound = df_clean[col].quantile(percentile_range[0])
            upper_bound = df_clean[col].quantile(percentile_range[1])

        else:
            continue

        if action == "drop":
            valid_mask = (
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ) | df_clean[col].isna()
            df_clean = df_clean[valid_mask]

        elif action == "cap":
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

        elif action == "nan":
            outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            df_clean.loc[outlier_mask, col] = np.nan

    return df_clean


def summarize_outliers(
    df: pd.DataFrame,
    columns: list,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    percentile_range: tuple = (0.01, 0.99),
) -> pd.DataFrame:
    """Summarize outlier counts in selected numerical columns.

    Performs a dry-run analysis using the chosen statistical thresholding method
    without modifying the underlying data.

    Args:
        df: Input DataFrame.
        columns: Numeric column names to analyze.
        method: Detection method. Supported values are ``"iqr"``, ``"zscore"``,
            and ``"percentile"``.
        iqr_multiplier: Multiplier applied to the interquartile range.
        zscore_threshold: Number of standard deviations used for Z-score bounds.
        percentile_range: Lower and upper quantiles used for percentile bounds.

    Returns:
        A summary DataFrame containing feature names, outlier counts, and the
        percentage of rows affected.
    """
    summary = []

    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (iqr_multiplier * iqr)
            upper_bound = q3 + (iqr_multiplier * iqr)

        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - (zscore_threshold * std)
            upper_bound = mean + (zscore_threshold * std)

        elif method == "percentile":
            lower_bound = df[col].quantile(percentile_range[0])
            upper_bound = df[col].quantile(percentile_range[1])

        else:
            continue

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        pct = (outliers / len(df)) * 100

        summary.append(
            {
                "Feature": col,
                "Outlier Count": outliers,
                "% of Data": pct,
            }
        )

    return pd.DataFrame(summary)


# =============================================================================
# Skewness transformation
# =============================================================================


def summarize_skewness(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Summarize skewness severity for selected numerical columns.

    Args:
        df: Input DataFrame.
        columns: Numeric column names to analyze.

    Returns:
        A summary DataFrame containing feature names, skewness values, and
        diagnostic labels.
    """
    summary = []

    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            skew_val = df[col].skew()

            if pd.isna(skew_val):
                continue

            if abs(skew_val) > 1.0:
                severity = "Highly Skewed"
            elif abs(skew_val) > 0.5:
                severity = "Moderately Skewed"
            else:
                severity = "Symmetric"

            summary.append(
                {
                    "Feature": col,
                    "Skewness": skew_val,
                    "Diagnosis": severity,
                }
            )

    return pd.DataFrame(summary)


def fix_numerical_skewness(
    df: pd.DataFrame,
    columns: list,
    method: str = "yeo-johnson",
) -> pd.DataFrame:
    """Apply distribution transformations to selected numerical columns.

    Safely ignores missing values during transformation by applying operations
    only to non-null rows and assigning the transformed values back to the same
    locations.

    Args:
        df: Input DataFrame.
        columns: Numeric column names to transform.
        method: Transformation method. Supported values are ``"yeo-johnson"``,
            ``"box-cox"``, ``"log1p"``, and ``"sqrt"``.

    Returns:
        A new DataFrame with transformed numerical columns.

    Raises:
        ValueError: If ``"log1p"``, ``"sqrt"``, or ``"box-cox"`` is selected
            for data outside the valid mathematical domain.
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(
            df_clean[col]
        ):
            continue

        df_clean[col] = df_clean[col].astype(float)

        not_null_mask = df_clean[col].notna()

        if not not_null_mask.any():
            continue

        min_val = df_clean.loc[not_null_mask, col].min()

        if method == "log1p":
            if min_val < -1:
                raise ValueError(
                    f"Feature '{col}' contains values < -1. Log1p requires "
                    "values >= -1."
                )

            df_clean.loc[not_null_mask, col] = np.log1p(
                df_clean.loc[not_null_mask, col]
            )

        elif method == "sqrt":
            if min_val < 0:
                raise ValueError(
                    f"Feature '{col}' contains negative values. Square Root "
                    "requires positive values."
                )

            df_clean.loc[not_null_mask, col] = np.sqrt(df_clean.loc[not_null_mask, col])

        elif method == "box-cox":
            if min_val <= 0:
                raise ValueError(
                    f"Feature '{col}' contains zero or negative values. Box-Cox "
                    "requires strictly positive data (>0). Use Yeo-Johnson "
                    "instead."
                )

            transformed_data, _ = boxcox(df_clean.loc[not_null_mask, col])
            df_clean.loc[not_null_mask, col] = transformed_data

        elif method == "yeo-johnson":
            transformed_data, _ = yeojohnson(df_clean.loc[not_null_mask, col])
            df_clean.loc[not_null_mask, col] = transformed_data

    return df_clean


# =============================================================================
# Scaling and encoding
# =============================================================================


def scale_numerical_features(
    df: pd.DataFrame,
    columns: list,
    method: str = "standard",
) -> pd.DataFrame:
    """Scale selected numerical columns.

    Applies the selected scikit-learn scaler independently to each numeric
    column. Missing values are ignored during fitting and transformation.

    Args:
        df: Input DataFrame.
        columns: Numeric column names to scale.
        method: Scaling method. Supported values are ``"standard"``,
            ``"minmax"``, ``"robust"``, and ``"maxabs"``.

    Returns:
        A new DataFrame with selected numerical columns scaled.

    Raises:
        ValueError: If the selected scaling method is not supported.
    """
    df_clean = df.copy()

    if not columns:
        return df_clean

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler(),
    }

    if method not in scalers:
        raise ValueError(f"Scaling method '{method}' is not supported.")

    scaler = scalers[method]

    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            not_null_mask = df_clean[col].notna()

            if not_null_mask.any():
                scaled_data = scaler.fit_transform(df_clean.loc[not_null_mask, [col]])
                df_clean[col] = df_clean[col].astype("float64")
                df_clean.loc[not_null_mask, col] = scaled_data.flatten()

    return df_clean


def encode_categorical_features(
    df: pd.DataFrame,
    columns: list,
    method: str = "onehot",
) -> pd.DataFrame:
    """Encode selected categorical columns.

    Supports one-hot encoding through pandas dummy variables and label encoding
    through pandas factorization.

    Args:
        df: Input DataFrame.
        columns: Categorical column names to encode.
        method: Encoding method. Supported values are ``"onehot"`` and
            ``"label"``.

    Returns:
        A new DataFrame with selected categorical columns encoded.
    """
    df_clean = df.copy()

    if not columns:
        return df_clean

    valid_cols = [col for col in columns if col in df_clean.columns]

    if method == "onehot":
        df_clean = pd.get_dummies(df_clean, columns=valid_cols, dtype=int)

    elif method == "label":
        for col in valid_cols:
            not_null_mask = df_clean[col].notna()
            labels, _ = pd.factorize(df_clean.loc[not_null_mask, col])
            df_clean.loc[not_null_mask, col] = labels
            df_clean[col] = df_clean[col].astype("float64")

    return df_clean
