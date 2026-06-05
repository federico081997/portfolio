import pandas as pd
import numpy as np
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)

# ==========================================
#           COLUMN & ROW MANAGEMENT
# ==========================================


def drop_selected_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Removes specified columns from the DataFrame safely.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (list): A list of column name strings to be removed.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns dropped.
                      Ignores errors if a column does not exist.
    """
    return df.drop(columns=columns_to_drop, errors="ignore")


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes exact duplicate rows across all columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing only unique rows.
    """
    return df.drop_duplicates().reset_index(drop=True)


def drop_missing_targets(df: pd.DataFrame, target_columns: list) -> pd.DataFrame:
    """
    Drops rows where critical target variables or geospatial coordinates are missing.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_columns (list): A list of column names that must not contain missing values.

    Returns:
        pd.DataFrame: A new DataFrame with rows containing NaNs in the target columns removed.
    """
    return df.dropna(subset=target_columns).reset_index(drop=True)


# ==========================================
#           TEMPORAL ENGINEERING
# ==========================================


def engineer_temporal_features(
    df: pd.DataFrame,
    date_column: str,
    features_to_extract: list,
    datetime_format: str = "mixed",
    apply_cyclical: bool = False,
    sort_chronological: bool = False,
    set_as_index: bool = False,
) -> pd.DataFrame:
    """
    Parses datetime strings, extracts granular temporal features, and structures
    the DataFrame for time-series forecasting.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column (str): The column containing datetime information.
        features_to_extract (list): List of temporal features to create
            (e.g., ['Month', 'Hour']).
        datetime_format (str, optional): The exact string format to parse dates
            efficiently (e.g., '%Y-%m-%d', 'ISO8601', or 'mixed'). Defaults to 'mixed'.
        apply_cyclical (bool, optional): Whether to generate Sin/Cos cyclical
            features to preserve continuous mathematical time. Defaults to False.
        sort_chronological (bool, optional): Sorts the DataFrame chronologically
            to prevent data leakage in time-series splits. Defaults to False.
        set_as_index (bool, optional): Sets the datetime column as the DataFrame
            index, required for some forecasting models. Defaults to False.

    Returns:
        pd.DataFrame: A structurally updated DataFrame with new temporal features.
    """
    df_clean = df.copy()

    if date_column in df_clean.columns:
        # Count valid data BEFORE conversion
        initial_valid_count = df_clean[date_column].notna().sum()

        # 1. Parse the strings into pandas datetime objects
        if datetime_format == "mixed":
            # Fallback to guessing if the user explicitly wants auto-detect
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column], format="mixed", errors="coerce"
            )
        elif datetime_format == "ISO8601":
            # Pandas built-in ISO parser
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column], format="ISO8601", errors="coerce"
            )
        else:
            # Strict, high-speed parsing using the provided format string
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column], format=datetime_format, errors="coerce"
            )

        # Count valid data AFTER conversion
        final_valid_count = df_clean[date_column].notna().sum()

        # If we started with data but ended with 0 valid dates, the format is entirely wrong
        # or the column cannot be coverted to datetime
        if initial_valid_count > 0 and final_valid_count == 0:
            if datetime_format == "mixed":
                # If Auto-Detect failed to find a single date, it's almost certainly the wrong column
                raise ValueError(
                    f"Validation failed: Could not detect any valid dates in '{date_column}'. "
                    "Please ensure you selected a column that actually contains datetime information."
                )
            else:
                # If a specific format failed, it could be the column OR the format
                raise ValueError(
                    f"Validation failed: The format '{datetime_format}' destroyed all data in '{date_column}'. "
                    "Ensure you selected a valid datetime column AND the correct format, or try 'Auto Detect'."
                )

        dt_col = df_clean[date_column].dt

        # 2. Standard & Business Feature Extraction
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

        # Boolean Flags (Converted to 1/0 for ML)
        if "Is Weekend" in features_to_extract:
            df_clean["Is_Weekend"] = (dt_col.dayofweek >= 5).astype(int)
        if "Is Month Start/End" in features_to_extract:
            df_clean["Is_Month_Start"] = dt_col.is_month_start.astype(int)
            df_clean["Is_Month_End"] = dt_col.is_month_end.astype(int)

        # 3. Cyclical Mathematical Encoding
        if apply_cyclical:
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

        # 4. Time-Series Structural Prep
        if sort_chronological:
            df_clean = df_clean.sort_values(by=date_column).reset_index(drop=True)

        if set_as_index:
            # drop=False keeps the column in the dataframe for feature extraction later
            df_clean = df_clean.set_index(date_column, drop=False)

    return df_clean


# ==========================================
#           TEXT STANDARDIZATION
# ==========================================


def clean_text_features(
    df: pd.DataFrame,
    columns: list,
    case_mode: str = "lower",
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    strip_whitespace: bool = True,
    collapse_spaces: bool = True,
) -> pd.DataFrame:
    """
    Applies modular string transformations to standardize text data.

    Operates using vectorized pandas string methods for high performance.
    Safely ignores null values to prevent converting NaNs into the string "nan".

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of text column names to process.
        case_mode (str): 'lower', 'upper', 'title', or 'none'. Defaults to 'lower'.
        remove_punctuation (bool): Strips all non-alphanumeric characters (excluding spaces).
        remove_numbers (bool): Strips all digits 0-9.
        strip_whitespace (bool): Removes leading and trailing spaces.
        collapse_spaces (bool): Converts multiple consecutive spaces into a single space.

    Returns:
        pd.DataFrame: DataFrame with standardized text columns.
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns or not pd.api.types.is_object_dtype(
            df_clean[col]
        ):
            continue

        # Isolate non-nulls using a boolean mask to prevent converting NaN to "nan"
        not_null_mask = df_clean[col].notna()
        if not not_null_mask.any():
            continue

        # Force valid rows to string for vectorized operations
        s = df_clean.loc[not_null_mask, col].astype(str)

        # 1. Case Normalization
        if case_mode == "lower":
            s = s.str.lower()
        elif case_mode == "upper":
            s = s.str.upper()
        elif case_mode == "title":
            s = s.str.title()

        # 2. Filtering (Regex)
        if remove_punctuation:
            # Replaces anything that is NOT a word character (\w) or whitespace (\s)
            s = s.str.replace(r"[^\w\s]", "", regex=True)

        if remove_numbers:
            # Replaces any digit
            s = s.str.replace(r"\d+", "", regex=True)

        # 3. Whitespace Management
        if collapse_spaces:
            # Replaces 2 or more spaces with a single space
            s = s.str.replace(r"\s{2,}", " ", regex=True)

        if strip_whitespace:
            s = s.str.strip()

        # Reassign the cleaned series back to the original DataFrame
        df_clean.loc[not_null_mask, col] = s

    return df_clean


# ==========================================
#           CARDINALITY REDUCTION
# ==========================================


def summarize_cardinality(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Calculates the number of unique categories for specified text/categorical columns.
    """
    summary = []
    for col in columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            summary.append({"Feature": col, "Unique Categories": unique_count})
    return pd.DataFrame(summary).sort_values(by="Unique Categories", ascending=False)


def reduce_cardinality(
    df: pd.DataFrame,
    columns: list,
    method: str = "frequency",
    threshold_percent: float = 0.01,
    top_n: int = 10,
    substring: str = "",
    replacement_label: str = "Other",
) -> pd.DataFrame:
    """
    Consolidates high-cardinality categorical features using multiple strategies.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): Columns to reduce.
        method (str): Reduction strategy ('frequency', 'top_n', or 'substring').
        threshold_percent (float): Minimum frequency percentage (for 'frequency').
        top_n (int): Number of top categories to keep (for 'top_n').
        substring (str): The text pattern to match and group (for 'substring').
        replacement_label (str): The string to replace grouped labels with.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns:
            continue

        # Create mask to ignore nulls (we don't want to group NaNs into "Other")
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
            # Replace anything NOT in the top labels (and not null) with the replacement
            replace_mask = not_null_mask & ~df_clean[col].isin(top_labels)
            df_clean.loc[replace_mask, col] = replacement_label

        elif method == "substring":
            if substring:
                # Case-insensitive substring match
                contains_mask = (
                    df_clean[col]
                    .astype(str)
                    .str.contains(substring, case=False, na=False)
                )
                df_clean.loc[contains_mask, col] = replacement_label

    return df_clean


# ==========================================
#           MEMORY OPTIMIZATION
# ==========================================


def optimize_memory_usage(
    df: pd.DataFrame,
    downcast_integers: bool = True,
    downcast_floats: bool = True,
    categorize_strings: bool = True,
) -> pd.DataFrame:
    """
    Reduces the memory footprint of a DataFrame by downcasting numeric types
    and converting low-cardinality string columns to categoricals.

    Args:
        df (pd.DataFrame): The input DataFrame.
        downcast_integers (bool): Whether to downcast int64 to int32, int16, or int8.
        downcast_floats (bool): Whether to downcast float64 to float32.
        categorize_strings (bool): Whether to convert object types to categories.

    Returns:
        pd.DataFrame: Memory-optimized DataFrame.
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        col_type = df_clean[col].dtype

        # 1. Optimize Integers
        if downcast_integers and pd.api.types.is_integer_dtype(col_type):
            c_min = df_clean[col].min()
            c_max = df_clean[col].max()
            # Check which integer type can safely hold the min and max values
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_clean[col] = df_clean[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_clean[col] = df_clean[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_clean[col] = df_clean[col].astype(np.int32)

        # 2. Optimize Floats
        elif downcast_floats and pd.api.types.is_float_dtype(col_type):
            c_min = df_clean[col].min()
            c_max = df_clean[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_clean[col] = df_clean[col].astype(np.float32)

        # 3. Optimize Strings (Categoricals)
        elif categorize_strings and pd.api.types.is_object_dtype(col_type):
            num_unique = df_clean[col].nunique()
            num_total = len(df_clean[col])
            # If the number of unique values is less than 50% of the total rows,
            # converting to category saves memory.
            if num_unique / num_total < 0.5:
                df_clean[col] = df_clean[col].astype("category")

    return df_clean


# ==========================================
#               DATA IMPUTATION
# ==========================================


def impute_numerical_features(
    df: pd.DataFrame, columns: list, strategy: str = "median"
) -> pd.DataFrame:
    """
    Selectively imputes missing values in specified numerical columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numerical column names to impute.
        strategy (str): Imputation method ('median', 'mean', 'mode', or 'zero').

    Returns:
        pd.DataFrame: DataFrame with specified numerical columns imputed.
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
    df: pd.DataFrame, columns: list, strategy: str = "Unknown"
) -> pd.DataFrame:
    """
    Selectively imputes missing values in specified categorical or text columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of categorical column names to impute.
        strategy (str): Imputation method ('unknown', 'mode', or 'forward fill').

    Returns:
        pd.DataFrame: DataFrame with specified categorical columns imputed.
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


# ==========================================
#              OUTLIER TREATMENT
# ==========================================


def handle_outliers(
    df: pd.DataFrame,
    columns: list,
    method: str = "iqr",
    action: str = "cap",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    percentile_range: tuple = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Detects and handles numerical outliers using various statistical methods.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numeric column names to process.
        method (str): Detection method ('iqr', 'zscore', 'percentile').
        action (str): What to do with outliers ('drop', 'cap', 'nan').
        iqr_multiplier (float): Multiplier for IQR bounds.
        zscore_threshold (float): Number of standard deviations for Z-score.
        percentile_range (tuple): (lower_percentile, upper_percentile) for clipping.

    Returns:
        pd.DataFrame: DataFrame with outliers processed.
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(
            df_clean[col]
        ):
            continue

        # 1. Calculate Mathematical Bounds
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

        # 2. Apply the Selected Action
        if action == "drop":
            # Keep rows that are strictly within the bounds (or are NaN, so we don't accidentally drop NaNs here)
            valid_mask = (
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ) | df_clean[col].isna()
            df_clean = df_clean[valid_mask]
        elif action == "cap":
            # Winsorization: Clip extreme values to the exact boundary limits
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        elif action == "nan":
            # Replace extreme values with NaN to be imputed later
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
    """
    Calculates the distribution of outliers in specified columns without altering the underlying data.

    Acts as a dry-run profiler to evaluate the impact of selected statistical thresholds.
    This allows the user interface to dynamically render preview tables before executing
    destructive operations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numeric column names to analyze.
        method (str, optional): Statistical detection method ('iqr', 'zscore', or 'percentile'). Defaults to 'iqr'.
        iqr_multiplier (float, optional): Multiplier for the Interquartile Range bounds. Defaults to 1.5.
        zscore_threshold (float, optional): Standard deviation threshold for Z-score bounds. Defaults to 3.0.
        percentile_range (tuple, optional): Tuple of (lower_percentile, upper_percentile). Defaults to (0.01, 0.99).

    Returns:
        pd.DataFrame: A summary table containing the feature name, raw outlier count, and percentage of total data affected.
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

        summary.append({"Feature": col, "Outlier Count": outliers, "% of Data": pct})

    return pd.DataFrame(summary)


# ==========================================
#           SKEWNESS TRANSFORMATION
# ==========================================


def summarize_skewness(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Calculates the skewness coefficient for numerical columns and categorizes the severity.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numerical column names to analyze.

    Returns:
        pd.DataFrame: A summary table with Skewness scores and diagnostic labels.
    """
    summary = []
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            skew_val = df[col].skew()

            # Skip columns where skewness cannot be calculated (e.g., all identical values)
            if pd.isna(skew_val):
                continue

            # Categorize severity based on standard statistical thresholds
            if abs(skew_val) > 1.0:
                severity = "Highly Skewed"
            elif abs(skew_val) > 0.5:
                severity = "Moderately Skewed"
            else:
                severity = "Symmetric"

            summary.append(
                {"Feature": col, "Skewness": skew_val, "Diagnosis": severity}
            )

    return pd.DataFrame(summary)


def fix_numerical_skewness(
    df: pd.DataFrame, columns: list, method: str = "yeo-johnson"
) -> pd.DataFrame:
    """
    Applies mathematical transformations to correct distribution skewness.
    Includes defensive checks to prevent mathematical domain errors and
    uses boolean masking to safely handle missing values (NaNs) without
    triggering length mismatch errors.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): Columns to transform.
        method (str): Transformation method ('yeo-johnson', 'box-cox', 'log1p', 'sqrt').

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(
            df_clean[col]
        ):
            continue

        # Create a boolean mask to safely ignore NaNs during transformation
        not_null_mask = df_clean[col].notna()

        # If the column is entirely nulls, skip it
        if not not_null_mask.any():
            continue

        min_val = df_clean.loc[not_null_mask, col].min()

        # 1. Log1p Transformation (log(1+x))
        if method == "log1p":
            if min_val < -1:
                raise ValueError(
                    f"Feature '{col}' contains values < -1. Log1p requires values >= -1."
                )
            df_clean.loc[not_null_mask, col] = np.log1p(
                df_clean.loc[not_null_mask, col]
            )

        # 2. Square Root Transformation
        elif method == "sqrt":
            if min_val < 0:
                raise ValueError(
                    f"Feature '{col}' contains negative values. Square Root requires positive values."
                )
            df_clean.loc[not_null_mask, col] = np.sqrt(df_clean.loc[not_null_mask, col])

        # 3. Box-Cox Power Transformation
        elif method == "box-cox":
            if min_val <= 0:
                raise ValueError(
                    f"Feature '{col}' contains zero or negative values. Box-Cox requires strictly positive data (>0). Use Yeo-Johnson instead."
                )
            # Apply only to non-null slice and assign back to the exact non-null slice
            transformed_data, _ = boxcox(df_clean.loc[not_null_mask, col])
            df_clean.loc[not_null_mask, col] = transformed_data

        # 4. Yeo-Johnson Power Transformation (Safest)
        elif method == "yeo-johnson":
            # Safely handles positive, zero, and negative values while ignoring NaNs
            transformed_data, _ = yeojohnson(df_clean.loc[not_null_mask, col])
            df_clean.loc[not_null_mask, col] = transformed_data

    return df_clean


# ==========================================
#           SCALING & ENCODING
# ==========================================


def scale_numerical_features(
    df: pd.DataFrame, columns: list, method: str = "standard"
) -> pd.DataFrame:
    """
    Scales numerical features using various statistical distributions.
    Safely ignores missing values (NaNs) during calculation to prevent crashes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): Numerical columns to scale.
        method (str): 'standard', 'minmax', 'robust', or 'maxabs'.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df_clean = df.copy()
    if not columns:
        return df_clean

    # Map string method to the actual scikit-learn object
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
            # Create a mask to scale ONLY non-null values
            not_null_mask = df_clean[col].notna()

            if not_null_mask.any():
                # scikit-learn expects 2D arrays, hence [[col]] reshaping
                scaled_data = scaler.fit_transform(df_clean.loc[not_null_mask, [col]])
                # Flatten back to 1D and assign
                df_clean.loc[not_null_mask, col] = scaled_data.flatten()

    return df_clean


def encode_categorical_features(
    df: pd.DataFrame, columns: list, method: str = "onehot"
) -> pd.DataFrame:
    """
    Converts text categories into machine-readable numerical formats.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): Categorical columns to encode.
        method (str): 'onehot' (dummy variables) or 'label' (ordinal integers).

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df_clean = df.copy()
    if not columns:
        return df_clean

    valid_cols = [col for col in columns if col in df_clean.columns]

    if method == "onehot":
        # Use pandas built-in dummy encoding (automatically drops original columns)
        # dtype=int ensures output is 0/1 integers instead of True/False booleans
        df_clean = pd.get_dummies(df_clean, columns=valid_cols, dtype=int)

    elif method == "label":
        for col in valid_cols:
            not_null_mask = df_clean[col].notna()
            # Factorize converts unique labels into integers (0, 1, 2...)
            labels, _ = pd.factorize(df_clean.loc[not_null_mask, col])
            df_clean.loc[not_null_mask, col] = labels

    return df_clean
