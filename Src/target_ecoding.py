import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from category_encoders import TargetEncoder


def leakage_free_target_encoding(
    train_df,
    test_df,
    target_col,
    cat_cols,
    seed,
    smoothing=100,
    n_splits=5,
):
    """
    Leakage-free target encoding with K-Fold cross-validation and smoothing.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Testing dataset.
    target_col : str
        Name of the target column.
    cat_cols : list of str
        List of categorical columns to encode.
    smoothing : float
        Smoothing factor to regularize encoding.
    n_splits : int
        Number of K-Folds.
    random_state : int
        Random seed for reproducibility.
    
    Returns:
    --------
    train_encoded : pd.DataFrame
        Training dataset with new encoded features.
    test_encoded : pd.DataFrame
        Testing dataset with new encoded features.
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Set up K-Fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Iterate over each categorical column for encoding
    for col in cat_cols:
        # Initialize column for out-of-fold encodings in train_encoded
        train_encoded[f'{col}_te'] = np.nan
        
        # Initialize array to hold out-of-fold encodings efficiently
        oof_encoded = np.empty(len(train_encoded))
        oof_encoded[:] = np.nan # Ensure it's filled with NaN initially
        
        # K-Fold cross-validation for leakage-free encoding
        for train_idx, val_idx in kf.split(train_encoded):
            train_fold = train_encoded.iloc[train_idx]
            val_fold = train_encoded.iloc[val_idx] # Keep as DataFrame for iloc
            
            encoder = TargetEncoder(cols=[col], smoothing=smoothing)
            
            # Fit on train_fold[[col]] (DataFrame) and train_fold[target_col] (Series)
            encoder.fit(train_fold[[col]], train_fold[target_col])
            
            # Transform val_fold[[col]] (DataFrame)
            oof_encoded[val_idx] = encoder.transform(val_fold[[col]]).values.ravel()
        
        train_encoded[f'{col}_te'] = oof_encoded
        
        # Fit encoder on full train data for test set
        final_encoder = TargetEncoder(cols=[col], smoothing=smoothing)
        # Fit on train_encoded[[col]] (DataFrame) and train_encoded[target_col] (Series)
        final_encoder.fit(train_encoded[[col]], train_encoded[target_col])
        
        # Transform test_encoded[[col]] (DataFrame)
        test_encoded[f'{col}_te'] = final_encoder.transform(test_encoded[[col]]).values.ravel()
    
    return train_encoded, test_encoded



def calculate_smoothed_avg_fraud_amount(dataframe, category_col, amount_col, target_col, smoothing_parameter):
    """
    Calculates a smoothed average monetary amount of ONLY FRAUDULENT transactions
    for each category.
    """
    fraud_df = dataframe[dataframe[target_col] == 1].copy()

    if fraud_df.empty:
        return pd.Series(index=dataframe[category_col].unique(), dtype=float), 0.0
    

    print(fraud_df)

    global_avg_fraud_amount = fraud_df[amount_col].mean()
    if pd.isna(global_avg_fraud_amount):
        global_avg_fraud_amount = 0.0

    category_fraud_stats = fraud_df.groupby(category_col, observed=False)[amount_col].agg(['count', 'sum'])
    category_fraud_stats.rename(columns={'count': 'fraud_count', 'sum': 'fraud_amount_sum'}, inplace=True)

    category_fraud_stats['raw_avg_fraud_amount'] = np.where(
        category_fraud_stats['fraud_count'] > 0,
        category_fraud_stats['fraud_amount_sum'] / category_fraud_stats['fraud_count'],
        0.0
    )

    numerator = (category_fraud_stats['fraud_count'] * category_fraud_stats['raw_avg_fraud_amount']) + \
                (smoothing_parameter * global_avg_fraud_amount)
    denominator = category_fraud_stats['fraud_count'] + smoothing_parameter

    smoothed_avg_fraud_amount = np.where(
        denominator > 0,
        numerator / denominator,
        global_avg_fraud_amount
    )
    
    smoothed_avg_fraud_amount_series = pd.Series(smoothed_avg_fraud_amount, index=category_fraud_stats.index)

    return smoothed_avg_fraud_amount_series, global_avg_fraud_amount


def calculate_smoothed_weighted_fraud_rate(dataframe, category_col, target_col, weight_col, smoothing_parameter):
    """
    Calculates a smoothed fraud rate weighted by a specified column (e.g., 'amount' or 'severity_score').
    """
    df_temp = dataframe.copy()
    df_temp[target_col] = pd.to_numeric(dataframe[target_col], errors='coerce') 
    
    total_fraud_amount_global = df_temp[df_temp[target_col] == 1][weight_col].sum()
    total_amount_global = df_temp[weight_col].sum()
    
    global_weighted_rate = 0.0
    if total_amount_global > 0:
        global_weighted_rate = total_fraud_amount_global / total_amount_global
    
    category_weighted_fraud_sum = df_temp[df_temp[target_col] == 1].groupby(category_col, observed=False)[weight_col].sum()
    category_total_weight_sum = df_temp.groupby(category_col, observed=False)[weight_col].sum()

    category_stats = pd.DataFrame({
        'weighted_fraud_sum': category_weighted_fraud_sum,
        'total_weight_sum': category_total_weight_sum
    }).fillna(0)

    category_stats['raw_weighted_rate'] = np.where(
        category_stats['total_weight_sum'] > 0,
        category_stats['weighted_fraud_sum'] / category_stats['total_weight_sum'],
        0.0
    )

    category_stats['category_count'] = df_temp.groupby(category_col, observed=False).size()

    numerator = (category_stats['category_count'] * category_stats['raw_weighted_rate']) + \
                (smoothing_parameter * global_weighted_rate)
    denominator = category_stats['category_count'] + smoothing_parameter

    smoothed_weighted_rates = np.where(
        denominator > 0,
        numerator / denominator,
        global_weighted_rate
    )
    
    smoothed_weighted_rates_series = pd.Series(smoothed_weighted_rates, index=category_stats.index)

    return smoothed_weighted_rates_series, global_weighted_rate



def leakage_free_amount_target_encoding(
    train_df,
    test_df,
    target_col,
    amount_col, # Column for transaction amount or severity score
    cat_cols,
    random_state, # Renamed from 'seed' for consistency with sklearn
    encoding_type='avg_fraud_amount', # 'avg_fraud_amount' or 'weighted_fraud_rate'
    smoothing=100,
    n_splits=5
):
    """
    Leakage-free amount-based target encoding with K-Fold cross-validation and smoothing.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset. Must include target_col and amount_col.
    test_df : pd.DataFrame
        Testing dataset. Must include amount_col (if needed for context, but not for encoding).
    target_col : str
        Name of the target column ('is_fraud').
    amount_col : str
        Name of the column to use as amount/weight ('amount' or 'severity_score').
    cat_cols : list of str
        List of categorical columns to encode.
    random_state : int
        Random seed for reproducibility (used for KFold shuffling).
    encoding_type : str
        Type of encoding to perform: 'avg_fraud_amount' or 'weighted_fraud_rate'.
    smoothing : float
        Smoothing factor.
    n_splits : int
        Number of K-Folds for OOF encoding.

    Returns:
    --------
    train_encoded : pd.DataFrame
        Training dataset with new encoded features.
    test_encoded : pd.DataFrame
        Testing dataset with new encoded features.
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    # Set up K-Fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Determine which calculation function to use
    calculator_func = None
    if encoding_type == 'avg_fraud_amount':
        calculator_func = calculate_smoothed_avg_fraud_amount
    elif encoding_type == 'weighted_fraud_rate':
        calculator_func = calculate_smoothed_weighted_fraud_rate
    else:
        raise ValueError("encoding_type must be 'avg_fraud_amount' or 'weighted_fraud_rate'")

    # Iterate over each categorical column for encoding
    for col in cat_cols:
        new_col_name = f'{col}_{encoding_type}_te'
        train_encoded[new_col_name] = np.nan # Initialize with NaN for OOF values
        
        # OOF encoding for training data (prevents in-fold leakage)
        for fold_train_idx, fold_val_idx in kf.split(train_encoded):
            fold_train_df = train_encoded.iloc[fold_train_idx]
            
            # Calculate mappings using only the current mini-training fold (no leakage from fold_val_idx)
            category_mapping, _ = calculator_func(
                fold_train_df, col, target_col, amount_col, smoothing
            )
            
            # Apply mapping to the current mini-validation fold
            # (Note: .map() returns NaN for categories not in mapping)
            train_encoded.loc[fold_val_idx, new_col_name] = train_encoded.iloc[fold_val_idx][col].map(category_mapping)

        # After the OOF loop, some categories in train_encoded might still be NaN
        # if they never appeared in any fold_train_df (e.g., very rare categories).
        # We need the global default from the *entire* training set for these.
        
        # Calculate final mapping and global default using the entire training data
        final_category_mapping, global_default_value = calculator_func(
            train_encoded, col, target_col, amount_col, smoothing
        )
        
        # FIX: Assign the result of fillna back to the column
        train_encoded[new_col_name] = train_encoded[new_col_name].fillna(global_default_value)
        
        # Apply to test data, filling unseen categories with the global default from train data
        test_encoded[new_col_name] = test_encoded[col].map(final_category_mapping).fillna(global_default_value)
    
    return train_encoded, test_encoded