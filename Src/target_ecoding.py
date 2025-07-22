import numpy as np
# import pandas as pd
from sklearn.model_selection import StratifiedKFold
from category_encoders import TargetEncoder


def leakage_free_target_encoding(
    train_df,
    test_df,
    target_col,
    cat_cols,
    seed,
    smoothing=100,
    n_splits=5):
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
        train_encoded[f'{col}_te'] = np.nan
        oof_encoded = np.empty(len(train_encoded))
        oof_encoded[:] = np.nan
        
        # K-Fold cross-validation for leakage-free encoding
        for train_idx, val_idx in kf.split(train_encoded, train_df[target_col]):
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]
            val_index = train_df.index[val_idx]

            # Fit encoder on the training fold and transform the validation fold
            encoder = TargetEncoder(cols=[col], smoothing=smoothing)
            encoder.fit(train_fold[[col]], train_fold[target_col])
            oof_encoded[val_index] = encoder.transform(val_fold[[col]]).values.ravel()
        
        # Assign out-of-fold encodings to the training data
        train_encoded[f'{col}_te'] = oof_encoded
        
        # Fit encoder on full original training data
        final_encoder = TargetEncoder(cols=[col], smoothing=smoothing)
        final_encoder.fit(train_df[[col]], train_df[target_col])
        test_encoded[f'{col}_te'] = final_encoder.transform(test_df[[col]]).values.ravel()
    
    return train_encoded, test_encoded