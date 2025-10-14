import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import skew 
from sklearn.preprocessing import StandardScaler


def chi_square_test(data, col1, col2, print_results=True):
    """
    Performs a Chi-Square Test for Independence between two categorical variables.
    Null Hypothesis: There is no association between the two categorical variables.

    Parameters:
    - df: pandas DataFrame
    - col1: str, name of the first categorical column
    - col2: str, name of the second categorical column
    - print_results: bool, if True, prints the test summary
    """
    # Create contingency table (counts)
    contingency_table = pd.crosstab(data[col1], data[col2])
    # Normalize the contingency table to get percentages
    contingency_table_norm = pd.crosstab(data[col1], data[col2], normalize='columns') * 100


    # Rename columns for clarity
    contingency_table_counts = contingency_table.add_suffix(' (count)')
    contingency_table_perc = contingency_table_norm.add_suffix(' (%)')

    # Concatenate along columns
    combined_table = pd.concat([contingency_table_counts, contingency_table_perc], axis=1)

    # Chi-square test
    chi2_stat, p_val, dof, expected_freqs = chi2_contingency(contingency_table)

    if print_results:
        print(f"--- Chi-Square Test between '{col1}' and '{col2}' ---")
        print("Contingency Table (Observed):")
        print(combined_table.to_string(float_format='%.2f'))  # Format for better readability
        print("\nExpected Frequencies:")
        print(pd.DataFrame(expected_freqs, index=contingency_table.index, columns=contingency_table.columns))
        print(f"\nChi-square Statistic = {chi2_stat:.4f}")
        print(f"Degrees of Freedom   = {dof}")
        print(f"P-value              = {p_val:.4f}")
    if p_val < 0.05:
        print("➡️ Statistically significant association (Reject Null Hypothesis)")
    else:
        print("➡️ Not statistically significant (Fail to Reject Null Hypothesis)")

    return  contingency_table_norm


def gaussian_mixture_binning(data, colum_list, seed, n_init=10):
    """
    Fits a Gaussian Mixture Model (GMM) with different numbers of components 
    after optionally applying log transformation for skewness and then 
    normalizing the input data. Uses AIC and BIC to determine the 
    optimal number of components and visualizes the results.
    """
    # Initialize lists and define component range
    aic = [] # AIC (Akaike Information Criterion) - Lower the Better
    bic = [] # BIC (Bayesian Information Criterion) - Lower the Better
    components_range = range(1, 15)  # 1 to 14 components
    
    # Prepare Data (Select columns and remove NaNs)
    prepared_data = data[colum_list].copy().dropna()

    if prepared_data.empty:
        print("Error: Data is empty after selecting columns or dropping NaNs.")
        return

    # Check for and correct extreme skewness (Highly recommended for GMM)
    # Using log1p transformation for columns with high absolute skewness
    skewness_threshold = 0.5
    print("--- Skewness Check & Log Transformation ---")
    for col in colum_list:
        # Check skewness on the original, prepared data
        current_skew = skew(prepared_data[col])
        
        if abs(current_skew) > skewness_threshold:
            # Apply log1p transformation (robust to non-negative values)
            min_val = prepared_data[col].min()
            if min_val < 0:
                # Shift data to be non-negative before log transformation
                prepared_data[col] = np.log1p(prepared_data[col] - min_val)
            else:
                prepared_data[col] = np.log1p(prepared_data[col])
            print(f"Applied log1p to '{col}' (Raw Skew: {current_skew:.2f})")
        else:
            print(f"'{col}' skipped log transform (Raw Skew: {current_skew:.2f})")
    print("------------------------------------------")
    
    # Apply Normalization (Standard Scaling)
    # GMM is sensitive to feature scaling; standardizing ensures equal weight.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(prepared_data)
    
    # Convert back to DataFrame for easy handling (optional, but good practice)
    scaled_df = pd.DataFrame(scaled_data, columns=colum_list)

    # Fit GMM for each component count
    for n in components_range:
        gmm = GaussianMixture(n_components=n, n_init=n_init, random_state=seed)
        
        # Fit and score using the SCALED data
        gmm.fit(scaled_df) 
        
        aic.append(gmm.aic(scaled_df))
        bic.append(gmm.bic(scaled_df))
    
    # 5. Plot AIC and BIC to find the optimal number of components
    plt.figure(figsize=(8,5))
    plt.plot(components_range, aic, label='AIC', marker='o')
    plt.plot(components_range, bic, label='BIC', marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC / BIC')
    plt.title('AIC and BIC for GMM on Scaled Data')
    plt.legend()
    plt.grid(True)
    plt.show()


def discretization(data, feature, newFeature, qcut, labelTxt):
    # use quartile bin
    _, bins = pd.qcut(data[feature].dropna(), q=qcut, retbins=True, precision=0)

    # create custom labels
    labels = [f'{labelTxt}({int(bins[i])}-{int(bins[i+1])})' for i in range(len(bins)-1)]
    
    # create the categorical column, initially with NaN for missing values
    data[newFeature] = pd.cut(data[feature], bins=bins, labels=labels, include_lowest=True)
    
    # replace NaN with 'Unknown'
    data[newFeature] = data[newFeature].cat.add_categories('Unknown').fillna('Unknown')

    # Remove any categories that do not have any observations after discretization and NaN handling.
    data[newFeature] = data[newFeature].cat.remove_unused_categories()

    return data