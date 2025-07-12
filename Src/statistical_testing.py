import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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
    This function is designed to fit a Gaussian Mixture Model (GMM) with different numbers of 
    components (clusters) and use information criteria (AIC and BIC) to determine the optimal 
    number of components. It then visualizes the results using a plot to help identify the best 
    number of components for the GMM.
    """
    # initialize fit GMM with different number of components and select the best using AIC or BIC
    aic = [] # AIC (Akaike Information Criterion)    Lower the Better
    bic = [] # BIC (Bayesian Information Criterion)  Lower the Better
    components_range = range(1, 11)  # 1 to 10 components
    # remove any NaNs
    data = data[colum_list].dropna()
    
    for n in components_range:
        gmm = GaussianMixture(n_components=n, n_init=n_init, random_state=seed)
        gmm.fit(data[colum_list])
        aic.append(gmm.aic(data[colum_list]))
        bic.append(gmm.bic(data[colum_list]))
    
    # plot AIC and BIC to find the optimal number of components
    plt.plot(components_range, aic, label='AIC')
    plt.plot(components_range, bic, label='BIC')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC/BIC')
    plt.legend()
    plt.title('AIC and BIC for GMM')
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

    return data