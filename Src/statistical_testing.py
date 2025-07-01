import pandas as pd
from scipy.stats import chi2_contingency

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