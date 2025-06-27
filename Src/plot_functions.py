# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# plot feature correlation with target variable
def plot_feature_correlation(data, numerical_cols, target_col):
    """
    Calculates and plots the Pearson correlation of specified numerical features
    with a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing numerical features and the target.
                           This DataFrame should be your 'stratified_sample' or similar.
        numerical_cols (list): A list of column names in `df` that are numerical features
                               for which to calculate correlations.
        target_col (str): The name of the target variable column in `df`.
    """
    # create a new list for the correlation matrix to avoid modifying the original numerical_cols.
    corr_cols_for_matrix = list(numerical_cols)
    if target_col not in corr_cols_for_matrix:
        corr_cols_for_matrix.append(target_col)

    # calculate correlations of all features with the target variable
    correlation_matrix = data[corr_cols_for_matrix].corr()
    correlations_with_target = correlation_matrix[target_col].drop(target_col)

    # sort correlations for better visualization
    correlations_with_target = correlations_with_target.sort_values(ascending=False)

    # Plotting
    plt.figure(figsize=(12, max(6, len(correlations_with_target) * 0.4))) # adjust height based on num features
    sns.barplot(y=correlations_with_target.index,
                x=correlations_with_target.values,
                hue=correlations_with_target.index,
                palette='viridis') # 'viridis' sequential palette

    plt.title(f'Feature Correlation with Target ({target_col})', fontsize=16)
    plt.ylabel('Features', fontsize=12)
    plt.xlabel('Pearson Correlation Coefficient', fontsize=12)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8) # Add a vertical line at 0 for clarity
    plt.grid(axis='x', linestyle='--', alpha=0.7) # add horizontal grid lines
    plt.tight_layout() # adjust layout to prevent labels from overlapping
    plt.show()



def plot_histogram(data, lst, bins=30, txt='', title_font=15, label_font=10, tick_font=10, KDE=True):
    """
    Plot histograms with optional KDE for multiple columns in a single figure.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing the data to plot.
    lst : list of str
        List of column names in `data` to plot histograms for.
    bins : int, optional (default=30)
        Number of bins for the histogram.
    txt : str, optional (default='')
        Additional text to append to each subplot title.
    title_font : int, optional (default=15)
        Font size for subplot titles.
    label_font : int, optional (default=10)
        Font size for x and y axis labels.
    tick_font : int, optional (default=10)
        Font size for x and y axis tick labels.
    KDE : bool, optional (default=True)
        Whether to overlay a Kernel Density Estimate (KDE) curve on the histogram.
    """
    # calculate the number of rows needed (two plots per row)
    num_cols = min(len(lst), 2)  # max 2 columns
    num_rows = int(np.ceil(len(lst) / num_cols))  # calculate number of rows
    
    # set up the matplotlib figure
    plt.figure(figsize=(10 * num_cols, 5 * num_rows))  # adjust the figure size as needed
    
    # iterate each categorical column and create a subplot
    for i, column in enumerate(lst):
        plt.subplot(num_rows, num_cols, i + 1)  # create a subplot for each numeric
        ax = sns.histplot(data=data, x=column, kde=KDE, bins=bins)
        ax.grid(False)  # remove grid
        
        # customize each plot
        plt.title(f'Histogram Plot for {column} {txt}', fontsize=title_font)
        plt.xlabel(column, fontsize=label_font, fontweight='bold')
        plt.ylabel('Frequency', fontsize=label_font, fontweight='bold')
        plt.xticks(fontsize=tick_font)
        plt.yticks(fontsize=tick_font)
    
    # adjust the space between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # show the plot
    plt.tight_layout()
    plt.show()



def plot_box(data, columns, orientation='h', fig_size=(15, 5)):
    """
    Plots boxplots for specified columns in a DataFrame.

    Parameters:
    data (DataFrame): The DataFrame containing the data to plot.
    columns (list): A list of column names to plot.
    orientation (str): Orientation of the boxplots - 'h' for horizontal, 'v' for vertical. Default is 'h'.
    figsize (tuple): Size of the figure as (width, height). Default is (15, 5).
    """
    # set the figure size
    plt.figure(figsize=fig_size)
    
    # determine the number of rows needed for subplots
    num_columns = len(columns)
    num_rows = (num_columns + 1) // 2
    
    # iterate through each specified column and create a boxplot
    for i, column in enumerate(columns, 1):
        plt.subplot(num_rows, 2, i)
        if orientation == 'v':
            sns.boxplot(y=data[column])
        else:
            sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.tight_layout()
    
    # display the plots
    plt.show()



def plot_distribution(data, x_col, hue_col, log_scale=True, bins=50):
    """
    Generates side-by-side plots (histogram and KDE) showing the distribution of
    transaction amounts by fraud status on a log scale.

    Args:
        data (pd.DataFrame): The input DataFrame containing transaction data.
        x_col (str): The name of the column containing transaction amounts.
        hue_col (str): The name of the column indicating fraud status (e.g., 'isFraud').
        bins (int, optional): The number of bins for the histogram. Defaults to 50.
    """
    # filter out non-positive values for log scale
    subset = data[data[x_col] > 0].copy() # .copy() to avoid SettingWithCopyWarning

    # side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Histogram
    sns.histplot(
        data=subset,
        x=x_col,
        hue=hue_col,
        bins=bins,
        stat='density',
        element='bars',
        common_norm=False,
        log_scale=log_scale,
        ax=axes[0]
    )
    axes[0].set_title(f'Histogram of {x_col} by {hue_col} Status')
    if log_scale:
        axes[0].set_xlabel(f'{x_col} (Log Scale)')
    else:
        axes[0].set_xlabel(f'{x_col}')
    axes[0].set_ylabel('Density')
    axes[0].grid(True)

    # Right plot: KDE
    sns.kdeplot(
        data=subset,
        x=x_col,
        hue=hue_col,
        log_scale=True,
        common_norm=False,
        fill=True,
        linewidth=2,
        ax=axes[1]
    )
    axes[1].set_title(f'KDE of {x_col} by {hue_col} Status')
    if log_scale:
        axes[1].set_xlabel(f'{x_col} (Log Scale)')
    else:
        axes[1].set_xlabel(f'{x_col}')
    axes[1].set_ylabel('Density')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()