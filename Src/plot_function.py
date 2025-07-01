import seaborn as sns
import matplotlib.pyplot as plt


def plot_violin_by_binary_category(data, binary_col, numeric_col, title = None, palette = 'pastel',
                                   x_labels = None, show_quartiles = True):
    """
    Plots a violin plot showing the distribution of a numerical variable
    by a binary categorical variable (e.g., 0/1 or 'No'/'Yes').

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - binary_col (str): Name of the binary categorical column (e.g., 'is_fraud').
    - numeric_col (str): Name of the numeric column to visualize.
    - title (str, optional): Optional title for the plot. Defaults to None, generates a default title.
    - palette (str, optional): Color palette for the violin plot. Defaults to 'pastel'.
    - x_labels (list, optional): Custom labels for x-axis ticks (list of two strings).
                                 Defaults to None. If None and binary_col is 0/1,
                                 defaults to ['Not Fraud', 'Fraud'].
    - show_quartiles (bool, optional): Whether to show quartile lines inside violins.
                                       If False, shows a default box plot. Defaults to True.
    """
    plt.figure(figsize=(10, 6))

    # Create violin plot with improved formatting
    ax = sns.violinplot(
        x=binary_col,
        y=numeric_col,
        data=data,
        hue=binary_col,
        palette=palette,
        legend=False,
        inner='quartile' if show_quartiles else 'box',
        cut=0
    )

    # Format labels
    xlabel = binary_col.replace("_", " ").title()
    ylabel = numeric_col.replace("_", " ").title()

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Set title
    if not title:
        title = f'Distribution of {ylabel} by {xlabel}'
    plt.title(title, fontsize=14, pad=15)

    # Custom x-axis labels
    if x_labels:
        ax.set_xticks(range(len(x_labels))) # Explicitly set tick locations
        ax.set_xticklabels(x_labels)
    elif data[binary_col].nunique() == 2 and sorted(data[binary_col].unique()) == [0, 1]:
        # For binary 0/1, ticks are typically at 0 and 1
        ax.set_xticks([0, 1]) # Explicitly set tick locations for 0 and 1
        ax.set_xticklabels(['Not Fraud', 'Fraud'])

    
    # Improve grid and layout
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.gcf().set_facecolor('#f8f9fa')
    sns.despine(left=True)

    # Add data count annotation (updated placement logic)
    min_y, max_y = ax.get_ylim()
    y_text_position = min_y + (max_y - min_y) * 0.01

    for i, category in enumerate(data[binary_col].unique()):
        count = (data[binary_col] == category).sum()
        ax.text(i, y_text_position, f'n={count:,}',
                ha='center', va='bottom', fontsize=9, color='dimgray')

    plt.tight_layout()
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