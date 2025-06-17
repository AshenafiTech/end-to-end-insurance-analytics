import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column, bins=50):
    df[column].hist(bins=bins)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, column):
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()