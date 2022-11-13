import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def split_train_test(data: pd.DataFrame, test_size=0.2):
    train, test = train_test_split(data, test_size=test_size)
    return train, test


def read_csv_data(csv_file_path: str):
    return pd.read_csv(csv_file_path)


def save_to_csv(data: pd.DataFrame, path: str):
    data.to_csv(path, index=False)


def show_histogram(data: pd.DataFrame, attribute_name: str, bins: int = 25):
    data.hist(column=attribute_name, bins=bins, grid=False, figsize=(12, 8), color='#86bf91', zorder=2, rwidth=0.9)
    plt.show()


def show_histograms(data: pd.DataFrame):
    params = {"color": '#86bf91', "zorder": 2, "rwidth": 0.9, "grid": False}
    fig, axes = plt.subplots(2, 3)
    data.hist(column="no_of_trainings", bins=5, ax=axes[0][0], **params)
    data.hist(column="age", bins=25, ax=axes[0][1], **params)
    data.hist(column="previous_year_rating", bins=5, ax=axes[0][2], **params)
    data.hist(column="length_of_service", bins=20, ax=axes[1][0], **params)
    data.hist(column="avg_training_score", bins=20, ax=axes[1][1], **params)

    plt.show()


def show_corr_matrix(data):
    corr = data.corr().round(2)
    sns.set(font_scale=0.7)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        annot=True,
        mask=mask,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()


def remove_missing_values(data: pd.DataFrame):
    return data.dropna()


def reduce_dataset(data: pd.DataFrame, desired_size: int):
    return data.sample(n=desired_size)
