import random
from dataclasses import dataclass
from typing import Iterable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import uniform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration shared by the season notebooks."""

    file_path: str
    csv_sep: str = ";"
    csv_decimal: str = "."
    csv_encoding: str = "latin1"
    min_minutes_played: int = 180
    hopkins_sample_ratio: float = 0.1
    hopkins_n_neighbors: int = 2
    kmeans_n_init: int = 10
    max_clusters: int = 10
    optimal_k: int = 3
    random_seed: int = 2137
    selected_features: tuple[str, ...] = (
        "Shots",
        "PasTotAtt",
        "Assists",
        "Tkl",
        "Blocks",
        "Fls",
        "Off",
        "AerWon%",
    )
    visualization_features: tuple[str, ...] = (
        "Min",
        "90s",
        "Goals",
        "PasTotAtt",
        "AerWon%",
    )


DEFAULT_PLOT_CONFIGS = (
    {
        "column": "Min",
        "title": "Distribution of Minutes Played (90 minutes intervals)",
        "xlabel": "Minutes Played (90 minutes bins)",
        "color": "skyblue",
        "binwidth": 90,
    },
    {
        "column": "total_goals",
        "title": "Distribution of Goals",
        "xlabel": "Goals",
        "color": "red",
        "binwidth": 1,
    },
    {
        "column": "PasTotAtt",
        "title": "Distribution of passes per 90 minutes",
        "xlabel": "Passes",
        "color": "green",
        "binwidth": 5,
    },
    {
        "column": "AerWon%",
        "title": "Distribution of Aerial Duels Won (%)",
        "xlabel": "Aerial Duels Won (%)",
        "color": "orange",
        "binwidth": 5,
    },
)


def configure_notebook(random_seed: int) -> None:
    """Apply deterministic seeds and default plotting style."""

    random.seed(random_seed)
    np.random.seed(random_seed)
    sns.set_theme(style="whitegrid")


def load_player_stats(config: AnalysisConfig) -> pd.DataFrame:
    """Load a season CSV using the shared dataset formatting rules."""

    return pd.read_csv(
        config.file_path,
        sep=config.csv_sep,
        decimal=config.csv_decimal,
        encoding=config.csv_encoding,
    )


def filter_by_minutes(stats: pd.DataFrame, min_minutes_played: int) -> pd.DataFrame:
    """Keep players with enough minutes to reduce small-sample distortion."""

    return stats[stats["Min"] >= min_minutes_played].copy()


def validate_no_missing_values(stats: pd.DataFrame) -> pd.Series:
    """Return missing-value counts and fail if any are present."""

    missing_vals = stats.isna().sum()[stats.isna().sum() > 0]
    assert missing_vals.sum() == 0, "Pipeline Error: Raw data contains missing values!"
    return missing_vals


def select_features(stats: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Create the reduced feature matrix used for clustering."""

    return stats[list(features)].copy()


def prepare_visualization_data(
    stats: pd.DataFrame,
    visualization_features: Iterable[str],
) -> pd.DataFrame:
    """Create the plotting dataframe and derived total-goals column."""

    player_stats = stats[list(visualization_features)].copy()
    # The source CSVs are inconsistent: 2021/22 column `Goals` stores goals per 90, while
    # 2022/23 stores total goals. Avoid multiplying already-total goals by 90s.
    goals_are_per_90 = player_stats["Goals"].max() <= 5
    player_stats["total_goals"] = player_stats["Goals"]
    if goals_are_per_90:
        player_stats["total_goals"] = player_stats["Goals"] * player_stats["90s"]
    return player_stats


def hopkins_statistic(
    data: pd.DataFrame,
    sample_ratio: float = 0.1,
    n_neighbors: int = 2,
) -> float:
    """
    Computes the Hopkins statistic to measure the cluster tendency of a dataset using vectorized operations.

    Parameters:
    data (pd.DataFrame): The input dataset to evaluate.
    sample_ratio (float): The fraction of the dataset to sample (default 0.1).
    n_neighbors (int): The number of nearest neighbors to search for (default 2).

    Returns:
    float: The Hopkins statistic (H). A value near 1 indicates strong clustering tendency,
           while a value near 0.5 indicates uniform data.
    """

    d = data.shape[1]
    n = len(data)
    m = int(sample_ratio * n)

    # Fit NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data.values)

    # Vectorized Generation of 'm' simulated uniform points
    min_vals = np.amin(data.values, axis=0)
    max_vals = np.amax(data.values, axis=0)
    simulated_points = uniform(min_vals, max_vals, (m, d))

    # Vectorized Selection of 'm' real points
    rand_indices = random.sample(range(n), m)
    real_points = data.iloc[rand_indices].values

    # Vectorized Distance Calculation for all points simultaneously
    u_dist, _ = nbrs.kneighbors(simulated_points, n_neighbors=n_neighbors)
    w_dist, _ = nbrs.kneighbors(real_points, n_neighbors=n_neighbors)

    # Extract the distance to the nth neighbor (index n_neighbors - 1)
    # and sum all distances without using any slow sequential Python loops
    sum_ujd = np.sum(u_dist[:, n_neighbors - 1])
    sum_wjd = np.sum(w_dist[:, n_neighbors - 1])

    return float(sum_ujd / (sum_ujd + sum_wjd))


def plot_distribution(
    data: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    color: str,
    binwidth: int | None = None,
) -> None:
    """
    Plots the distribution of a given column in a dataframe to avoid repetitive code.
    """

    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], binwidth=binwidth, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of players")
    plt.show()


def plot_distributions(
    data: pd.DataFrame,
    plot_configs: Iterable[dict] = DEFAULT_PLOT_CONFIGS,
) -> None:
    """Plot the standard exploratory distributions."""

    for config in plot_configs:
        plot_distribution(data=data, **config)


def plot_correlation_matrix(data: pd.DataFrame, title: str) -> None:
    """
    Plots a heatmap for the correlation matrix of the provided dataframe.
    """

    plt.figure(figsize=(10, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()


def scale_features(
    data: pd.DataFrame,
    feature_names: Iterable[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize selected features and return a dataframe plus fitted scaler."""

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data)
    return pd.DataFrame(scaled_array, columns=list(feature_names)), scaler


def evaluate_clusters(
    data: pd.DataFrame,
    max_k: int,
    random_seed: int,
    n_init: int,
) -> pd.DataFrame:
    """
    Evaluates optimal number of clusters using WCSS and Silhouette Score to avoid hardcoding test boundaries.
    """

    rows = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=n_init)
        kmeans.fit(data)
        rows.append(
            {
                "k": k,
                "wcss": kmeans.inertia_,
                "silhouette_score": silhouette_score(data, kmeans.labels_),
            }
        )
    return pd.DataFrame(rows)


def plot_cluster_evaluation(evaluation: pd.DataFrame) -> None:
    """Plot elbow and silhouette metrics from ``evaluate_clusters`` output."""

    # Plotting metrics
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(evaluation["k"], evaluation["wcss"], "bo-", markersize=8)
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("WCSS")

    axes[1].plot(evaluation["k"], evaluation["silhouette_score"], "ro-", markersize=8)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    plt.show()


def fit_kmeans(
    data: pd.DataFrame,
    n_clusters: int,
    random_seed: int,
    n_init: int,
) -> KMeans:
    """Fit the final K-Means model."""

    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=n_init)
    model.fit(data)
    return model


def add_cluster_labels(stats: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach numeric and categorical cluster labels to a copy of the stats."""

    labelled_stats = stats.copy()
    labelled_stats["Cluster"] = labels
    labelled_stats["Cluster_Category"] = labelled_stats["Cluster"].astype(str)
    return labelled_stats


def plot_silhouette_profile(
    data: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int,
) -> float:
    """
    Generates a silhouette plot for the distinct clusters to evaluate clustering quality.
    """

    _, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    y_lower = 10

    sample_silhouette_values = silhouette_samples(data, labels)
    avg_silhouette = silhouette_score(data, labels)

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(f"Silhouette Plot for {n_clusters} Clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=avg_silhouette, color="red", linestyle="--")
    plt.show()

    print(f"Average Silhouette Width: {avg_silhouette:.2f}")
    return float(avg_silhouette)


def add_pca_components(stats: pd.DataFrame, scaled_data: pd.DataFrame) -> pd.DataFrame:
    """Attach two PCA components to a copy of the clustered stats dataframe."""

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    stats_with_pca = stats.copy()
    stats_with_pca["PCA1"] = pca_components[:, 0]
    stats_with_pca["PCA2"] = pca_components[:, 1]
    return stats_with_pca


def plot_pca_clusters(stats: pd.DataFrame) -> None:
    """Plot clusters using PCA1 and PCA2 columns."""

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="PCA1",
        y="PCA2",
        hue="Cluster_Category",
        data=stats,
        palette="tab10",
        s=60,
        alpha=0.8,
        edgecolor="black",
    )
    plt.title("Clusters Visualized in 2D using PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()


def cluster_feature_means(
    stats: pd.DataFrame,
    selected_features: Iterable[str],
) -> pd.DataFrame:
    """Return original-scale feature averages by cluster."""

    return stats.groupby("Cluster")[list(selected_features)].mean()
