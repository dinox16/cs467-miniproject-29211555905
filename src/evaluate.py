import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recommender import TravelRecommender

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

RANKING_METHODS = ["random", "popularity", "content", "collaborative", "hybrid"]
RATING_METHODS = ["popularity", "content", "collaborative", "hybrid"]
METHOD_LABELS = {
    "random": "Random",
    "popularity": "Popularity",
    "content": "NLP Content",
    "collaborative": "User CF",
    "hybrid": "Hybrid",
}


def split_ratings(ratings, rng, holdout=3):
    test_parts = []
    for _, group in ratings.groupby("user_id"):
        picked = rng.choice(group.index.to_numpy(), size=holdout, replace=False)
        test_parts.append(ratings.loc[picked])
    test = pd.concat(test_parts)
    train = ratings.drop(test.index)
    return train, test


def evaluate_split(items, ratings, seed, k=5, like_threshold=4):
    rng = np.random.default_rng(seed)
    train, test = split_ratings(ratings, rng)
    model = TravelRecommender(items, train)
    precision = {m: [] for m in RANKING_METHODS}
    errors = {m: [] for m in RATING_METHODS}
    for user_id, group in test.groupby("user_id"):
        liked = set(group.loc[group["rating"] >= like_threshold, "item_id"])
        for method in RANKING_METHODS:
            indexes = model.rank_for_evaluation(method, user_id, k, rng)
            hits = sum(1 for i in indexes if model.item_ids[i] in liked)
            precision[method].append(hits / k)
        for method in RATING_METHODS:
            predicted = model.predict_ratings(method, user_id)
            for row in group.itertuples():
                errors[method].append(predicted[model.item_index[row.item_id]] - row.rating)
    return precision, errors


def run_evaluation(items, ratings, runs=30):
    precision = {m: [] for m in RANKING_METHODS}
    errors = {m: [] for m in RATING_METHODS}
    for seed in range(runs):
        run_precision, run_errors = evaluate_split(items, ratings, seed)
        for method in RANKING_METHODS:
            precision[method].extend(run_precision[method])
        for method in RATING_METHODS:
            errors[method].extend(run_errors[method])
    rows = []
    for method in RANKING_METHODS:
        row = {"method": METHOD_LABELS[method], "precision_at_5": float(np.mean(precision[method])), "rmse": np.nan, "mae": np.nan}
        if method in errors:
            diffs = np.array(errors[method])
            row["rmse"] = float(np.sqrt(np.mean(diffs ** 2)))
            row["mae"] = float(np.mean(np.abs(diffs)))
        rows.append(row)
    return pd.DataFrame(rows)


def save_rating_distribution(ratings, path):
    counts = ratings["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#3b7ea1")
    for x, y in zip(counts.index.astype(str), counts.values):
        ax.text(x, y + 0.5, str(y), ha="center")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    ax.set_title("Rating distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_category_distribution(items, path):
    table = pd.crosstab(items["region"], items["budget_level"])
    table = table[[c for c in ["low", "medium", "high"] if c in table.columns]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    table.plot(kind="bar", stacked=True, ax=axes[0], color=["#8fbf9f", "#f2c14e", "#d1495b"])
    axes[0].set_title("Destinations by region and budget")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Number of destinations")
    axes[0].tick_params(axis="x", rotation=0)
    tags = items["activities_tags"].str.split("|").explode().value_counts().head(12)
    axes[1].barh(tags.index[::-1], tags.values[::-1], color="#3b7ea1")
    axes[1].set_title("Top 12 activity tags")
    axes[1].set_xlabel("Number of destinations")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_precision_chart(results, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#b0b0b0", "#f2c14e", "#3b7ea1", "#8fbf9f", "#d1495b"]
    bars = ax.bar(results["method"], results["precision_at_5"], color=colors)
    for bar, value in zip(bars, results["precision_at_5"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center")
    ax.set_ylabel("Precision@5")
    ax.set_title("Precision@5 by method")
    ax.set_ylim(0, max(results["precision_at_5"]) * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_error_chart(results, path):
    data = results.dropna(subset=["rmse"])
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rmse_bars = ax.bar(x - width / 2, data["rmse"], width, label="RMSE", color="#3b7ea1")
    mae_bars = ax.bar(x + width / 2, data["mae"], width, label="MAE", color="#f2c14e")
    for bars in (rmse_bars, mae_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(data["method"])
    ax.set_ylabel("Error (lower is better)")
    ax.set_title("Rating prediction error by method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    results = run_evaluation(items, ratings)
    results.to_csv(os.path.join(RESULTS_DIR, "evaluation_results.csv"), index=False)
    save_rating_distribution(ratings, os.path.join(RESULTS_DIR, "rating_distribution.png"))
    save_category_distribution(items, os.path.join(RESULTS_DIR, "category_distribution.png"))
    save_precision_chart(results, os.path.join(RESULTS_DIR, "precision_at_5.png"))
    save_error_chart(results, os.path.join(RESULTS_DIR, "rating_error.png"))
    print(results.round(3).to_string(index=False))
    return results


if __name__ == "__main__":
    main()
