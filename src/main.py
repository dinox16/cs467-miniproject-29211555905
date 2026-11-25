import argparse
import os

import pandas as pd
import numpy as np

from recommender import TravelRecommender

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

TEST_QUERY = "relaxing coastal town with seafood, cafes, and heritage"
TEST_USER = None
TEST_COLD_START = "I enjoy quiet mountains, scenic hiking trails, and nature."


def format_results(title, results, score_name):
    lines = [title]
    if not results:
        lines.append("  No match found.")
        return lines
    for rank, item in enumerate(results, 1):
        line = f"  {rank}. {item['destination_name']} ({item['region']}, {item['budget_level']} budget) | {score_name}: {item['score']:.3f}"
        lines.append(line)
        details = []
        if item["keywords"]:
            details.append("keywords: " + ", ".join(item["keywords"]))
        if item["note"]:
            details.append(item["note"])
        if details:
            lines.append("     " + " | ".join(details))
    return lines


def describe_user(model, user_id):
    rows = model.ratings[model.ratings["user_id"] == user_id].sort_values("rating", ascending=False)
    names = dict(zip(model.items["item_id"], model.items["destination_name"]))
    liked = [names[i] for i in rows[rows["rating"] >= 4]["item_id"]][:5]
    return f"Traveler {user_id} rated {len(rows)} trips. Top liked: {', '.join(liked)}."


def test_case_query(model):
    lines = ["=== TEST CASE 1: Preference query ===", f"Query: {TEST_QUERY}", ""]
    lines += format_results("NLP content-based Top-5:", model.recommend_by_query(TEST_QUERY), "similarity")
    return lines


def test_case_returning(model, user_id=TEST_USER):
    if user_id is None:
        eligible = [u for u in model.user_ids if len(model.rated_indexes(u)) >= 5]
        if not eligible:
            return ["No traveler has at least five ratings."]
        user_id = max(eligible, key=lambda u: np.isfinite(model.collaborative_prediction(u)[0][np.setdiff1d(np.arange(len(model.item_ids)), model.rated_indexes(u))]).sum())
    lines = ["=== TEST CASE 2: Returning traveler ===", describe_user(model, user_id), ""]
    lines += format_results("Popularity baseline Top-5:", model.recommend_popularity(user_id), "weighted rating")
    lines.append("")
    lines += format_results("NLP content-based Top-5 (user profile):", model.recommend_content_user(user_id), "similarity")
    lines.append("")
    lines += format_results("User-based collaborative Top-5:", model.recommend_collaborative(user_id), "predicted rating")
    lines.append("")
    lines += format_results("Hybrid Top-5:", model.recommend_hybrid(user_id), "hybrid score")
    return lines


def test_case_cold_start(model, text=TEST_COLD_START):
    lines = ["=== TEST CASE 3: Cold-start visitor ===", f"New visitor says: {text}", ""]
    lines += format_results("Recommendations from the text only:", model.recommend_cold_start(text), "similarity")
    return lines


def run_demo(model):
    blocks = [test_case_query(model), test_case_returning(model), test_case_cold_start(model)]
    lines = []
    for block in blocks:
        lines += block + [""]
    return lines


def run_custom(model, user_id, query):
    if user_id and query:
        title = f"Hybrid Top-5 for {user_id} with your text"
        lines = [describe_user(model, user_id)] if model.has_history(user_id) else [f"No history for {user_id}. Using text only."]
        lines += [""] + format_results(title, model.recommend_hybrid(user_id, query), "score")
        return lines
    if user_id:
        if not model.has_history(user_id):
            return [f"No history for {user_id}. Please try another user ID or use --query."]
        return test_case_returning(model, user_id)
    lines = ["=== YOUR QUERY ===", f"Text: {query}", ""]
    lines += format_results("Recommendations from the text only:", model.recommend_cold_start(query), "similarity")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Travel destination recommender")
    parser.add_argument("--user", help="User ID, for example U005")
    parser.add_argument("--query", help="Your travel wish in simple words")
    parser.add_argument("--demo", action="store_true", help="Run the 3 test cases")
    parser.add_argument("--save", action="store_true", help="Save custom output to results/custom_output.txt")
    parser.add_argument("--skip-evaluation", action="store_true", help="Run demo without offline evaluation")
    args = parser.parse_args()

    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    model = TravelRecommender(items, ratings)

    if args.user or args.query:
        lines = run_custom(model, args.user, args.query)
    else:
        lines = run_demo(model)

    text = "\n".join(lines)
    print(text)
    custom = bool(args.user or args.query)
    if not custom or args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, "custom_output.txt" if custom else "sample_outputs.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Saved recommendations: {output_path}")
    if not custom and not args.skip_evaluation:
        from evaluate import main as evaluate_main
        print("\n=== OFFLINE EVALUATION (30 splits, 3 hidden ratings per user) ===", flush=True)
        evaluate_main()


if __name__ == "__main__":
    main()
