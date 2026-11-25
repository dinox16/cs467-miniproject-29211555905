import numpy as np
import pandas as pd

from nlp_pipeline import NLPPipeline


def min_max(values):
    values = np.asarray(values, dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class TravelRecommender:
    def __init__(self, items, ratings, min_count=2, neighbors=5, min_common=3, alpha=0.6, damping=1.0):
        self.items = items.reset_index(drop=True)
        self.ratings = ratings.copy()
        self.min_count = min_count
        self.neighbors = neighbors
        self.min_common = min_common
        self.alpha = alpha
        self.damping = damping
        self.item_ids = list(self.items["item_id"])
        self.item_index = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.pipeline = NLPPipeline().fit(self.items)
        self.user_ids = sorted(self.ratings["user_id"].unique())
        self.user_index = {user_id: i for i, user_id in enumerate(self.user_ids)}
        self.matrix = np.full((len(self.user_ids), len(self.item_ids)), np.nan)
        for row in self.ratings.itertuples():
            self.matrix[self.user_index[row.user_id], self.item_index[row.item_id]] = row.rating
        self.global_mean = float(np.nanmean(self.matrix))
        self.user_means = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else self.global_mean for r in self.matrix])
        self.item_counts = (~np.isnan(self.matrix)).sum(axis=0)
        self.popularity = self.compute_popularity()

    def has_history(self, user_id):
        return user_id in self.user_index

    def rated_indexes(self, user_id):
        if not self.has_history(user_id):
            return np.array([], dtype=int)
        row = self.matrix[self.user_index[user_id]]
        return np.where(~np.isnan(row))[0]

    def compute_popularity(self):
        sums = np.nansum(self.matrix, axis=0)
        prior = 3
        return (sums + prior * self.global_mean) / (self.item_counts + prior)

    def top_k(self, scores, exclude, k, allowed=None):
        scores = np.array(scores, dtype=float)
        scores[np.isnan(scores)] = -np.inf
        if len(exclude):
            scores[exclude] = -np.inf
        if allowed is not None:
            scores[~allowed] = -np.inf
        order = np.argsort(-scores)
        return [i for i in order if np.isfinite(scores[i])][:k]

    def build_results(self, indexes, scores, keyword_vector=None, notes=None):
        results = []
        for i in indexes:
            keywords = []
            if keyword_vector is not None:
                keywords = self.pipeline.matching_keywords(keyword_vector, i)
            row = self.items.iloc[i]
            results.append({
                "item_id": row["item_id"],
                "destination_name": row["destination_name"],
                "region": row["region"],
                "budget_level": row["budget_level"],
                "score": float(scores[i]),
                "keywords": keywords,
                "note": "" if notes is None else notes.get(i, ""),
            })
        return results

    def popularity_scores(self):
        return self.popularity

    def recommend_popularity(self, user_id=None, k=5):
        exclude = self.rated_indexes(user_id) if user_id else []
        allowed = self.item_counts >= self.min_count
        indexes = self.top_k(self.popularity, exclude, k, allowed)
        notes = {i: f"average rating from {int(self.item_counts[i])} travelers" for i in indexes}
        return self.build_results(indexes, self.popularity, notes=notes)

    def query_scores(self, text):
        vector = self.pipeline.transform_text(text)
        return vector, self.pipeline.similarity_to_items(vector)

    def recommend_by_query(self, text, k=5, exclude_user=None):
        vector, scores = self.query_scores(text)
        exclude = self.rated_indexes(exclude_user) if exclude_user else []
        indexes = self.top_k(scores, exclude, k, scores > 0)
        return self.build_results(indexes, scores, keyword_vector=vector)

    def recommend_similar_items(self, item_id, k=5):
        index = self.item_index[item_id]
        scores = self.pipeline.similarity_between_items(index)
        indexes = self.top_k(scores, [index], k)
        vector = self.pipeline.item_matrix[index]
        return self.build_results(indexes, scores, keyword_vector=vector)

    def user_profile(self, user_id):
        indexes = self.rated_indexes(user_id)
        if len(indexes) == 0:
            return None
        ratings = self.matrix[self.user_index[user_id], indexes]
        weights = np.clip(ratings - 3, 0, None)
        if weights.sum() == 0:
            weights = ratings
        return self.pipeline.build_profile(indexes, weights)

    def content_user_scores(self, user_id):
        profile = self.user_profile(user_id)
        if profile is None:
            return None, np.zeros(len(self.item_ids))
        return profile, self.pipeline.similarity_to_items(profile)

    def recommend_content_user(self, user_id, k=5):
        profile, scores = self.content_user_scores(user_id)
        indexes = self.top_k(scores, self.rated_indexes(user_id), k, scores > 0)
        return self.build_results(indexes, scores, keyword_vector=profile)

    def content_rating_prediction(self, user_id, top_n=5):
        rated = self.rated_indexes(user_id)
        fallback = self.user_means[self.user_index[user_id]] if self.has_history(user_id) else self.global_mean
        predictions = np.full(len(self.item_ids), fallback, dtype=float)
        if len(rated) == 0:
            return predictions
        user_row = self.matrix[self.user_index[user_id]]
        for i in range(len(self.item_ids)):
            sims = self.pipeline.similarity_between_items(i)[rated]
            order = np.argsort(-sims)[:top_n]
            picked = sims[order]
            if picked.sum() > 0:
                predictions[i] = np.dot(picked, user_row[rated][order]) / picked.sum()
        return predictions

    def user_similarities(self, user_id):
        sims = np.zeros(len(self.user_ids))
        if not self.has_history(user_id):
            return sims
        me = self.user_index[user_id]
        for other in range(len(self.user_ids)):
            if other == me:
                continue
            common = ~np.isnan(self.matrix[me]) & ~np.isnan(self.matrix[other])
            count = int(common.sum())
            if count < self.min_common:
                continue
            a = self.matrix[me, common] - self.user_means[me]
            b = self.matrix[other, common] - self.user_means[other]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm < 1e-12:
                continue
            sims[other] = float(np.dot(a, b) / norm) * count / (count + 5)
        return sims

    def collaborative_prediction(self, user_id):
        predictions = np.full(len(self.item_ids), np.nan)
        counts = np.zeros(len(self.item_ids), dtype=int)
        if not self.has_history(user_id):
            return predictions, counts
        me = self.user_index[user_id]
        sims = self.user_similarities(user_id)
        for i in range(len(self.item_ids)):
            raters = np.where(~np.isnan(self.matrix[:, i]) & (sims > 0))[0]
            if len(raters) == 0:
                continue
            raters = raters[np.argsort(-sims[raters])][: self.neighbors]
            weights = sims[raters]
            centered = self.matrix[raters, i] - self.user_means[raters]
            predictions[i] = self.user_means[me] + np.dot(weights, centered) / (weights.sum() + self.damping)
            counts[i] = len(raters)
        return np.clip(predictions, 1, 5), counts

    def collaborative_with_fallback(self, user_id):
        predictions, counts = self.collaborative_prediction(user_id)
        fallback = self.popularity.copy()
        filled = np.where(np.isnan(predictions), fallback, predictions)
        return filled, counts

    def recommend_collaborative(self, user_id, k=5):
        predictions, counts = self.collaborative_prediction(user_id)
        indexes = self.top_k(predictions, self.rated_indexes(user_id), k)
        notes = {i: f"liked by {counts[i]} similar travelers" for i in indexes}
        return self.build_results(indexes, predictions, notes=notes)

    def hybrid_scores(self, user_id, query=None):
        cf_scores, counts = self.collaborative_with_fallback(user_id)
        profile, content_scores = self.content_user_scores(user_id)
        keyword_vector = profile
        if query:
            query_vector, query_scores = self.query_scores(query)
            content_scores = 0.5 * min_max(content_scores) + 0.5 * min_max(query_scores)
            if profile is None:
                keyword_vector = query_vector
            else:
                keyword_vector = np.asarray(query_vector.todense()).ravel() + profile / max(np.linalg.norm(profile), 1e-12)
        blended = self.alpha * min_max(cf_scores) + (1 - self.alpha) * min_max(content_scores)
        return blended, counts, keyword_vector

    def recommend_hybrid(self, user_id, query=None, k=5):
        if not self.has_history(user_id):
            if query:
                return self.recommend_by_query(query, k)
            return self.recommend_popularity(None, k)
        blended, counts, keyword_vector = self.hybrid_scores(user_id, query)
        indexes = self.top_k(blended, self.rated_indexes(user_id), k)
        notes = {i: (f"liked by {counts[i]} similar travelers" if counts[i] else "matches your taste") for i in indexes}
        return self.build_results(indexes, blended, keyword_vector=keyword_vector, notes=notes)

    def recommend_cold_start(self, text, k=5):
        if not self.pipeline.query_is_known(text):
            return self.recommend_popularity(None, k)
        return self.recommend_by_query(text, k)

    def predict_ratings(self, method, user_id):
        if method == "popularity":
            return self.popularity.copy()
        if method == "content":
            return self.content_rating_prediction(user_id)
        if method == "collaborative":
            return self.collaborative_with_fallback(user_id)[0]
        if method == "hybrid":
            cf = self.collaborative_with_fallback(user_id)[0]
            content = self.content_rating_prediction(user_id)
            return self.alpha * cf + (1 - self.alpha) * content
        raise ValueError(method)

    def rank_for_evaluation(self, method, user_id, k=5, rng=None):
        exclude = self.rated_indexes(user_id)
        if method == "popularity":
            return self.top_k(self.popularity, exclude, k, self.item_counts >= self.min_count)
        if method == "content":
            return self.top_k(self.content_user_scores(user_id)[1], exclude, k)
        if method == "collaborative":
            return self.top_k(self.collaborative_prediction(user_id)[0], exclude, k)
        if method == "hybrid":
            return self.top_k(self.hybrid_scores(user_id)[0], exclude, k)
        if method == "random":
            candidates = np.setdiff1d(np.arange(len(self.item_ids)), exclude)
            return list(rng.choice(candidates, size=k, replace=False))
        raise ValueError(method)
