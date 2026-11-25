import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from recommender import TravelRecommender
from evaluate import split_ratings, evaluate_split
from main import run_demo


class RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = pd.read_csv(ROOT / 'data/items.csv')
        cls.ratings = pd.read_csv(ROOT / 'data/ratings.csv')
        cls.model = TravelRecommender(cls.items, cls.ratings)

    def test_data(self):
        self.assertEqual(len(self.items), 40)
        self.assertEqual(self.ratings.user_id.nunique(), 15)
        self.assertEqual(len(self.ratings), 303)
        self.assertFalse(self.ratings.duplicated(['user_id', 'item_id']).any())
        self.assertTrue(self.ratings.rating.between(1, 5).all())
        self.assertTrue(set(self.ratings.item_id) <= set(self.items.item_id))

    def test_all_users(self):
        for user in self.model.user_ids:
            seen = set(self.ratings.loc[self.ratings.user_id == user, 'item_id'])
            for fn in [self.model.recommend_popularity, self.model.recommend_content_user, self.model.recommend_collaborative, self.model.recommend_hybrid]:
                results = fn(user)
                ids = [r['item_id'] for r in results]
                self.assertEqual(len(ids), 5)
                self.assertEqual(len(set(ids)), 5)
                self.assertFalse(set(ids) & seen)
                self.assertTrue(all(np.isfinite(r['score']) for r in results))
            self.assertTrue(all(0 <= r['score'] <= 1 for r in self.model.recommend_hybrid(user)))

    def test_constant_ratings(self):
        self.assertEqual(self.ratings.loc[self.ratings.user_id == 'U005', 'rating'].nunique(), 1)
        self.assertTrue(all('fallback' in r['note'].lower() for r in self.model.recommend_collaborative('U005')))
        self.assertTrue(all('Content-only' in r['note'] for r in self.model.recommend_hybrid('U005')))

    def test_query(self):
        for q in ['relaxing coastal town with seafood, cafes, and heritage', 'I like quiet mountains and tea hills', 'zzzzzzzzzz', '']:
            results = self.model.recommend_cold_start(q)
            self.assertEqual(len(results), 5)
            self.assertTrue(all(0 <= r['score'] <= 1 for r in results))
        self.assertTrue(all('fallback' in r['note'].lower() for r in self.model.recommend_cold_start('zzzzzzzzzz')))

    def test_unknown_hybrid(self):
        self.assertTrue(all(0 <= r['score'] <= 1 for r in self.model.recommend_hybrid('unknown')))

    def test_split(self):
        train, test = split_ratings(self.ratings, np.random.default_rng(0))
        self.assertEqual(len(test), 45)
        self.assertFalse(set(zip(train.user_id, train.item_id)) & set(zip(test.user_id, test.item_id)))
        precision, errors = evaluate_split(self.items, self.ratings, 0)
        self.assertTrue(all(len(v) == 15 for v in precision.values()))
        self.assertTrue(all(len(v) == 45 and np.isfinite(v).all() for v in errors.values()))

    def test_demo(self):
        self.assertNotIn('No match found.', '\n'.join(run_demo(self.model)))


if __name__ == '__main__':
    unittest.main()
