import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docqa.chunking import Chunk
from docqa.retriever import TfidfRetriever, tokenize


def make_chunk(chunk_id, source, text, chunk_index=0):
    return Chunk(id=chunk_id, source=source, chunk_index=chunk_index, text=text)


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        tokens = tokenize("Hello, World! This is Solstice's robot.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("solstice's", tokens)
        self.assertIn("robot", tokens)

    def test_stopwords_are_removed(self):
        tokens = tokenize("the cat and the dog")
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)
        self.assertIn("cat", tokens)
        self.assertIn("dog", tokens)


class TestTfidfRetriever(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            make_chunk("c0", "docA.txt", "Scout Mini is the entry-level greenhouse robot with a battery."),
            make_chunk("c1", "docB.txt", "Scout Pro adds a high-resolution camera for pest inspection."),
            make_chunk("c2", "docC.txt", "The company headquarters is located in Portland Oregon."),
            make_chunk("c3", "docD.txt", "Customer support can be reached by phone or email for help."),
        ]
        self.retriever = TfidfRetriever()
        self.retriever.fit(self.chunks)

    def test_is_fitted_after_fit(self):
        self.assertTrue(self.retriever.is_fitted)
        self.assertEqual(self.retriever.matrix.shape[0], 4)

    def test_fit_rejects_empty_chunk_list(self):
        with self.assertRaises(ValueError):
            TfidfRetriever().fit([])

    def test_query_returns_most_relevant_chunk_first(self):
        results = self.retriever.query("Does Scout Pro have a camera?", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.id, "c1")

    def test_query_ranks_unrelated_chunk_lower(self):
        results = self.retriever.query("greenhouse robot battery", top_k=4)
        ids_in_order = [r.chunk.id for r in results]
        # c0 is about the robot/battery and should outrank c3 (customer support).
        self.assertIn("c0", ids_in_order)
        if "c3" in ids_in_order:
            self.assertLess(ids_in_order.index("c0"), ids_in_order.index("c3"))

    def test_query_with_no_vocabulary_overlap_returns_empty(self):
        results = self.retriever.query("xylophone zeppelin quokka", top_k=4)
        self.assertEqual(results, [])

    def test_scores_are_sorted_descending(self):
        results = self.retriever.query("Portland headquarters company location", top_k=4)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_is_respected(self):
        results = self.retriever.query("robot camera support company", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_save_and_load_round_trip(self):
        tmpdir = tempfile.mkdtemp()
        try:
            index_dir = os.path.join(tmpdir, "index")
            self.retriever.save(index_dir)
            loaded = TfidfRetriever.load(index_dir)
            self.assertEqual(len(loaded.chunks), len(self.retriever.chunks))
            self.assertEqual(loaded.vocab, self.retriever.vocab)

            original_results = self.retriever.query("Scout Pro camera", top_k=2)
            loaded_results = loaded.query("Scout Pro camera", top_k=2)
            self.assertEqual(
                [r.chunk.id for r in original_results],
                [r.chunk.id for r in loaded_results],
            )
            for orig, load_r in zip(original_results, loaded_results):
                self.assertAlmostEqual(orig.score, load_r.score, places=9)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
