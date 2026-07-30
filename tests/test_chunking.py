import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docqa.chunking import Chunk, chunk_documents, chunk_text
from docqa.loaders import Document


class TestChunkText(unittest.TestCase):
    def test_short_text_returns_single_chunk(self):
        text = "one two three four five"
        chunks = chunk_text(text, chunk_size=150, overlap=30)
        self.assertEqual(chunks, ["one two three four five"])

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("   \n  \t  "), [])
        self.assertEqual(chunk_text(""), [])

    def test_long_text_splits_into_multiple_chunks(self):
        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # Every chunk should be at most chunk_size words.
        for c in chunks:
            self.assertLessEqual(len(c.split()), 100)

    def test_overlap_is_respected_between_consecutive_chunks(self):
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        self.assertGreaterEqual(len(chunks), 2)
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        # The last `overlap` words of chunk 1 should equal the first
        # `overlap` words of chunk 2.
        self.assertEqual(first_words[-5:], second_words[:5])

    def test_no_words_lost_across_chunks(self):
        words = [f"tok{i}" for i in range(237)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        rejoined = set()
        for c in chunks:
            rejoined.update(c.split())
        self.assertEqual(rejoined, set(words))

    def test_whitespace_is_normalized(self):
        text = "hello    world\n\nfoo\tbar"
        chunks = chunk_text(text, chunk_size=150, overlap=30)
        self.assertEqual(chunks, ["hello world foo bar"])

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_size=10, overlap=10)
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_size=10, overlap=-1)
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_size=0, overlap=0)


class TestChunkDocuments(unittest.TestCase):
    def test_produces_chunks_with_correct_source_and_index(self):
        docs = [
            Document(source="a.txt", path="/tmp/a.txt", text="alpha beta gamma"),
            Document(source="b.txt", path="/tmp/b.txt", text="delta epsilon zeta"),
        ]
        chunks = chunk_documents(docs, chunk_size=150, overlap=30)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))
        sources = {c.source for c in chunks}
        self.assertEqual(sources, {"a.txt", "b.txt"})
        for c in chunks:
            self.assertEqual(c.chunk_index, 0)
            self.assertTrue(c.id.startswith(c.source))

    def test_empty_document_produces_no_chunks(self):
        docs = [Document(source="empty.txt", path="/tmp/empty.txt", text="   ")]
        chunks = chunk_documents(docs)
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
