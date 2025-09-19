import unittest
from unittest.mock import patch, MagicMock
import rag_ingest


class TestRagIngest(unittest.TestCase):
    """Tests for RAG ingestion functions (scraping + embedding)."""

    def test_chunk_text_overlap(self):
        """chunk_text should split text into overlapping chunks."""
        text = "abcdefghij"
        chunks = rag_ingest.chunk_text(text, size=5, overlap=2)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "abcde")
        self.assertEqual(chunks[1], "defgh")
        self.assertEqual(chunks[2], "ghij")

    @patch("rag_ingest.requests.get")
    def test_scrape_city_returns_chunks(self, mock_get):
        """scrape_city should return a list of text chunks from fake HTML."""
        mock_get.return_value.text = (
            "<html><body><p>Hello world!</p><p>Another para.</p></body></html>"
        )

        chunks = rag_ingest.scrape_city("Tokyo", size=5, overlap=2)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(any("Hello" in c for c in chunks))

    @patch("rag_ingest.Pinecone")
    @patch("rag_ingest.scrape_city")
    def test_embed_db_success(self, mock_scrape_city, mock_pinecone):
        """embed_db should upsert chunks into Pinecone and return success."""
        mock_scrape_city.return_value = ["chunk1", "chunk2"]

        mock_index = MagicMock()
        mock_pinecone.return_value.Index.return_value = mock_index

        result = rag_ingest.embed_db("Tokyo")

        self.assertEqual(result["status"], "success")
        self.assertIn("Successfully embedded", result["message"])
        self.assertEqual(mock_index.upsert_records.call_count, 2)

    @patch("rag_ingest.Pinecone")
    @patch("rag_ingest.scrape_city")
    def test_embed_db_failure(self, mock_scrape_city, mock_pinecone):
        """embed_db should return error dict if scraping fails."""
        mock_scrape_city.side_effect = Exception("Boom!")

        result = rag_ingest.embed_db("Tokyo")
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
