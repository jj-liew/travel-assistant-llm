import unittest
from unittest.mock import patch, MagicMock
import rag_ingest


class TestRagIngest(unittest.TestCase):
    """Tests for rag_ingest (scraping + embedding)."""

    def test_chunk_text(self):
        """chunk_text should split text into overlapping chunks."""
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = rag_ingest.chunk_text(text, size=10, overlap=3)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(any("abc" in c for c in chunks))

    @patch("rag_ingest.requests.get")
    def test_scrape_city_returns_chunks(self, mock_get):
        """scrape_city should return a list of text chunks from fake HTML."""
        mock_get.return_value.text = (
            "<html><body><p>Hello world!</p><p>Another para.</p></body></html>"
        )

        chunks = rag_ingest.scrape_city("Tokyo", size=20, overlap=5)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(any("Hello" in c for c in chunks))

    @patch("rag_ingest.Pinecone")
    @patch("rag_ingest.scrape_city")
    @patch("rag_ingest.os.getenv")
    def test_embed_db_success(self, mock_getenv, mock_scrape_city, mock_pinecone):
        """embed_db should upsert chunks into Pinecone and return success dict."""
        mock_getenv.return_value = "fake-key"
        mock_scrape_city.return_value = ["chunk1", "chunk2"]

        mock_index = MagicMock()
        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        result = rag_ingest.embed_db("Tokyo")

        self.assertEqual(result["status"], "success")
        self.assertIn("Successfully embedded", result["message"])
        mock_index.upsert_records.assert_called()

    @patch("rag_ingest.Pinecone")
    @patch("rag_ingest.scrape_city", side_effect=Exception("Scrape failed"))
    @patch("rag_ingest.os.getenv")
    def test_embed_db_failure(self, mock_getenv, mock_scrape_city, mock_pinecone):
        """embed_db should return error dict if scraping fails."""
        mock_getenv.return_value = "fake-key"
        mock_pinecone.return_value = MagicMock()

        result = rag_ingest.embed_db("Tokyo")

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
