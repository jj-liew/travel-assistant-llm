import unittest
from unittest.mock import patch, MagicMock


class TestQueryCity(unittest.TestCase):
    @patch("itinerary_agent.OpenAI")
    @patch("itinerary_agent.Pinecone")
    @patch("itinerary_agent.os.getenv")
    def test_query_city_success(
        self, mock_getenv, mock_pinecone_class, mock_openai_class
    ):
        from itinerary_agent import query_city

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "GROQ_API_KEY": "test-groq-key",
            "PINECONE_API_KEY": "test-pinecone-key",
        }.get(key)

        # Mock Pinecone
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.search.return_value = {
            "result": {
                "hits": [
                    {"fields": {"text": "Attraction A details."}},
                    {"fields": {"text": "Attraction B details."}},
                ]
            }
        }

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Here is an answer about the city."
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp

        result = query_city("paris", "What to see?")

        self.assertIn("answer", result.lower())

    @patch("itinerary_agent.OpenAI")
    @patch("itinerary_agent.Pinecone")
    @patch("itinerary_agent.os.getenv")
    def test_query_city_exception(
        self, mock_getenv, mock_pinecone_class, mock_openai_class
    ):
        from itinerary_agent import query_city

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "GROQ_API_KEY": "test-groq-key",
            "PINECONE_API_KEY": "test-pinecone-key",
        }.get(key)

        # Mock Pinecone
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.search.side_effect = Exception("Search failed")

        result = query_city("paris", "What to see?")

        self.assertIsInstance(result, str)
        self.assertIn("error while querying", result.lower())
