import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

system_prompt = """
You are a travel planning assistant. 
You ONLY answer questions related to flights, travel itineraries, or trip planning. 
If the user asks something outside of this domain, politely respond with:
"I'm only able to help with travel itineraries and flights."
"""
format = "Always return the response in simple plaintext, no tables."


class TestAPI(unittest.TestCase):
    """Tests for FastAPI endpoints with API key authentication."""

    def setUp(self):
        """Set up test client."""
        from main import app

        self.client = TestClient(app)

    @patch("main.run_travel_llm")
    def test_ask_endpoint(self, mock_ask):
        """Test ask endpoint."""
        mock_ask.return_value = "LLM answer"
        resp = self.client.post("/ask", json={"prompt": "hi"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "LLM answer")
        mock_ask.assert_called_once_with("hi" + system_prompt + format)
