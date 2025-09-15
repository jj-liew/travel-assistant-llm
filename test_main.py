import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient


class TestAPI(unittest.TestCase):
    """Tests for FastAPI endpoints with API key authentication."""

    def setUp(self):
        """Set up test client."""
        from main import app

        self.client = TestClient(app)

    @patch("main.search_flights")
    def test_search_flights_endpoint(self, mock_search):
        """Test flight search endpoint."""
        mock_search.return_value = [{"price": "100.00"}]

        resp = self.client.post(
            "/search-flights",
            json={
                "origin": "SFO",
                "destination": "LAX",
                "date": "2025-10-10",
                "adults": 1,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [{"price": "100.00"}])
        mock_search.assert_called_once_with(
            origin="SFO", destination="LAX", date="2025-10-10", adults=1
        )

    @patch("main.query_city")
    def test_query_city_endpoint(self, mock_query):
        """Test city query endpoint."""
        mock_query.return_value = "City answer"
        resp = self.client.post(
            "/query-city",
            json={"city": "paris", "question": "what to do?"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), "City answer")
        mock_query.assert_called_once_with("paris", "what to do?")

    @patch("main.run_travel_llm")
    def test_ask_endpoint(self, mock_ask):
        """Test ask endpoint."""
        mock_ask.return_value = "LLM answer"
        resp = self.client.post("/ask", json={"prompt": "hi"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), "LLM answer")
        mock_ask.assert_called_once_with("hi")
