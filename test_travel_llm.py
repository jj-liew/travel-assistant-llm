import unittest
from unittest.mock import patch, MagicMock


class TestRunTravelLLM(unittest.TestCase):
    """Tests for tool-enabled LLM flow orchestrated in travel_llm.run_travel_llm."""

    @patch("travel_llm.search_flights")
    @patch("travel_llm.query_city")
    @patch("travel_llm.OpenAI")
    @patch("travel_llm.os.getenv")
    def test_run_travel_llm_with_tool_call(
        self, mock_getenv, mock_openai_class, mock_query_city, mock_search_flights
    ):
        """When model requests a tool, we execute it and do a follow-up call."""
        from travel_llm import run_travel_llm

        # Mock environment variable
        mock_getenv.return_value = "test-groq-key"

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        tool_call = MagicMock()
        tool_call.id = "tc_123"
        tool_call.function.name = "search_flights"
        tool_call.function.arguments = "{ 'origin': 'SFO', 'destination': 'LAX', 'date': '2025-10-10', 'adults': 1 }"

        initial_choice = MagicMock()
        initial_choice.message.tool_calls = [tool_call]
        initial_resp = MagicMock()
        initial_resp.choices = [initial_choice]

        follow_choice = MagicMock()
        follow_choice.message.content = "Here are the results."
        follow_resp = MagicMock()
        follow_resp.choices = [follow_choice]

        mock_client.chat.completions.create.side_effect = [initial_resp, follow_resp]
        mock_search_flights.return_value = [{"price": "123.45"}]

        result = run_travel_llm("Find me a flight")
        self.assertIn("results", result.lower())

    @patch("travel_llm.OpenAI")
    @patch("travel_llm.os.getenv")
    def test_run_travel_llm_no_tool_calls(self, mock_getenv, mock_openai_class):
        """When no tool calls are present, return the model's first answer."""
        from travel_llm import run_travel_llm

        # Mock environment variable
        mock_getenv.return_value = "test-groq-key"

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        initial_choice = MagicMock()
        initial_choice.message.tool_calls = None
        initial_choice.message.content = "Plain answer"
        initial_resp = MagicMock()
        initial_resp.choices = [initial_choice]
        mock_client.chat.completions.create.return_value = initial_resp

        result = run_travel_llm("Hello")
        self.assertEqual(result, "Plain answer")
