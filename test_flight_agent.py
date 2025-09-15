import unittest
from unittest.mock import patch, MagicMock


class TestSearchFlights(unittest.TestCase):
    """Tests for the flight search wrapper over the Amadeus client."""

    @patch("flight_agent.Client")
    @patch("flight_agent.os.getenv")
    def test_search_flights_success(self, mock_getenv, mock_client_class):
        """It returns simplified flight results when Amadeus responds successfully."""
        from flight_agent import search_flights

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "AMADEUS_ID": "test-id",
            "AMADEUS_SECRET": "test-secret",
        }.get(key)

        # Mock Amadeus client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            {
                "price": {"total": "123.45"},
                "itineraries": [
                    {
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "SFO",
                                    "at": "2025-10-10T08:00",
                                },
                                "arrival": {
                                    "iataCode": "LAX",
                                    "at": "2025-10-10T09:30",
                                },
                                "carrierCode": "UA",
                            }
                        ]
                    }
                ],
            }
        ]

        mock_client.shopping.flight_offers_search.get.return_value = mock_response

        results = search_flights("SFO", "LAX", "2025-10-10", adults=1)

        self.assertIsInstance(results, list)
        self.assertEqual(results[0]["price"], "123.45")
        self.assertEqual(results[0]["itineraries"][0]["from"], "SFO")
        self.assertEqual(results[0]["itineraries"][0]["to"], "LAX")
        self.assertEqual(results[0]["itineraries"][0]["carrier"], "UA")

    @patch("flight_agent.Client")
    @patch("flight_agent.os.getenv")
    def test_search_flights_error(self, mock_getenv, mock_client_class):
        """It returns an error dict when the Amadeus client raises an error."""
        from flight_agent import search_flights

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "AMADEUS_ID": "test-id",
            "AMADEUS_SECRET": "test-secret",
        }.get(key)

        # Mock Amadeus client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create a simple exception that will be caught by the except block
        mock_client.shopping.flight_offers_search.get.side_effect = Exception("Boom")

        result = search_flights("SFO", "LAX", "2025-10-10")

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
