import os
from amadeus import Client
from dotenv import load_dotenv

load_dotenv()


def search_flights(origin: str, destination: str, date: str, adults: int = 1):
    """Search flights using the Amadeus client and return simplified results.

    Args:
        origin: Origin airport or city IATA code.
        destination: Destination airport or city IATA code.
        date: Departure date in YYYY-MM-DD format.
        adults: Number of adult passengers.

    Returns:
        A list of flight options with price and first-itinerary segment details, or
        an error dict if the Amadeus client raises a ResponseError.
    """

    AMADEUS_ID = os.getenv("AMADEUS_ID")
    AMADEUS_SECRET = os.getenv("AMADEUS_SECRET")

    amadeus = Client(client_id=AMADEUS_ID, client_secret=AMADEUS_SECRET)

    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=date,
            adults=adults,
            max=3,
        )
        results = []
        for flight in response.data:
            results.append(
                {
                    "price": flight["price"]["total"],
                    "itineraries": [
                        {
                            "from": seg["departure"]["iataCode"],
                            "to": seg["arrival"]["iataCode"],
                            "carrier": seg["carrierCode"],
                            "departure_time": seg["departure"]["at"],
                            "arrival_time": seg["arrival"]["at"],
                        }
                        for seg in flight["itineraries"][0]["segments"]
                    ],
                }
            )
        return results
    except Exception as error:
        return {"error": str(error)}
