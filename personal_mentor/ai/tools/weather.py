import os

import requests

from personal_mentor.ai.llms import LLM
from personal_mentor.utils.logging import logger


def generate_forecast_text(daily_forecast):
    # Convert the forecast response into a nice text using OpenAI
    prompt = f"Convert the following forecast data into a friendly text which will be read: {daily_forecast}"
    response = LLM.invoke(
        [
            {
                "role": "system",
                "content": """
You are a helpful AI assistant which conveys the current weather forecast. 
Provide clear, concise, and friendly responses for humans to understand.
We're dealing with celsius and the metric system.
Summarize days if appropriate.
No special highlighting or emphasis.
""",
            },
            {"role": "user", "content": prompt},
        ]
    )

    return response.content


def get_weather_forecast(city, country):
    """
    Retrieves the weather forecast for a given city and country.

    Args:
        city: A string representing the city.
        country: A string representing the country.

    Returns:
        A string containing the weather forecast text if successful,
        otherwise returns an error message string in case of an exception.
    """
    try:
        geocode_response = requests.get(
            f'https://maps.googleapis.com/maps/api/geocode/json?address={city},{country}&key={os.getenv("GOOGLE_GEOCODING_API_KEY")}'
        )

        geocode_data = geocode_response.json()
        results = geocode_data.get("results", [])

        if not results:
            return "Location not found"

        geometry = results[0]["geometry"]
        lat = geometry["location"]["lat"]
        lng = geometry["location"]["lng"]

        weather_response = requests.get(
            f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lng}&units=metric&exclude=minutely,hourly&appid={os.getenv("OPENWEATHER_API_KEY")}'
        )

        if weather_response.status_code != 200:
            return weather_response.reason

        forecast_data = weather_response.json().get("list", [])

        forecast_text = generate_forecast_text(forecast_data)

        return forecast_text

    except Exception as e:
        return str(e)


def weather_forecast(location: str) -> str:
    """
    Retrieves the weather forecast for a given location.

    Args:
        location: A string representing the location in the format 'City, Country'.

    Returns:
        A string containing the weather forecast text if successful,
        otherwise logs an error and returns None for invalid input format
        or returns an error message string in case of an exception.
    """
    try:
        # Split the location into city and country if possible
        location_parts = location.split(",")

        if len(location_parts) == 2:
            city, country = location_parts[0].strip(), location_parts[1].strip()
            forecast_response = get_weather_forecast(city, country)

            return forecast_response
        else:
            logger.error("Invalid location format. Use 'City, Country'")
            return None

    except Exception as e:
        return logger.error("Error: " + str(e))


if __name__ == "__main__":
    print(weather_forecast("Bangkok, Thailand"))
