import httpx

class GeoService:

    async def get_coordinates(self, city: str):

        if city == "Unknown":
            return None, None

        url = "https://nominatim.openstreetmap.org/search"

        params = {
            "q": f"{city}, Ukraine",
            "format": "json",
            "limit": 1
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)

        data = response.json()

        if len(data) == 0:
            return None, None

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])

        return lat, lon