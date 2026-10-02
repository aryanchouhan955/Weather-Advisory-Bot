"""
tools/weather.py
----------------
Pure Python module for weather data retrieval.
NO LLM involved. All numbers come directly from Open-Meteo APIs.

Public API:
    geocode(city)             -> dict | None
    fetch_weather(lat, lon)   -> dict | None
    get_weather_for_city(city) -> dict  (always returns a status dict)
"""

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Explicit field list — MUST be listed or Open-Meteo returns no values
CURRENT_FIELDS = ",".join([
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "weathercode",
    "visibility",
    "cloud_cover",
])

REQUEST_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode(city: str) -> dict | None:
    """
    Resolve a city name to lat/lon via Open-Meteo Geocoding API.
    Returns the first result as a dict, or None on failure/no results.

    Return shape:
        {
            "lat": float,
            "lon": float,
            "display_name": str,   # e.g. "London, England, United Kingdom"
            "country": str,
            "timezone": str,
        }
    """
    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"name": city, "count": 5, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return None

    results = data.get("results")
    if not results:
        return None

    # Take the first result (best match by Open-Meteo's ranking)
    r = results[0]
    parts = [r.get("name", ""), r.get("admin1", ""), r.get("country", "")]
    display = ", ".join(p for p in parts if p)
    return {
        "lat": r["latitude"],
        "lon": r["longitude"],
        "display_name": display,
        "country": r.get("country", ""),
        "timezone": r.get("timezone", ""),
    }


# ---------------------------------------------------------------------------
# Weather Fetch
# ---------------------------------------------------------------------------

def fetch_weather(lat: float, lon: float) -> dict | None:
    """
    Fetch current weather for the given coordinates.
    Returns raw Open-Meteo 'current' dict, or None on failure.

    The caller receives ONLY numbers that actually came from the API —
    no defaults, no estimations, no model-filled values.
    """
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": CURRENT_FIELDS,
                "wind_speed_unit": "kmh",
                "timezone": "auto",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return None

    current = data.get("current")
    if not current:
        return None

    return current  # raw API response — never modified


# ---------------------------------------------------------------------------
# Combined Helper
# ---------------------------------------------------------------------------

def get_weather_for_city(city: str) -> dict:
    """
    Full pipeline: geocode city -> fetch weather.
    Always returns a dict with a "status" key.

    Success:
        {
            "status": "ok",
            "city": str,
            "display_name": str,
            "lat": float,
            "lon": float,
            "weather": { ...raw Open-Meteo current fields... }
        }

    Failure variants:
        {"status": "geocode_failed", "city": str, "reason": str}
        {"status": "weather_failed", "city": str, "lat": float, "lon": float, "reason": str}
    """
    # Step 1 — geocode
    geo = geocode(city)
    if geo is None:
        return {
            "status": "geocode_failed",
            "city": city,
            "reason": f"Could not resolve '{city}' to a location. "
                      "The city name may be misspelled or not in the geocoding database.",
        }

    # Step 2 — fetch weather
    weather = fetch_weather(geo["lat"], geo["lon"])
    if weather is None:
        return {
            "status": "weather_failed",
            "city": city,
            "lat": geo["lat"],
            "lon": geo["lon"],
            "reason": "Weather data could not be retrieved. "
                      "The Open-Meteo API may be temporarily unavailable.",
        }

    return {
        "status": "ok",
        "city": city,
        "display_name": geo["display_name"],
        "lat": geo["lat"],
        "lon": geo["lon"],
        "weather": weather,  # raw numbers — untouched
    }


# ---------------------------------------------------------------------------
# Weathercode human description (for display only — NOT used in SOP matching)
# ---------------------------------------------------------------------------

WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def describe_weathercode(code: int) -> str:
    """Return a human-readable description for a WMO weathercode."""
    return WMO_DESCRIPTIONS.get(code, f"Unknown condition (code {code})")


# ---------------------------------------------------------------------------
# Standalone test — run via: python -m tools.weather
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    SEPARATOR = "-" * 60

    def run_test(label: str, city: str, fail_fetch: bool = False):
        print(f"\n{SEPARATOR}")
        print(f"TEST: {label}")
        print(SEPARATOR)

        # When running as __main__, patch("tools.weather.fetch_weather") patches
        # the module as imported by name — we must call through that same import
        # so both the patch target and the call site are the same object.
        import tools.weather as _wmod

        if fail_fetch:
            from unittest.mock import patch, MagicMock
            mock = MagicMock(return_value=None)
            with patch.object(_wmod, "fetch_weather", mock):
                result = _wmod.get_weather_for_city(city)
        else:
            result = _wmod.get_weather_for_city(city)

        print(f"  status       : {result['status']}")
        if result["status"] == "ok":
            w = result["weather"]
            print(f"  display_name : {result['display_name']}")
            print(f"  lat/lon      : {result['lat']}, {result['lon']}")
            print(f"  temperature  : {w.get('temperature_2m')} deg C")
            print(f"  wind_speed   : {w.get('wind_speed_10m')} km/h")
            print(f"  wind_gusts   : {w.get('wind_gusts_10m')} km/h")
            print(f"  precipitation: {w.get('precipitation')} mm")
            print(f"  precip_prob  : {w.get('precipitation_probability')} %")
            print(f"  uv_index     : {w.get('uv_index')}")
            print(f"  humidity     : {w.get('relative_humidity_2m')} %")
            code = w.get("weathercode")
            print(f"  condition    : {describe_weathercode(code)} (code={code})")
        else:
            print(f"  reason       : {result.get('reason')}")
        return result

    all_passed = True

    # --- Test 1: Real city (London) ---
    r = run_test("Real city — London", "London")
    if r["status"] != "ok":
        print("  [FAIL] Expected status=ok for London")
        all_passed = False
    else:
        print("  [PASS]")

    # --- Test 2: Real city (Bhopal) ---
    r = run_test("Real city — Bhopal", "Bhopal")
    if r["status"] != "ok":
        print("  [FAIL] Expected status=ok for Bhopal")
        all_passed = False
    else:
        print("  [PASS]")

    # --- Test 3: Non-existent city ---
    r = run_test("Non-existent city — ZXQNOTACITY999", "ZXQNOTACITY999")
    if r["status"] != "geocode_failed":
        print("  [FAIL] Expected status=geocode_failed")
        all_passed = False
    else:
        print("  [PASS]")

    # --- Test 4: Simulated API failure (patch fetch_weather to return None) ---
    r = run_test("Simulated weather API failure", "Paris", fail_fetch=True)
    if r["status"] != "weather_failed":
        print("  [FAIL] Expected status=weather_failed")
        all_passed = False
    else:
        print("  [PASS]")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if all_passed:
        print("CHUNK 2 GATE: ALL TESTS PASSED — ready for Chunk 3")
    else:
        print("CHUNK 2 GATE: SOME TESTS FAILED — fix before proceeding")
        sys.exit(1)
    print("=" * 60)
