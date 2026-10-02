"""
agent/nodes.py
--------------
All node functions for the LangGraph graph.

Node inventory:
    parse_intent        (LLM)  — extracts city + activity from user message
    fetch_weather       (Python) — calls Open-Meteo, stores raw numbers
    match_sop           (Python) — runs SOP engine, ranks matches
    compose_reply       (LLM)  — writes prose reply grounded in SOP + weather
    no_sop_reply        (canned) — fired when no SOP matches
    weather_fail_reply  (canned) — fired when weather API is unreachable

Each node:
  - Takes AgentState
  - Returns a PARTIAL dict of only the fields it sets
  - Never reads or writes fields it doesn't own
  - Has a single, stated responsibility
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.sop_engine import get_all_activity_keywords, load_sops, match_sops
from agent.state import AgentState
from tools.weather import get_weather_for_city

load_dotenv(override=True)  # override=True ensures .env wins over OS-level env vars

# ---------------------------------------------------------------------------
# Shared LLM client — all LLM nodes use this
# Gemini 1.5 Flash: free tier, fast, good instruction-following
# ---------------------------------------------------------------------------

_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # free tier: 1500 RPD
    temperature=0.1,                # low = deterministic, less creative
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ---------------------------------------------------------------------------
# Shared SOP cache — loaded once at module import, not on every request
# ---------------------------------------------------------------------------

_SOPS_DIR = os.path.join(os.path.dirname(__file__), "..", "sops")
_SOPS: list[dict] = load_sops(_SOPS_DIR)

# Flat set of every keyword covered by the SOPs — used in no_sop_reply to
# distinguish "recognized activity with safe conditions" from "out of scope".
_RECOGNIZED_KEYWORDS: set[str] = get_all_activity_keywords(_SOPS)


# ===========================================================================
# NODE 1: parse_intent
# Responsibility: extract city + activity from the latest user message,
#                 using conversation history if needed (follow-up questions).
# Type: LLM call
# ===========================================================================

_PARSE_INTENT_SYSTEM = """You are an intent extraction assistant.
Your ONLY job is to extract two pieces of information from the user's message:
  1. city     - the city or location the user is asking about
  2. activity - what outdoor activity or action the user is asking about

Rules:
- If the user does NOT mention a city in their current message, look at the
  conversation history to find the most recently mentioned city and use that.
- If no city can be found anywhere in the conversation, set city to "".
- If no clear activity is mentioned, set activity to "general outdoor activity".
- Do NOT answer the safety question. Do NOT give advice. Extract ONLY.
- Always respond with valid JSON and nothing else. Example:
  {"city": "Bhopal", "activity": "cycling to work", "city_from_context": false}

city_from_context should be true if you had to get the city from history."""

def parse_intent(state: AgentState) -> dict:
    """
    LLM node: extract city and activity from the latest user message.

    If the city was resolved from conversation history (follow-up question),
    we reuse the previously fetched weather data — no redundant API call.

    Reads:  state["messages"] (full history for context)
    Writes: city, activity, hour_of_day
             + optionally preserves weather_data from previous turn
    """
    messages_for_llm = [SystemMessage(content=_PARSE_INTENT_SYSTEM)]

    # Include full conversation history so follow-up questions work
    messages_for_llm.extend(state["messages"])

    response = _llm.invoke(messages_for_llm)
    raw = response.content
    if isinstance(raw, list):
        raw = raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
    raw = str(raw).strip()

    # Strip markdown fences if the model wraps its JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        city = parsed.get("city", "").strip()
        activity = parsed.get("activity", "general outdoor activity").strip()
        city_from_context = parsed.get("city_from_context", False)
    except (json.JSONDecodeError, KeyError):
        city = ""
        activity = "general outdoor activity"
        city_from_context = False

    # Current hour in UTC — used by SOP conditions like "between 11 and 16"
    hour_of_day = datetime.now(timezone.utc).hour

    # If this is a follow-up (city from context) AND we already have weather
    # for the same city, reuse it — no need to hit the API again.
    prev_city = state.get("city", "").strip().lower()
    prev_weather = state.get("weather_data", {})
    same_city = city.lower() == prev_city if city and prev_city else False
    reuse_weather = city_from_context and same_city and bool(prev_weather)

    updates: dict = {
        "city": city,
        "activity": activity,
        "hour_of_day": hour_of_day,
        "matched_sops": [],
        "primary_sop": None,
        "reply": "",
    }

    if reuse_weather:
        # Preserve previous weather data — skip the fetch_weather node's work
        # by keeping existing weather_data and clearing weather_error
        updates["weather_error"] = None
        # weather_data stays as-is (not reset here — fetch_weather will be skipped
        # only if we route past it, which we do if weather_error stays None)
    else:
        # Fresh turn or new city — clear stale weather values
        updates["weather_data"] = {}
        updates["weather_error"] = None

    return updates



# ===========================================================================
# NODE 2: fetch_weather
# Responsibility: geocode city, call Open-Meteo, store raw numbers.
#                 Never modifies numbers. Sets weather_error on failure.
# Type: Pure Python (no LLM)
# ===========================================================================

def fetch_weather(state: AgentState) -> dict:
    """
    Python node: call Open-Meteo and store raw weather numbers.

    Reads:  state["city"]
    Writes: display_name, lat, lon, weather_data  — OR —  weather_error
    """
    result = get_weather_for_city(state["city"])

    if result["status"] == "ok":
        return {
            "display_name": result["display_name"],
            "lat": result["lat"],
            "lon": result["lon"],
            "weather_data": result["weather"],   # raw API numbers, untouched
            "weather_error": None,
        }
    else:
        # Both geocode_failed and weather_failed go here
        return {
            "display_name": "",
            "lat": 0.0,
            "lon": 0.0,
            "weather_data": {},
            "weather_error": result["reason"],
        }


# ===========================================================================
# NODE 3: match_sop
# Responsibility: run the SOP engine and rank results by severity.
#                 Never uses LLM. Matching is fully deterministic.
# Type: Pure Python (no LLM)
# ===========================================================================

def match_sop(state: AgentState) -> dict:
    """
    Python node: find which SOPs apply to the current weather + activity.

    Reads:  state["weather_data"], state["activity"], state["hour_of_day"]
    Writes: matched_sops, primary_sop
    """
    results = match_sops(
        sops=_SOPS,
        weather=state["weather_data"],
        activity=state["activity"],
        hour=state["hour_of_day"],
    )

    primary = results[0] if results else None

    return {
        "matched_sops": results,
        "primary_sop": primary,
    }


# ===========================================================================
# NODE 4: compose_reply
# Responsibility: write a natural-language reply grounded in the SOP and
#                 real weather numbers. The LLM is a LANGUAGE COMPOSER only —
#                 it cannot change facts, invent numbers, or alter advice.
# Type: LLM call (constrained)
# ===========================================================================

_COMPOSE_REPLY_SYSTEM = """You are a weather safety advisor composing a reply for a user.

STRICT RULES — violation of any rule is unacceptable:
1. Use ONLY the weather numbers from the WEATHER DATA block below.
   Never recall, estimate, or make up any numbers.
2. Base your safety advice ONLY on the SOP ADVICE block below.
   Do not add, soften, or change any advice from the SOP.
3. If multiple SOPs are provided, lead with the highest-severity one.
4. Always end your reply with the citation line(s) from the CITATIONS block.
5. Keep the reply concise, clear, and professional — 3-6 sentences typically.
6. Do not repeat the user's question back to them."""

def compose_reply(state: AgentState) -> dict:
    """
    LLM node: compose a grounded, policy-traceable reply.

    Reads:  state["matched_sops"], state["weather_data"],
            state["display_name"], state["messages"]
    Writes: reply
    """
    matched = state["matched_sops"]
    weather = state["weather_data"]
    location = state["display_name"] or state["city"]

    # Build the grounding block given to the LLM
    # The LLM sees: location, raw weather numbers, SOP advice, citations
    # It does NOT see instructions to invent anything.

    weather_block = "\n".join(
        f"  {k}: {v}" for k, v in weather.items()
        if k != "time"  # skip timestamp, irrelevant to user
    )

    sop_block_parts = []
    citation_parts = []
    for m in matched:
        sop = m["matched_sop"]
        severity_label = m["effective_severity"].upper()
        sop_block_parts.append(
            f"[{severity_label}] {sop['name']} ({sop['id']})\n"
            f"{m['formatted_advice']}"
        )
        citation_parts.append(m["cite"])

    sop_block = "\n\n".join(sop_block_parts)
    citations = "\n".join(citation_parts)

    grounding_message = f"""LOCATION: {location}

WEATHER DATA (from Open-Meteo API — these are the only numbers you may use):
{weather_block}

SOP ADVICE (this is the policy — compose around it, do not change it):
{sop_block}

CITATIONS (include exactly these at the end of your reply):
{citations}"""

    messages_for_llm = [
        SystemMessage(content=_COMPOSE_REPLY_SYSTEM),
        # Include recent conversation turns for continuity (last 6 messages)
        *state["messages"][-6:],
        HumanMessage(content=grounding_message),
    ]

    response = _llm.invoke(messages_for_llm)
    raw = response.content
    if isinstance(raw, list):
        raw = raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
    reply_text = str(raw).strip()

    return {
        "reply": reply_text,
    }


# ===========================================================================
# NODE 5: no_sop_reply
# Responsibility: return a canned "no policy found" message.
#                 Never uses LLM. An honest "I don't know" is correct here.
# Type: Canned string (no LLM, no API)
# ===========================================================================

def no_sop_reply(state: AgentState) -> dict:
    """
    Canned node: fired when no SOP matches.

    Two behaviours depending on whether the activity is recognised:
      - Recognized activity + no hazard fired  -> positive "looks safe" reply
        with real weather numbers and a clear explanation of why no SOP fired.
      - Unknown / out-of-scope activity        -> honest "no policy" reply.

    Reads:  state["activity"], state["display_name"], state["weather_data"]
    Writes: reply
    """
    activity = state.get("activity", "that activity")
    location = state.get("display_name") or state.get("city", "your location")
    weather  = state.get("weather_data", {})

    # Check if the activity matches any keyword in our SOP catalogue
    activity_lower = activity.lower()
    is_recognized = any(kw in activity_lower for kw in _RECOGNIZED_KEYWORDS)

    if is_recognized and weather:
        # No hazard SOP fired → conditions are within safe thresholds
        temp  = weather.get("temperature_2m")
        wind  = weather.get("wind_speed_10m")
        rain  = weather.get("precipitation_probability")
        uv    = weather.get("uv_index")
        gusts = weather.get("wind_gusts_10m")

        reply = (
            f"Based on current conditions in {location}, **{activity} looks safe today**. "
            f"No active weather hazard advisories apply.\n\n"
            f"**Current conditions:** {temp}\u00b0C, wind {wind} km/h "
            f"(gusts up to {gusts} km/h), rain probability {rain}%, UV index {uv}.\n\n"
            f"Standard precautions always apply: stay hydrated, wear appropriate "
            f"clothing, and monitor conditions if you plan to be out for an extended period."
        )
    else:
        # Activity genuinely outside advisory scope
        weather_summary = ""
        if weather:
            temp = weather.get("temperature_2m")
            wind = weather.get("wind_speed_10m")
            rain = weather.get("precipitation_probability")
            if temp is not None:
                weather_summary = (
                    f" (Current conditions in {location}: "
                    f"{temp}\u00b0C, wind {wind} km/h, rain probability {rain}%)"
                )
        reply = (
            f"I don't have a safety policy that covers '{activity}' under the current "
            f"weather conditions.{weather_summary} "
            f"For questions outside our advisory scope, please consult a local "
            f"meteorological authority or relevant safety guideline."
        )

    return {"reply": reply}


# ===========================================================================
# NODE 6: weather_fail_reply
# Responsibility: return a canned "weather data unavailable" message.
#                 Never uses LLM. Never guesses weather.
# Type: Canned string (no LLM, no API)
# ===========================================================================

def weather_fail_reply(state: AgentState) -> dict:
    """
    Canned node: fired when weather data cannot be retrieved.

    Reads:  state["city"], state["weather_error"]
    Writes: reply
    """
    city = state.get("city", "the requested location")
    error = state.get("weather_error", "Unknown error.")

    reply = (
        f"I was unable to retrieve weather data for '{city}'. "
        f"Reason: {error} "
        f"Please verify the city name and try again, or check a local "
        f"weather service directly."
    )

    return {"reply": reply}


# ===========================================================================
# NODE 7: ask_clarification
# Responsibility: ask the user to provide a city when none was found.
#                 Loops back to parse_intent after user responds.
# Type: Canned string (no LLM)
# ===========================================================================

def ask_clarification(state: AgentState) -> dict:
    """
    Canned node: fired when no city was found in the message or history.

    Writes: reply
    """
    reply = (
        "I'd be happy to help with outdoor safety advice! "
        "Could you let me know which city or location you're asking about? "
        "For example: 'Is it safe to cycle in Mumbai today?'"
    )
    return {"reply": reply}
