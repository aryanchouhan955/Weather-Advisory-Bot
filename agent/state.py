"""
agent/state.py
--------------
Defines AgentState — the single shared data structure that flows
through every node in the LangGraph graph.

Design principles:
  - `messages` uses LangGraph's add_messages reducer so it ACCUMULATES
    across turns (session memory) rather than being overwritten.
  - All other fields are plain types — each node overwrites only the
    fields it is responsible for.
  - Fields that represent "no value yet" default to None so downstream
    nodes can check them cleanly.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # -----------------------------------------------------------------
    # Conversation memory — accumulates across ALL turns in one session.
    # add_messages reducer appends new messages instead of overwriting.
    # Resets only when the user clicks "Clear conversation" in the UI.
    # -----------------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]

    # -----------------------------------------------------------------
    # Extracted from user message by parse_intent node (LLM)
    # -----------------------------------------------------------------
    city: str                    # e.g. "Bhopal"
    activity: str                # e.g. "cycling to work"
    hour_of_day: int             # 0-23, derived from current local time

    # -----------------------------------------------------------------
    # Set by fetch_weather node (pure Python, no LLM)
    # -----------------------------------------------------------------
    display_name: str            # e.g. "Bhopal, Madhya Pradesh, India"
    lat: float
    lon: float
    weather_data: dict           # raw Open-Meteo numbers — never modified
    weather_error: Optional[str] # set if geocode or API call fails

    # -----------------------------------------------------------------
    # Set by match_sop node (pure Python, no LLM)
    # -----------------------------------------------------------------
    matched_sops: list[dict]     # all matched SOPs, sorted by severity
    primary_sop: Optional[dict]  # highest-severity match (used in reply)

    # -----------------------------------------------------------------
    # Final reply — set by one of: compose_reply, no_sop_reply,
    # weather_fail_reply, or clarification nodes
    # -----------------------------------------------------------------
    reply: str
