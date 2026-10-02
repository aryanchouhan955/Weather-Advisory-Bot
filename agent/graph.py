"""
agent/graph.py
--------------
Assembles the LangGraph StateGraph and wires all nodes + conditional edges.

Graph flow:
    START
      |
    parse_intent (LLM: extract city + activity)
      |
      +-- no city found --------> ask_clarification --> END
      |
    fetch_weather (Python: call Open-Meteo)
      |
      +-- API failed -----------> weather_fail_reply --> END
      |
    match_sop (Python: run SOP engine)
      |
      +-- no match found -------> no_sop_reply ------> END
      |
    compose_reply (LLM: write grounded reply)
      |
     END

Session memory: the `messages` field in AgentState accumulates across
invocations within the same session. Call graph.invoke() with the growing
state dict to maintain continuity.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    ask_clarification,
    compose_reply,
    fetch_weather,
    match_sop,
    no_sop_reply,
    parse_intent,
    weather_fail_reply,
)
from agent.state import AgentState


# ---------------------------------------------------------------------------
# Router functions — each returns the NAME of the next node as a string.
# These are pure Python; they never call the LLM.
# ---------------------------------------------------------------------------

def route_after_parse(state: AgentState) -> str:
    """
    After parse_intent: did we get a usable city?
    If not, ask for clarification. If yes, fetch weather.
    """
    city = state.get("city", "").strip()
    if not city:
        return "ask_clarification"
    return "fetch_weather"


def route_after_weather(state: AgentState) -> str:
    """
    After fetch_weather: did the API call succeed?
    If there's an error, give the honest failure reply.
    """
    if state.get("weather_error"):
        return "weather_fail_reply"
    return "match_sop"


def route_after_match(state: AgentState) -> str:
    """
    After match_sop: did any SOP match?
    If nothing matched, give the honest "no policy" reply.
    """
    if not state.get("matched_sops"):
        return "no_sop_reply"
    return "compose_reply"


# ---------------------------------------------------------------------------
# Graph builder — registers nodes and wires edges
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Returns a compiled graph ready to be invoked.
    """
    builder = StateGraph(AgentState)

    # --- Register nodes ---
    builder.add_node("parse_intent",       parse_intent)
    builder.add_node("fetch_weather",      fetch_weather)
    builder.add_node("match_sop",          match_sop)
    builder.add_node("compose_reply",      compose_reply)
    builder.add_node("no_sop_reply",       no_sop_reply)
    builder.add_node("weather_fail_reply", weather_fail_reply)
    builder.add_node("ask_clarification",  ask_clarification)

    # --- Fixed edges ---
    builder.add_edge(START, "parse_intent")
    builder.add_edge("compose_reply",      END)
    builder.add_edge("no_sop_reply",       END)
    builder.add_edge("weather_fail_reply", END)
    builder.add_edge("ask_clarification",  END)

    # --- Conditional (branching) edges ---
    builder.add_conditional_edges(
        "parse_intent",
        route_after_parse,
        {
            "fetch_weather":    "fetch_weather",
            "ask_clarification": "ask_clarification",
        },
    )

    builder.add_conditional_edges(
        "fetch_weather",
        route_after_weather,
        {
            "match_sop":          "match_sop",
            "weather_fail_reply": "weather_fail_reply",
        },
    )

    builder.add_conditional_edges(
        "match_sop",
        route_after_match,
        {
            "compose_reply": "compose_reply",
            "no_sop_reply":  "no_sop_reply",
        },
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph instance — import and use this in your code:
#     from agent.graph import graph
#     result = graph.invoke({"messages": [HumanMessage("...")]})
# ---------------------------------------------------------------------------

graph = build_graph()


# ---------------------------------------------------------------------------
# Standalone gate test — run via: python -m agent.graph
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from langchain_core.messages import HumanMessage

    print("=" * 65)
    print("CHUNK 4 GATE TEST -- LangGraph Agent")
    print("=" * 65)

    all_passed = True

    def run_graph_test(label: str, messages: list, prev_state: dict = None) -> dict:
        """Invoke the graph and return the final state."""
        state = prev_state or {}
        state["messages"] = state.get("messages", []) + messages
        print(f"\n  Invoking graph for: '{messages[-1].content}'")
        result = graph.invoke(state)
        return result

    def check(label, condition, detail=""):
        global all_passed
        status = "[PASS]" if condition else "[FAIL]"
        print(f"  {status} {label}")
        if not condition and detail:
            print(f"         -> {detail}")
        if not condition:
            all_passed = False

    # =========================================================================
    # TEST A — SOP should fire for cycling question
    # =========================================================================
    print("\n" + "-" * 65)
    print("TEST A: SOP fires for cycling safety question")
    print("-" * 65)
    state_a = run_graph_test(
        "SOP fires for cycling",
        [HumanMessage("Is it safe to cycle in Mumbai today?")]
    )

    print(f"\n  city         : {state_a.get('city')}")
    print(f"  activity     : {state_a.get('activity')}")
    print(f"  weather_error: {state_a.get('weather_error')}")
    print(f"  primary_sop  : {state_a.get('primary_sop', {}).get('matched_sop', {}).get('id') if state_a.get('primary_sop') else None}")
    print(f"  reply snippet: {state_a.get('reply', '')[:150]}...")

    check("city was extracted", state_a.get("city") != "")
    check("weather_data is populated (real API numbers)",
          bool(state_a.get("weather_data")))
    check("weather_error is None", state_a.get("weather_error") is None)
    check("reply is not empty", bool(state_a.get("reply")))

    # =========================================================================
    # TEST B — No SOP match for unrecognised activity
    # =========================================================================
    print("\n" + "-" * 65)
    print("TEST B: No SOP match for out-of-scope question")
    print("-" * 65)
    state_b = run_graph_test(
        "No SOP match",
        [HumanMessage("Will the stars be visible tonight in Delhi?")]
    )

    print(f"\n  city         : {state_b.get('city')}")
    print(f"  primary_sop  : {state_b.get('primary_sop')}")
    print(f"  reply snippet: {state_b.get('reply', '')[:150]}...")

    check("primary_sop is None (no match)", state_b.get("primary_sop") is None)
    check("reply says no guidance",
          any(phrase in state_b.get("reply", "").lower()
              for phrase in ["don't have", "no policy", "don't have a safety", "outside our"]),
          detail=f"Actual reply: {state_b.get('reply', '')}")

    # =========================================================================
    # TEST C — API failure (bad city name)
    # =========================================================================
    print("\n" + "-" * 65)
    print("TEST C: Honest failure for unresolvable city")
    print("-" * 65)
    state_c = run_graph_test(
        "API failure / bad city",
        [HumanMessage("Is it safe to jog in ZXQBADCITY999 today?")]
    )

    print(f"\n  city         : {state_c.get('city')}")
    print(f"  weather_error: {state_c.get('weather_error')}")
    print(f"  reply snippet: {state_c.get('reply', '')[:150]}...")

    check("weather_error is set", bool(state_c.get("weather_error")))
    check("reply contains honest failure message",
          any(phrase in state_c.get("reply", "").lower()
              for phrase in ["unable", "couldn't", "could not", "failed", "resolve"]),
          detail=f"Actual reply: {state_c.get('reply', '')}")
    check("weather_data is empty (no invented data)",
          not state_c.get("weather_data"))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 65)
    if all_passed:
        print("CHUNK 4 GATE: ALL TESTS PASSED -- ready for Chunk 5")
    else:
        print("CHUNK 4 GATE: SOME TESTS FAILED -- review output above")
        sys.exit(1)
    print("=" * 65)
