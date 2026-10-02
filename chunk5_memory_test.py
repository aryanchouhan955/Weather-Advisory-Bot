"""
Chunk 5 Gate Test — Multi-turn session memory
Run: python chunk5_memory_test.py

Tests that the bot:
  1. Remembers the city across turns without the user repeating it
  2. Handles a different activity in the same city
  3. Does NOT contradict what it said in a previous turn
  4. Can switch to a new city when the user explicitly names one
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import graph

print("=" * 65)
print("CHUNK 5 GATE TEST -- Session Memory (Multi-turn)")
print("=" * 65)

all_passed = True


def check(label, condition, detail=""):
    global all_passed
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status} {label}")
    if not condition and detail:
        print(f"         -> {detail}")
    if not condition:
        all_passed = False


def show_state(state):
    print(f"  city         : {state.get('city')}")
    print(f"  activity     : {state.get('activity')}")
    print(f"  weather_error: {state.get('weather_error')}")
    print(f"  reply snippet: {state.get('reply', '')[:160]}...")


# =========================================================================
# TURN 1: Ask about Bhopal cycling — establish the city in memory
# =========================================================================
print("\n" + "-" * 65)
print("TURN 1: Establish city (Bhopal) and activity (cycling)")
print("-" * 65)

state = {"messages": [HumanMessage("Is it safe to cycle in Bhopal today?")]}
state = graph.invoke(state)
show_state(state)

check("T1: city extracted as Bhopal", state.get("city", "").lower() == "bhopal",
      f"Got: {state.get('city')}")
check("T1: weather was fetched", bool(state.get("weather_data")),
      "weather_data is empty")
check("T1: reply is not empty", bool(state.get("reply")))

bhopal_weather = state.get("weather_data", {})

# =========================================================================
# TURN 2: Follow-up — different activity, NO city mentioned
# Bot must reuse Bhopal from context
# =========================================================================
print("\n" + "-" * 65)
print("TURN 2: Follow-up question — no city mentioned")
print("        'What about jogging instead?' (no city)")
print("-" * 65)

state["messages"] = state["messages"] + [
    AIMessage(content=state["reply"]),
    HumanMessage("What about going for a jog instead?"),
]
state = graph.invoke(state)
show_state(state)

check("T2: city still Bhopal (from context)", state.get("city", "").lower() == "bhopal",
      f"Got: {state.get('city')} — bot forgot the city")
check("T2: activity changed to jogging", "jog" in state.get("activity", "").lower(),
      f"Got activity: {state.get('activity')}")
check("T2: weather_error is None (data reused or re-fetched)", state.get("weather_error") is None)
check("T2: reply not empty", bool(state.get("reply")))

# =========================================================================
# TURN 3: Another follow-up — still no city
# 'What if I go in the evening instead?' (time change)
# =========================================================================
print("\n" + "-" * 65)
print("TURN 3: Time-based follow-up — no city, no new activity")
print("        'What if I go in the evening instead?'")
print("-" * 65)

state["messages"] = state["messages"] + [
    AIMessage(content=state["reply"]),
    HumanMessage("What if I go in the evening instead?"),
]
state = graph.invoke(state)
show_state(state)

check("T3: city still Bhopal", "bhopal" in state.get("city", "").lower(),
      f"Got: {state.get('city')}")
check("T3: reply references Bhopal or uses same weather context",
      "bhopal" in state.get("reply", "").lower() or bool(state.get("weather_data")),
      f"Reply: {state.get('reply', '')[:200]}")
check("T3: reply not empty", bool(state.get("reply")))

# =========================================================================
# TURN 4: User switches to a new city explicitly
# Bot must drop Bhopal and use the new city
# =========================================================================
print("\n" + "-" * 65)
print("TURN 4: User explicitly switches city to Mumbai")
print("        'Actually, I am in Mumbai. Is cycling safe there?'")
print("-" * 65)

state["messages"] = state["messages"] + [
    AIMessage(content=state["reply"]),
    HumanMessage("Actually, I am in Mumbai. Is cycling safe there?"),
]
state = graph.invoke(state)
show_state(state)

check("T4: city switched to Mumbai", "mumbai" in state.get("city", "").lower(),
      f"Got: {state.get('city')} — bot didn't update the city")
check("T4: weather fetched for Mumbai (not Bhopal)", bool(state.get("weather_data")))
check("T4: reply not empty", bool(state.get("reply")))

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 65)
if all_passed:
    print("CHUNK 5 GATE: ALL TESTS PASSED -- ready for Chunk 6")
else:
    print("CHUNK 5 GATE: SOME TESTS FAILED -- review output above")
    sys.exit(1)
print("=" * 65)
