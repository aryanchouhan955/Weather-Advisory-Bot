"""
frontend/app.py
---------------
Premium Streamlit chat UI for the Weather Advisory Bot.

Run: streamlit run frontend/app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import graph

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Weather Advisory Bot",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — premium dark glassmorphism design
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Load external CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    st.session_state.graph_state = {
        "messages": [], "city": "", "activity": "", "hour_of_day": 12,
        "display_name": "", "lat": 0.0, "lon": 0.0,
        "weather_data": {}, "weather_error": None,
        "matched_sops": [], "primary_sop": None, "reply": "",
    }
    st.session_state.chat_history = []

if "graph_state" not in st.session_state:
    init_session()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_ICONS = {
    "critical": "🔴", "high": "🟠", "medium": "🟡",
    "low": "🟢", "none": "⚪", "variable": "🔵", "safe": "✅",
}

def get_citation_info(state: dict) -> tuple[str, str]:
    matched = state.get("matched_sops", [])
    if not matched:
        # Distinguish safe reply vs truly out-of-scope
        reply = state.get("reply", "")
        if "looks safe today" in reply or "looks safe" in reply:
            return "No hazard advisory — conditions within safe thresholds", "safe"
        return "Outside advisory scope", "none"
    primary = matched[0]
    cite = primary.get("cite", primary["matched_sop"]["id"])
    severity = primary.get("effective_severity", "low")
    if len(matched) > 1:
        others = ", ".join(m["matched_sop"]["id"] for m in matched[1:])
        cite += f"  ·  also: {others}"
    return cite, severity

def badge_class(severity: str) -> str:
    return {
        "critical": "badge-critical", "high": "badge-high",
        "medium": "badge-medium", "low": "badge-low",
        "safe": "badge-safe",
    }.get(severity, "badge-none")

def render_weather_card(weather: dict, display_name: str):
    if not weather:
        return
    temp   = weather.get("temperature_2m", "—")
    feels  = weather.get("apparent_temperature", "—")
    wind   = weather.get("wind_speed_10m", "—")
    gusts  = weather.get("wind_gusts_10m", "—")
    rain   = weather.get("precipitation_probability", "—")
    uv     = weather.get("uv_index", "—")
    humid  = weather.get("relative_humidity_2m", "—")
    vis_raw = weather.get("visibility")
    vis    = f"{int(vis_raw/1000)} km" if vis_raw is not None else "—"

    location_short = display_name.split(",")[0] if display_name else "—"

    st.sidebar.markdown(f"""
<div class="wx-card">
    <div class="wx-location">📍 {location_short}</div>
    <div class="wx-grid">
        <div class="wx-metric">
            <div class="wx-metric-label">Temperature</div>
            <div class="wx-metric-value">{temp}<span class="wx-metric-unit">°C</span></div>
        </div>
        <div class="wx-metric">
            <div class="wx-metric-label">Feels Like</div>
            <div class="wx-metric-value">{feels}<span class="wx-metric-unit">°C</span></div>
        </div>
        <div class="wx-metric">
            <div class="wx-metric-label">Wind</div>
            <div class="wx-metric-value">{wind}<span class="wx-metric-unit">km/h</span></div>
        </div>
        <div class="wx-metric">
            <div class="wx-metric-label">Gusts</div>
            <div class="wx-metric-value">{gusts}<span class="wx-metric-unit">km/h</span></div>
        </div>
        <div class="wx-metric">
            <div class="wx-metric-label">Rain Chance</div>
            <div class="wx-metric-value">{rain}<span class="wx-metric-unit">%</span></div>
        </div>
        <div class="wx-metric">
            <div class="wx-metric-label">UV Index</div>
            <div class="wx-metric-value">{uv}</div>
        </div>
        <div class="wx-metric">
            <div class="wx-metric-label">Humidity</div>
            <div class="wx-metric-value">{humid}<span class="wx-metric-unit">%</span></div>
        </div>
        <div class="wx-metric">
            <div class="wx-metric-label">Visibility</div>
            <div class="wx-metric-value">{vis}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style="padding:20px 4px 0 4px;">
    <div style="font-size:20px;font-weight:700;color:#4ea1ff;letter-spacing:-0.3px;">
        🌤️ Weather Advisory
    </div>
    <div style="font-size:11px;color:#35506a;margin-top:3px;">Powered by Open-Meteo + Gemini</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="thin-divider" style="margin-top:16px;"></div>', unsafe_allow_html=True)

    # Weather panel
    state = st.session_state.graph_state
    weather = state.get("weather_data", {})
    display_name = state.get("display_name", "")

    if weather:
        st.markdown('<div class="sidebar-section">Live Conditions</div>', unsafe_allow_html=True)
        render_weather_card(weather, display_name)
    else:
        st.markdown("""
<div style="text-align:center;padding:30px 10px;color:#2a3a4a;">
    <div style="font-size:36px;margin-bottom:10px;opacity:0.4;">🌍</div>
    <div style="font-size:12px;">Weather data will appear here after your first query.</div>
</div>
""", unsafe_allow_html=True)

    # Session info
    chat_history = st.session_state.chat_history
    if chat_history:
        st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
        city = state.get("city", "")
        activity = state.get("activity", "")
        turns = len([t for t in chat_history if t["role"] == "user"])
        st.markdown(f"""
<div style="font-size:12px;color:#45687a;line-height:1.9;">
    {'📍 ' + city if city else ''}{'<br>' if city else ''}
    {'🏃 ' + activity if activity else ''}{'<br>' if activity else ''}
    💬 {turns} message{'s' if turns != 1 else ''}
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section" style="margin-top:auto;"></div>', unsafe_allow_html=True)
    if st.button("🔄 New Conversation", use_container_width=True):
        init_session()
        st.rerun()

    # About
    st.markdown("""
<div style="margin-top:24px;padding:12px;background:rgba(0,0,0,0.2);border-radius:10px;font-size:11px;color:#304050;line-height:1.7;">
    <strong style="color:#3a5568;">How it works</strong><br>
    Safety decisions use deterministic YAML policies — not LLM guesswork. The AI only composes language around verified data.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-col">', unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">Live Weather · Policy-Grounded · Zero Hallucination</div>
    <h1>Weather Advisory Bot</h1>
    <p class="hero-sub">Ask about outdoor activity safety for any city in the world.</p>
</div>
""", unsafe_allow_html=True)

# Suggested prompts (shown only when no history)
if not st.session_state.chat_history:
    st.markdown("""
<div class="chip-row">
    <span class="chip">🚲 Cycling in Bhopal</span>
    <span class="chip">👧 Kids at park in Chennai</span>
    <span class="chip">🧺 Picnic in Pune</span>
    <span class="chip">👴 Elderly walk in Delhi</span>
    <span class="chip">🚗 Drive to Mumbai</span>
    <span class="chip">🐕 Walk the dog in Jaipur</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────

for turn in st.session_state.chat_history:
    if turn["role"] == "user":
        st.markdown(f"""
<div class="msg-row-user">
    <div class="bubble-user">{turn["content"]}</div>
    <div class="avatar avatar-user">👤</div>
</div>
""", unsafe_allow_html=True)

    else:
        # Format markdown bold (**text**) in reply for display
        content_html = turn["content"].replace("**", "<strong>", 1)
        # Toggle: alternate strong tags
        i = 0
        result_parts = []
        raw = turn["content"]
        while "**" in raw:
            idx = raw.find("**")
            result_parts.append(raw[:idx])
            raw = raw[idx+2:]
            result_parts.append("<strong>" if i % 2 == 0 else "</strong>")
            i += 1
        result_parts.append(raw)
        content_html = "".join(result_parts)
        # Newlines to <br>
        content_html = content_html.replace("\n", "<br>")

        cite = turn.get("cite", "")
        severity = turn.get("severity", "none")
        icon = SEVERITY_ICONS.get(severity, "⚪")
        bc = badge_class(severity)

        st.markdown(f"""
<div class="msg-row-bot">
    <div class="avatar avatar-bot">🤖</div>
    <div class="bubble-bot">{content_html}</div>
</div>
""", unsafe_allow_html=True)

        if cite:
            st.markdown(f"""
<div class="sop-row">
    <span class="sop-badge {bc}">{icon} {cite}</span>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-col

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask about outdoor safety — e.g. 'Is it safe to cycle in Delhi today?'")

if user_input and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    current_state = st.session_state.graph_state
    current_state["messages"] = current_state.get("messages", []) + [
        HumanMessage(content=user_input)
    ]

    with st.spinner("Fetching live weather and checking safety policies…"):
        try:
            result = graph.invoke(current_state)
            st.session_state.graph_state = result

            reply = result.get("reply", "Sorry, I encountered an error.")
            cite, severity = get_citation_info(result)

            st.session_state.chat_history.append({
                "role": "bot",
                "content": reply,
                "cite": cite,
                "severity": severity,
            })

            st.session_state.graph_state["messages"] = result.get("messages", []) + [
                AIMessage(content=reply)
            ]

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "bot",
                "content": f"⚠️ An error occurred: {str(e)}",
                "cite": "System error",
                "severity": "none",
            })

    st.rerun()
