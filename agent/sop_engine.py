"""
agent/sop_engine.py
-------------------
Pure Python SOP loader and matcher. No LLM. No external API calls.

Structure (one function per responsibility):
    load_sops()              -> loads all YAML files from sops/ directory
    evaluate_condition()     -> checks ONE condition against weather data
    activity_matches()       -> checks if activity string matches keyword list
    match_numeric_sop()      -> evaluates a standard numeric SOP
    match_composite_sop()    -> evaluates the fuzzy weighted-score SOP (GEN-001)
    match_override_sop()     -> evaluates the severe weather override (GEN-002)
    match_sops()             -> runs all matchers, returns ranked list
    format_advice()          -> injects live weather values into advice template

Run directly for the gate test:
    python -m agent.sop_engine
"""

import os
import glob
import yaml
from typing import Optional

# ---------------------------------------------------------------------------
# Severity ranking — used to sort matched SOPs (highest first)
# ---------------------------------------------------------------------------

SEVERITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "variable": 1,  # composite SOPs are ranked by their computed level at match time
}


# ---------------------------------------------------------------------------
# 1. SOP Loader
# ---------------------------------------------------------------------------

def load_sops(sop_dir: str) -> list[dict]:
    """
    Load all YAML files from sop_dir and return a flat list of SOP dicts.
    Skips .gitkeep and non-YAML files silently.

    Args:
        sop_dir: absolute or relative path to the sops/ directory

    Returns:
        list of SOP dicts, each guaranteed to have at least: id, name, severity, type
    """
    sops = []
    pattern = os.path.join(sop_dir, "*.yaml")
    files = sorted(glob.glob(pattern))  # sorted for deterministic load order

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            sops.extend(data)

    return sops


# ---------------------------------------------------------------------------
# 2. Condition Evaluators
# ---------------------------------------------------------------------------

def activity_matches(activity: str, keywords: list[str]) -> bool:
    """
    Returns True if any keyword appears in the activity string (case-insensitive).
    This is a substring check, not a word-boundary check — intentionally broad
    so "cycling in the park" matches ["cycling"].

    Args:
        activity: the user's activity string extracted by parse_intent
        keywords: list of keywords from the SOP condition
    """
    activity_lower = activity.lower()
    return any(kw.lower() in activity_lower for kw in keywords)


def evaluate_condition(condition: dict, weather: dict, activity: str, hour: int) -> bool:
    """
    Evaluate a single SOP condition against weather data.

    Supported operators:
        >=, <=, >, <, ==   — numeric comparison against weather[field]
        in                  — weather[field] is in a list of values
        between             — weather[field] is within [min, max] inclusive
        contains_any        — activity string contains any of the listed keywords

    Special field names:
        "activity"          — uses the activity string, not weather dict
        "hour_of_day"       — uses the hour integer, not weather dict

    Args:
        condition: dict with keys: field, operator, value
        weather:   raw Open-Meteo weather dict
        activity:  activity string (e.g., "cycling", "hiking with kids")
        hour:      current hour (0-23)

    Returns:
        True if the condition is satisfied, False otherwise.
        Returns False (not an error) if the field is missing from weather.
    """
    field = condition["field"]
    operator = condition["operator"]
    expected = condition["value"]

    # Special fields
    if field == "activity":
        if operator == "contains_any":
            return activity_matches(activity, expected)
        return False

    if field == "hour_of_day":
        actual = hour
    else:
        actual = weather.get(field)
        if actual is None:
            return False  # field not in API response — condition fails safely

    # Numeric operators
    if operator == ">=":
        return actual >= expected
    if operator == "<=":
        return actual <= expected
    if operator == ">":
        return actual > expected
    if operator == "<":
        return actual < expected
    if operator == "==":
        return actual == expected
    if operator == "in":
        return actual in expected
    if operator == "between":
        lo, hi = expected[0], expected[1]
        return lo <= actual <= hi

    # Unknown operator — fail safely
    return False


# ---------------------------------------------------------------------------
# 3. SOP Type Matchers
# ---------------------------------------------------------------------------

def match_numeric_sop(sop: dict, weather: dict, activity: str, hour: int) -> bool:
    """
    Evaluate a standard numeric SOP.
    All conditions in the `conditions` list must be True (AND logic).

    Returns True if the SOP matches, False otherwise.
    """
    conditions = sop.get("conditions", [])
    if not conditions:
        return False
    return all(evaluate_condition(c, weather, activity, hour) for c in conditions)


def match_composite_sop(sop: dict, weather: dict, activity: str) -> Optional[dict]:
    """
    Evaluate a composite (fuzzy) SOP like GEN-001 (picnic suitability).

    First checks if the activity matches the SOP's keyword list.
    Then scores sub-conditions: each met sub_condition awards its points.
    Maps total score to an advice level.

    Returns a match result dict if activity matches, None otherwise.
    The result dict includes the computed advice_level and score.
    """
    # Check activity keywords first
    keywords = sop.get("activity_keywords", [])
    if not activity_matches(activity, keywords):
        return None

    # Score sub-conditions
    sub_conditions = sop.get("sub_conditions", [])
    total_points = sum(sc.get("points", 1) for sc in sub_conditions)
    earned_points = 0
    met_labels = []
    missed_labels = []

    for sc in sub_conditions:
        condition = {
            "field": sc["field"],
            "operator": sc["operator"],
            "value": sc["value"],
        }
        if evaluate_condition(condition, weather, activity, hour=12):  # hour irrelevant here
            earned_points += sc.get("points", 1)
            met_labels.append(sc.get("label", sc["field"]))
        else:
            missed_labels.append(sc.get("label", sc["field"]))

    # Map score to advice level
    ratio = earned_points / total_points if total_points > 0 else 0
    if ratio >= 0.8:
        level = "great"
        effective_severity = "low"
    elif ratio >= 0.6:
        level = "acceptable"
        effective_severity = "low"
    elif ratio >= 0.4:
        level = "not_ideal"
        effective_severity = "medium"
    else:
        level = "avoid"
        effective_severity = "high"

    return {
        "sop": sop,
        "advice_level": level,
        "effective_severity": effective_severity,
        "score": earned_points,
        "max_score": total_points,
        "met_conditions": met_labels,
        "missed_conditions": missed_labels,
    }


def match_override_sop(sop: dict, weather: dict) -> bool:
    """
    Evaluate a severe weather override SOP like GEN-002.
    Fires when at least `min_triggers` of the trigger_conditions are True.

    Returns True if enough triggers are met, False otherwise.
    """
    triggers = sop.get("trigger_conditions", [])
    min_required = sop.get("min_triggers", 2)

    met = sum(
        1 for t in triggers
        if evaluate_condition(t, weather, activity="", hour=12)
    )
    return met >= min_required


# ---------------------------------------------------------------------------
# 4. Advice Formatter
# ---------------------------------------------------------------------------

def format_advice(advice_template: str, weather: dict) -> str:
    """
    Replace {field_name} placeholders in the advice template with actual
    weather values from the API response.

    Only injects values that came from the API — if a field is missing,
    the placeholder is replaced with "[data unavailable]" rather than
    silently keeping the placeholder or fabricating a number.

    Args:
        advice_template: raw advice string with {field_name} placeholders
        weather:         raw Open-Meteo weather dict

    Returns:
        Formatted advice string with real values injected.
    """
    result = advice_template
    # Find all {field_name} patterns
    import re
    placeholders = re.findall(r"\{(\w+)\}", advice_template)
    for field in placeholders:
        value = weather.get(field)
        if value is not None:
            result = result.replace(f"{{{field}}}", str(value))
        else:
            result = result.replace(f"{{{field}}}", "[data unavailable]")
    return result.strip()


# ---------------------------------------------------------------------------
# 5. Main Matcher — combines all type matchers
# ---------------------------------------------------------------------------

def match_sops(
    sops: list[dict],
    weather: dict,
    activity: str,
    hour: int,
) -> list[dict]:
    """
    Run all SOPs against the current weather + activity and return
    a list of matched results, sorted by severity (critical first).

    Each result dict contains:
        matched_sop      : the SOP dict
        effective_severity: severity string used for sorting
        formatted_advice : advice with live weather values injected
        cite             : citation string (SOP ID + name)
        match_type       : "numeric" | "composite" | "override"
        extra            : additional info (e.g., composite score details)

    Args:
        sops:     list of SOP dicts (from load_sops)
        weather:  raw Open-Meteo weather dict
        activity: activity string (e.g., "cycling to work")
        hour:     current hour of day (0-23)

    Returns:
        List of match result dicts, sorted highest severity first.
        Returns [] if nothing matches.
    """
    results = []

    for sop in sops:
        sop_type = sop.get("type", "numeric")

        # --- Override SOPs (e.g., GEN-002) ---
        if sop_type == "override":
            if match_override_sop(sop, weather):
                advice = format_advice(sop.get("advice", ""), weather)
                results.append({
                    "matched_sop": sop,
                    "effective_severity": sop.get("severity", "high"),
                    "formatted_advice": advice,
                    "cite": sop.get("cite", sop["id"]),
                    "match_type": "override",
                    "extra": {},
                })

        # --- Composite SOPs (e.g., GEN-001) ---
        elif sop_type == "composite":
            composite_result = match_composite_sop(sop, weather, activity)
            if composite_result is not None:
                level = composite_result["advice_level"]
                advice_template = sop.get("advice_levels", {}).get(level, "")
                advice = format_advice(advice_template, weather)
                results.append({
                    "matched_sop": sop,
                    "effective_severity": composite_result["effective_severity"],
                    "formatted_advice": advice,
                    "cite": sop.get("cite", sop["id"]),
                    "match_type": "composite",
                    "extra": {
                        "advice_level": level,
                        "score": composite_result["score"],
                        "max_score": composite_result["max_score"],
                        "met_conditions": composite_result["met_conditions"],
                        "missed_conditions": composite_result["missed_conditions"],
                    },
                })

        # --- Numeric SOPs (default) ---
        else:
            if match_numeric_sop(sop, weather, activity, hour):
                advice = format_advice(sop.get("advice", ""), weather)
                results.append({
                    "matched_sop": sop,
                    "effective_severity": sop.get("severity", "medium"),
                    "formatted_advice": advice,
                    "cite": sop.get("cite", sop["id"]),
                    "match_type": "numeric",
                    "extra": {},
                })

    # Sort: critical(4) > high(3) > medium(2) > low(1)
    results.sort(key=lambda r: SEVERITY_RANK.get(r["effective_severity"], 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Standalone test — run via: python -m agent.sop_engine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Resolve sops directory relative to this file
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    SOPS_DIR = os.path.join(THIS_DIR, "..", "sops")

    print("=" * 65)
    print("CHUNK 3 GATE TEST — SOP Engine")
    print("=" * 65)

    # Load SOPs
    sops = load_sops(SOPS_DIR)
    print(f"\nLoaded {len(sops)} SOPs from {SOPS_DIR}")
    for s in sops:
        print(f"  [{s['id']}] {s['name']} (severity={s['severity']}, type={s['type']})")

    all_passed = True
    SEPARATOR = "-" * 65

    def run_case(label, weather, activity, hour, expect_ids=None, expect_none=False):
        """Helper to run one test case and print results."""
        global all_passed
        print(f"\n{SEPARATOR}")
        print(f"CASE: {label}")
        print(f"      activity='{activity}', hour={hour}")
        key_fields = ["wind_speed_10m", "uv_index", "precipitation_probability",
                      "temperature_2m", "apparent_temperature", "weathercode",
                      "precipitation", "wind_gusts_10m", "visibility"]
        print(f"      weather={ {k: weather[k] for k in key_fields if k in weather} }")
        print(SEPARATOR)

        matched = match_sops(sops, weather, activity, hour)

        if matched:
            print(f"  Matched {len(matched)} SOP(s):")
            for m in matched:
                print(f"    -> [{m['matched_sop']['id']}] {m['matched_sop']['name']}"
                      f" (severity={m['effective_severity']}, type={m['match_type']})")
                if m["match_type"] == "composite":
                    ex = m["extra"]
                    print(f"       Score: {ex['score']}/{ex['max_score']}"
                          f" | level={ex['advice_level']}")
                    print(f"       Met: {ex['met_conditions']}")
                    print(f"       Missed: {ex['missed_conditions']}")
                print(f"       Advice preview: {m['formatted_advice'][:120]}...")
        else:
            print("  No SOPs matched.")

        matched_ids = [m["matched_sop"]["id"] for m in matched]

        if expect_none:
            if not matched:
                print("  [PASS] Correctly returned no match.")
            else:
                print(f"  [FAIL] Expected no match, got: {matched_ids}")
                all_passed = False
        else:
            ids_to_check = expect_ids or []
            hits = [eid for eid in ids_to_check if eid in matched_ids]
            if len(hits) == len(ids_to_check):
                print(f"  [PASS] All expected SOPs matched: {ids_to_check}")
            else:
                missing = [eid for eid in ids_to_check if eid not in matched_ids]
                print(f"  [FAIL] Missing expected SOPs: {missing}")
                print(f"         Got: {matched_ids}")
                all_passed = False

    # =========================================================================
    # Test Vector 1 — High wind + cycling → should match OE-001
    # =========================================================================
    run_case(
        label="High wind cycling (wind=55 km/h)",
        weather={
            "wind_speed_10m": 55.0, "wind_gusts_10m": 70.0,
            "precipitation": 0.0, "precipitation_probability": 10,
            "uv_index": 3.0, "temperature_2m": 22.0, "apparent_temperature": 20.0,
            "weathercode": 1, "relative_humidity_2m": 50, "visibility": 10000,
            "cloud_cover": 20,
        },
        activity="cycling to work",
        hour=9,
        expect_ids=["OE-001"],
    )

    # =========================================================================
    # Test Vector 2 — High UV at midday + cycling → should match both OE-002
    #                 and OE-001 (wind is also high) — OE-002 listed first (both high)
    # =========================================================================
    run_case(
        label="High UV midday + high wind cycling",
        weather={
            "wind_speed_10m": 45.0, "wind_gusts_10m": 60.0,
            "precipitation": 0.0, "precipitation_probability": 5,
            "uv_index": 9.5, "temperature_2m": 30.0, "apparent_temperature": 32.0,
            "weathercode": 0, "relative_humidity_2m": 30, "visibility": 10000,
            "cloud_cover": 5,
        },
        activity="cycling in the park",
        hour=13,
        expect_ids=["OE-001", "OE-002"],
    )

    # =========================================================================
    # Test Vector 3 — Picnic conditions (fuzzy composite GEN-001)
    #                 Good conditions → should match GEN-001 with 'great' level
    # =========================================================================
    run_case(
        label="Perfect picnic day (all conditions met)",
        weather={
            "wind_speed_10m": 10.0, "wind_gusts_10m": 15.0,
            "precipitation": 0.0, "precipitation_probability": 10,
            "uv_index": 4.0, "temperature_2m": 24.0, "apparent_temperature": 23.0,
            "weathercode": 1, "relative_humidity_2m": 45, "visibility": 15000,
            "cloud_cover": 25,
        },
        activity="picnic in the park with family",
        hour=11,
        expect_ids=["GEN-001"],
    )

    # =========================================================================
    # Test Vector 4 — Severe weather system (GEN-002 override)
    #                 High precip + high gusts → should match GEN-002
    # =========================================================================
    run_case(
        label="Severe weather override (heavy rain + gusts)",
        weather={
            "wind_speed_10m": 40.0, "wind_gusts_10m": 58.0,
            "precipitation": 15.0, "precipitation_probability": 90,
            "uv_index": 0.0, "temperature_2m": 22.0, "apparent_temperature": 19.0,
            "weathercode": 63, "relative_humidity_2m": 95, "visibility": 500,
            "cloud_cover": 100,
        },
        activity="jogging outside",
        hour=14,
        expect_ids=["GEN-002"],
    )

    # =========================================================================
    # Test Vector 5 — No SOP should match (clear calm evening, moon-watching)
    # =========================================================================
    run_case(
        label="No match — calm night, unrecognised activity",
        weather={
            "wind_speed_10m": 5.0, "wind_gusts_10m": 8.0,
            "precipitation": 0.0, "precipitation_probability": 2,
            "uv_index": 0.0, "temperature_2m": 18.0, "apparent_temperature": 17.0,
            "weathercode": 0, "relative_humidity_2m": 55, "visibility": 20000,
            "cloud_cover": 5,
        },
        activity="watching the moon and stars",
        hour=22,
        expect_none=True,
    )

    # =========================================================================
    # Test Vector 6 — Elderly person walking in extreme heat → VG-002
    # =========================================================================
    run_case(
        label="Elderly person walking in extreme heat",
        weather={
            "wind_speed_10m": 8.0, "wind_gusts_10m": 12.0,
            "precipitation": 0.0, "precipitation_probability": 0,
            "uv_index": 7.0, "temperature_2m": 39.0, "apparent_temperature": 43.0,
            "weathercode": 0, "relative_humidity_2m": 20, "visibility": 15000,
            "cloud_cover": 10,
        },
        activity="walking my grandma outside",
        hour=14,
        expect_ids=["VG-002"],
    )

    # Summary
    print(f"\n{'=' * 65}")
    if all_passed:
        print("CHUNK 3 GATE: ALL TESTS PASSED -- ready for Chunk 4")
    else:
        print("CHUNK 3 GATE: SOME TESTS FAILED -- fix before proceeding")
        sys.exit(1)
    print("=" * 65)
