"""
Chunk 1 Gate Test â€” verifies all imports work and .env key loads.
Run: python chunk1_gate.py
"""
import sys
from importlib.metadata import version as pkg_version

print("=" * 50)
print("CHUNK 1 GATE TEST")
print("=" * 50)

errors = []

# Test 1: Core imports
print("\n[1] Checking package imports...")
try:
    import langgraph
    import langchain_google_genai
    import langchain_core
    import yaml
    import requests
    import streamlit
    print(f"    langgraph        : {pkg_version('langgraph')}")
    print(f"    langchain-core   : {pkg_version('langchain-core')}")
    print(f"    requests         : {pkg_version('requests')}")
    print(f"    pyyaml           : {pkg_version('pyyaml')}")
    print(f"    streamlit        : {pkg_version('streamlit')}")
    print("    âœ… All imports OK")
except ImportError as e:
    errors.append(f"Import failed: {e}")
    print(f"    âŒ {e}")

# Test 2: .env key loads
print("\n[2] Checking .env setup...")
try:
    from dotenv import load_dotenv
    import os
    loaded = load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if key and key != "your_gemini_api_key_here":
        print(f"    Key loaded       : True (length={len(key)})")
        print("    âœ… GEMINI_API_KEY found and looks real")
    elif key == "your_gemini_api_key_here":
        errors.append(".env exists but still has the placeholder value â€” replace it with your real key")
        print("    âŒ .env key is still the placeholder. Edit .env and add your real GEMINI_API_KEY.")
    else:
        errors.append("GEMINI_API_KEY not found in .env")
        print("    âŒ GEMINI_API_KEY not found.")
        print("       â†’ Copy .env.example to .env and fill in your key.")
        print("       â†’ Get a free key at: https://aistudio.google.com/app/apikey")
except Exception as e:
    errors.append(str(e))
    print(f"    âŒ {e}")

# Test 3: Project package structure
print("\n[3] Checking package structure...")
packages = ["agent", "tools", "frontend", "eval"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"    {pkg:12s} : âœ… importable")
    except ImportError as e:
        errors.append(f"Package '{pkg}' not importable: {e}")
        print(f"    {pkg:12s} : âŒ {e}")

# Summary
print("\n" + "=" * 50)
if not errors:
    print("âœ… CHUNK 1 GATE: ALL CHECKS PASSED â€” ready for Chunk 2")
else:
    print(f"âŒ CHUNK 1 GATE: {len(errors)} issue(s) found:")
    for i, err in enumerate(errors, 1):
        print(f"   {i}. {err}")
    sys.exit(1)
print("=" * 50)

