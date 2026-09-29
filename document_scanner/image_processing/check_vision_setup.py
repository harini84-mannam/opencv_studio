"""
Quick standalone check for the AI vision fallback setup - run this
directly (`python check_vision_setup.py`) from the SAME terminal/shell
you use to run app.py, to confirm the package and API key are both
actually visible to Python before re-testing through the web app.
"""

import os
import sys

print("Python executable:", sys.executable)

try:
    import anthropic
    print("anthropic package: INSTALLED (version", anthropic.__version__, ")")
except ImportError:
    print("anthropic package: NOT INSTALLED  -> run: pip install anthropic")
    sys.exit(1)

key = os.environ.get("ANTHROPIC_API_KEY")
if not key:
    print("ANTHROPIC_API_KEY: NOT SET in this shell/process")
    sys.exit(1)
else:
    print(f"ANTHROPIC_API_KEY: found ({key[:6]}...{key[-4:]}, length {len(key)})")

print("\nBoth checks passed - trying a real, tiny API call now...")

try:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    print("API call succeeded. Response:", response.content[0].text)
    print("\nEverything is working - the vision fallback should trigger correctly now.")
except Exception as exc:
    print("API call FAILED:", exc)
    print("\nThis is the actual error to fix - likely an invalid/expired key,")
    print("no credits/billing on the account, or a network/firewall block.")