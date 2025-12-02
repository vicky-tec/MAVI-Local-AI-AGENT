# gemini_test.py
import os
import traceback
import sys

# load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env in current working directory
except Exception:
    pass

try:
    import google.generativeai as genai
except Exception:
    print("Please install google-generativeai first:")
    print("    pip install --upgrade google-generativeai python-dotenv")
    sys.exit(1)

# Read API key from environment (or .env)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY (or GOOGLE_API_KEY) env var not set.")
    print("Put GEMINI_API_KEY=your_key in a .env file or export it in your shell, then re-run.")
    print()
    print("Examples:")
    print("  Windows PowerShell:")
    print("    $env:GEMINI_API_KEY = 'YOUR_KEY_HERE'")
    print("    python gemini_test.py")
    print()
    print("  Linux / macOS:")
    print("    export GEMINI_API_KEY='YOUR_KEY_HERE'")
    print("    python gemini_test.py")
    sys.exit(1)

# Ensure GOOGLE_API_KEY is set as well (some google libs check this env var)
os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)

# Configure genai client with API key (safe even if ADC is used)
try:
    genai.configure(GEMINI_API_KEY=GEMINI_API_KEY)
except Exception as e:
    print("genai.configure(GEMINI_API_KEY=...) raised an exception (non-fatal):", e)

print("Using GEMINI API key from environment (GEMINI_API_KEY).")
print("NOTE: if using a service-account JSON, set GOOGLE_APPLICATION_CREDENTIALS instead.")
print()

model_names = [
    "gemini-2.5",
    "gemini-2.5-pro",
    "gemini-2.5-vision",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-pro",
]

for mn in model_names:
    try:
        print(f"Trying model: {mn}")
        model = genai.GenerativeModel(mn)
        resp = model.generate_content("Say Hello and identify the model.")
        # successful
        print("SUCCESS:")
        print(resp.text)
        break
    except Exception as ex:
        print(f"Model {mn} failed: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        print("-" * 60)

# If reached end without success, print helpful next steps
else:
    print("All attempts failed. Next steps:")
    print(" 1) Confirm your GEMINI_API_KEY value is correct and has Generative AI access.")
    print(" 2) Ensure the Generative AI API is ENABLED and billing is active in Google Cloud for that key's project.")
    print(" 3) If you use a service-account JSON, set GOOGLE_APPLICATION_CREDENTIALS to its path instead of GEMINI_API_KEY.")
    print(" 4) If your key has restrictions (HTTP referrer / IP), remove them for testing.")
    print(" 5) Paste the printed exception(s) here and I will decode the exact cause.")
