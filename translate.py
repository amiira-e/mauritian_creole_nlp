import json
import os
import time
from google.cloud import translate_v2 as translate
from tqdm import tqdm

# --------------------------------------------------
# Google Cloud credentials
# --------------------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    r"C:\Users\ae\OneDrive\thesis\project-4f7d28ef-96f0-423e-ac1-7c3c5c8fc68c.json"
)

client = translate.Client()

# --------------------------------------------------
# Files
# --------------------------------------------------
INPUT_FILE = r"data\Oomar_standardized.json"
OUTPUT_FILE = r"data\Oomar_standardized_translated.json"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# --------------------------------------------------
# Translate
# --------------------------------------------------
for item in tqdm(data):

    # Skip if already translated
    if item.get("translated", "").strip():
        continue

    try:
        result = client.translate(
            item["input"],
            source_language="mfe",
            target_language="en"
        )

        # Save translation
        item["translated"] = result["translatedText"]

        # Save progress after every translation
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # Small delay
        time.sleep(0.05)

    except Exception as e:
        print(f"Error on ID {item['id']}: {e}")

# --------------------------------------------------
# Final save
# --------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Translation completed successfully!")