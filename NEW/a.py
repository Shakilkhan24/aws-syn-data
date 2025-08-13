import pandas as pd
from pathlib import Path
import time
from google import genai
from tqdm import tqdm
from dotenv import load_dotenv
import os
import sys

# ----------------- LOAD ENV -----------------
load_dotenv()

API_KEYS = os.getenv("API_KEYS").split(",")

TASKS = [
    {
        "input_column": "text",
        "output_column": "trans_bangla",
        "prompt_template": """নিচের ইংরেজি গল্পটি বাংলায় অনুবাদ করুন একজন অভিজ্ঞ বাঙালি সাহিত্যিকের মতো।  
শিশুদের উপযোগী সরল অথচ হৃদয়স্পর্শী ভাষা ব্যবহার করুন, যাতে গল্পের আবেগ, দৃশ্যপট, ও কল্পনার রঙ অক্ষুণ্ণ থাকে।  
বাক্যগুলো হবে ছন্দময় ও সাবলীল, যেন পাঠক গল্পে ডুবে যেতে পারে।  
শব্দচয়নে বাংলার স্বাদ বজায় রাখুন, তবে মূল কাহিনীর ভাব ও প্রবাহ ঠিক রাখবেন।  
শুধুমাত্র অনুবাদিত গল্প দিন, কোনো অতিরিক্ত ব্যাখ্যা বা মন্তব্য যোগ করবেন না।

অনুবাদিত গল্প:
{}"""
    }
]

SNAPSHOT_SIZE = 100

# ----------------- API CLIENT MANAGER -----------------

class GenAIClientManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.index = 0
        self.client = self._create_client(self.api_keys[self.index])

    def _create_client(self, api_key):
        return genai.Client(api_key=api_key)

    def get_client(self):
        return self.client

    def switch_key(self):
        self.index += 1
        if self.index >= len(self.api_keys):
            raise Exception("All API keys exhausted or failed.")
        print(f"Switching to next API key: {self.index + 1}/{len(self.api_keys)}")
        self.client = self._create_client(self.api_keys[self.index])

# ----------------- GENERATE -----------------

def generate_content(client_manager, prompt):
    while True:
        client = client_manager.get_client()
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"API error: {e}. Switching key...")
            client_manager.switch_key()
            time.sleep(1)

# ----------------- PROCESSING -----------------

def is_empty_or_nan(value):
    return pd.isna(value) or pd.isnull(value) or (isinstance(value, str) and value.strip() == "")

def process_csv(file_path, client_manager):
    df = pd.read_csv(file_path)

    for task in TASKS:
        if task["output_column"] not in df.columns:
            df[task["output_column"]] = None

    # Resume: find first unprocessed row
    start_idx = 0
    for idx, row in df.iterrows():
        incomplete = any(is_empty_or_nan(row[task["output_column"]]) for task in TASKS)
        if incomplete:
            start_idx = idx
            break
    else:
        print("All rows already processed.")
        return

    print(f"Resuming from row {start_idx}...")

    for idx in tqdm(range(start_idx, len(df)), desc="Processing rows"):
        row = df.iloc[idx]
        for task in TASKS:
            if is_empty_or_nan(row[task["output_column"]]):
                input_text = row[task["input_column"]]
                prompt = task["prompt_template"].format(input_text)
                output = generate_content(client_manager, prompt)
                df.at[idx, task["output_column"]] = output

        # Save snapshot every SNAPSHOT_SIZE rows
        if (idx + 1) % SNAPSHOT_SIZE == 0:
            snapshot_file = f"batch_{idx + 1}.csv"
            df.to_csv(snapshot_file, index=False, encoding='utf-8-sig')
            print(f"Snapshot saved: {snapshot_file}")

    # Final save
    final_file = f"final_output.csv"
    df.to_csv(final_file, index=False, encoding='utf-8-sig')
    print(f"Final output saved: {final_file}")

# ----------------- ENTRY POINT -----------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python a.py <csv_file>")
        return

    csv_file = sys.argv[1]
    file_path = Path(__file__).parent / csv_file

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    client_manager = GenAIClientManager(API_KEYS)
    process_csv(file_path, client_manager)

if __name__ == "__main__":
    main()
