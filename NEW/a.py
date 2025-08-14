import pandas as pd
import time
import os
import sys
from google import genai
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
import json

# ----------------- LOAD ENV -----------------
load_dotenv()

API_KEYS = os.getenv("API_KEYS").split(",")
BATCH_SIZE = 100
PROGRESS_FILE = "processing_progress.json"
TASKS = [
    {
        "input_column": "Song",
        "output_column": "new_song",
        "prompt_template": """নিচে একটি গান দেওয়া হলো। আপনার কাজ হলো এই গানটির মূল ভাব, আবেগ এবং বিষয়বস্তুকে অনুপ্রেরণা হিসেবে ব্যবহার করে একটি সম্পূর্ণ নতুন ও মৌলিক গান রচনা করা।

নতুন গানটি যেন প্রদত্ত গানের মতো আবেগ ও বার্তা বহন করে, কিন্তু এর শব্দচয়ন, চিত্রকল্প, এবং গানের গঠন সম্পূর্ণ আলাদা হবে।  
এটি হবে এমন একটি সৃষ্টি, যা অনুপ্রাণিত হলেও স্বতন্ত্র এবং নতুন মনে হবে।  

একজন দক্ষ গীতিকারের মতো কাব্যিক, শ্রুতিমধুর এবং আবেগঘন শব্দ ব্যবহার করে গানটি লিখুন।  
কোনো অতিরিক্ত ব্যাখ্যা বা মন্তব্য যোগ করবেন না। শুধুমাত্র আপনার লেখা নতুন গানটি প্রদান করুন।

প্রদত্ত গান:
{}"""
    }
]


# ----------------- PROGRESS MANAGER -----------------

class ProgressManager:
    def __init__(self, progress_file=PROGRESS_FILE):
        self.progress_file = progress_file
        self.progress_data = self.load_progress()

    def load_progress(self):
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"current_batch": 0, "completed_rows": 0, "total_rows": 0}
        except Exception as e:
            print(f"Error loading progress: {e}")
            return {"current_batch": 0, "completed_rows": 0, "total_rows": 0}

    def save_progress(self, current_batch, completed_rows, total_rows):
        self.progress_data = {
            "current_batch": current_batch,
            "completed_rows": completed_rows,
            "total_rows": total_rows
        }
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")

    def get_resume_point(self):
        return self.progress_data["current_batch"], self.progress_data["completed_rows"]

    def cleanup(self):
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        except Exception as e:
            print(f"Error cleaning up progress file: {e}")


# ----------------- API CLIENT MANAGER -----------------

class GenAIClientManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.key_names = [f"KEY_{i+1}" for i in range(len(api_keys))]
        self.index = 0
        self.request_count = 0
        self.key_usage_stats = {name: 0 for name in self.key_names}
        self.failed_keys = set()
        
        print(f"🔑 Initialized with {len(self.api_keys)} API keys")
        self.client = self._create_client(self.api_keys[self.index])

    def _create_client(self, api_key):
        try:
            client = genai.Client(api_key=api_key)
            current_key_name = self.key_names[self.index]
            print(f"✅ Successfully connected with {current_key_name}")
            return client
        except Exception as e:
            current_key_name = self.key_names[self.index]
            print(f"❌ Error creating client with {current_key_name}: {e}")
            self.failed_keys.add(self.index)
            return None

    def get_client(self):
        return self.client

    def get_current_key_info(self):
        current_key_name = self.key_names[self.index]
        return {
            "name": current_key_name,
            "index": self.index,
            "usage_count": self.key_usage_stats[current_key_name],
            "total_requests": self.request_count
        }

    def increment_usage(self):
        self.request_count += 1
        current_key_name = self.key_names[self.index]
        self.key_usage_stats[current_key_name] += 1

    def switch_key(self):
        old_key_name = self.key_names[self.index]
        self.failed_keys.add(self.index)
        
        # Find next available key
        original_index = self.index
        while True:
            self.index = (self.index + 1) % len(self.api_keys)
            
            # If we've cycled through all keys
            if self.index == original_index:
                available_keys = len(self.api_keys) - len(self.failed_keys)
                if available_keys == 0:
                    raise Exception("❌ All API keys have been exhausted or failed.")
                break
                
            # If this key hasn't failed, try it
            if self.index not in self.failed_keys:
                break
        
        new_key_name = self.key_names[self.index]
        print(f"🔄 Switching from {old_key_name} → {new_key_name}")
        print(f"📊 Available keys: {len(self.api_keys) - len(self.failed_keys)}/{len(self.api_keys)}")
        
        self.client = self._create_client(self.api_keys[self.index])
        
        if self.client is None:
            # If the new key also fails, try switching again
            return self.switch_key()

    def print_usage_stats(self):
        print("\n📈 API Key Usage Statistics:")
        print("=" * 40)
        for key_name in self.key_names:
            usage = self.key_usage_stats[key_name]
            status = "❌ Failed" if self.key_names.index(key_name) in self.failed_keys else "✅ Active"
            current = "🎯 Current" if self.key_names.index(key_name) == self.index else ""
            print(f"{key_name}: {usage} requests | {status} {current}")
        print(f"Total requests made: {self.request_count}")
        print("=" * 40)

# ----------------- GENERATE -----------------

def generate_content(client_manager, prompt):
    max_retries = len(client_manager.api_keys)
    retry_count = 0
    
    while retry_count < max_retries:
        client = client_manager.get_client()
        if client is None:
            print("❌ No valid API client available.")
            return None
            
        current_key_info = client_manager.get_current_key_info()
        
        try:
            # Show which key is being used
            if retry_count == 0:  # Only show on first attempt to avoid spam
                print(f"🔑 Using {current_key_info['name']} (Usage: {current_key_info['usage_count']})")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            # Increment usage counter on successful request
            client_manager.increment_usage()
            return response.text
            
        except genai.errors.ClientError as e:
            retry_count += 1
            print(f"⚠️ API error with {current_key_info['name']}: {e}")
            
            if retry_count < max_retries:
                print("🔄 Attempting to switch to next API key...")
                try:
                    client_manager.switch_key()
                    time.sleep(2)  # Wait a bit before retrying
                except Exception as switch_error:
                    print(f"❌ Failed to switch keys: {switch_error}")
                    return None
            else:
                print("❌ All API keys have been tried and failed.")
                return None
                
        except Exception as e:
            print(f"❌ Unexpected error with {current_key_info['name']}: {e}")
            return None
    
    return None

# ----------------- PROCESSING -----------------

def get_working_file_path(original_file):
    """Get the working file path - creates a working copy if it doesn't exist"""
    working_file = original_file.with_name(f"working_{original_file.name}")
    
    if not working_file.exists():
        # Create working copy from original
        df = pd.read_csv(original_file)
        # Add output columns if they don't exist
        for task in TASKS:
            if task["output_column"] not in df.columns:
                df[task["output_column"]] = None
        df.to_csv(working_file, index=False, encoding='utf-8-sig')
        print(f"Created working file: {working_file}")
    
    return working_file

def process_csv_in_batches(file_path, client_manager):
    progress_manager = ProgressManager()
    working_file = get_working_file_path(file_path)
    
    # Load the working file
    df = pd.read_csv(working_file)
    total_rows = len(df)
    
    # Get resume point
    current_batch, completed_rows = progress_manager.get_resume_point()
    
    if completed_rows > 0:
        print(f"Resuming from batch {current_batch + 1}, row {completed_rows + 1}...")
    else:
        print("Starting fresh processing...")
    
    # Show initial API key status
    print(f"\n🔑 Current API Key: {client_manager.get_current_key_info()['name']}")
    
    # Calculate batches
    start_row = completed_rows
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    
    try:
        for batch_num in range(current_batch, total_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, total_rows)
            
            print(f"\n📦 Processing Batch {batch_num + 1}/{total_batches} (rows {batch_start + 1} to {batch_end})")
            
            # Show current API key status at start of each batch
            key_info = client_manager.get_current_key_info()
            print(f"🔑 Active Key: {key_info['name']} | Usage: {key_info['usage_count']} requests")
            
            # Process each row in the current batch
            batch_progress = tqdm(range(max(batch_start, start_row), batch_end), 
                                desc=f"Batch {batch_num + 1}")
            
            for idx in batch_progress:
                row = df.iloc[idx]
                row_updated = False
                
                # Process each task for this row
                for task in TASKS:
                    if pd.isna(row[task["output_column"]]) or row[task["output_column"]] == "":
                        input_text = row[task["input_column"]]
                        if pd.notna(input_text) and input_text.strip():
                            prompt = task["prompt_template"].format(input_text)
                            output = generate_content(client_manager, prompt)
                            if output:
                                df.at[idx, task["output_column"]] = output
                                row_updated = True
                
                # Update progress after each row
                progress_manager.save_progress(batch_num, idx, total_rows)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            # Save batch completion
            batch_file = f"batch_{batch_num + 1}_complete.csv"
            df.to_csv(batch_file, index=False, encoding='utf-8-sig')
            
            # Update working file
            df.to_csv(working_file, index=False, encoding='utf-8-sig')
            
            print(f"✅ Batch {batch_num + 1} completed and saved: {batch_file}")
            
            # Show API usage stats after each batch
            key_info = client_manager.get_current_key_info()
            print(f"📊 {key_info['name']} used {key_info['usage_count']} times total")
            
            # Reset start_row for next batch
            start_row = 0
            
            # Save progress for completed batch
            progress_manager.save_progress(batch_num + 1, batch_end - 1, total_rows)
    
    except KeyboardInterrupt:
        print("\n⏸️ Processing interrupted by user. Progress has been saved.")
        df.to_csv(working_file, index=False, encoding='utf-8-sig')
        print(f"💾 Current progress saved to: {working_file}")
        client_manager.print_usage_stats()
        return
    
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        df.to_csv(working_file, index=False, encoding='utf-8-sig')
        print(f"💾 Current progress saved to: {working_file}")
        client_manager.print_usage_stats()
        return
    
    # Processing completed successfully
    final_file = file_path.with_name(f"final_{file_path.stem}_output.csv")
    df.to_csv(final_file, index=False, encoding='utf-8-sig')
    print(f"\n🎉 All processing completed! Final output saved: {final_file}")
    
    # Show final API usage statistics
    client_manager.print_usage_stats()
    
    # Clean up progress file
    progress_manager.cleanup()
    print("🧹 Progress tracking cleaned up.")

# ----------------- ENTRY POINT -----------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        print("Example: python script.py songs.csv")
        return

    csv_file = sys.argv[1]
    file_path = Path(__file__).parent / csv_file

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Processing file: {file_path}")
    print(f"📊 Batch size: {BATCH_SIZE} rows")
    print(f"🔑 Available API keys: {len(API_KEYS)}")

    try:
        client_manager = GenAIClientManager(API_KEYS)
        process_csv_in_batches(file_path, client_manager)
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()