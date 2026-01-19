import os
from datasets import load_dataset

def download_data():
    print("Downloading DailyDialog dataset...")
    try:
        # 1. Try loading without trust_remote_code (Attempting standard format)
        # 1. trust_remote_code 없이 로드 시도 (표준 형식 시도)
        dataset = load_dataset("daily_dialog")
        
        save_path = "data/dailydialog"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        print(f"Saving dataset to {save_path}...")
        dataset.save_to_disk(save_path)
        print("Download complete!")
        
    except Exception as e:
        print(f"Error downloading daily_dialog: {e}")
        print("Trying alternative: 'knkarthick/dialogsum' (Parquet format)...")
        try:
            # 2. Fallback to a modern, script-free dataset
            # 2. 스크립트가 없는 최신 데이터셋으로 대체 시도
            dataset = load_dataset("knkarthick/dialogsum")
            save_path = "data/dialogsum"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            dataset.save_to_disk(save_path)
            print("Download complete (dialogsum)!")
        except Exception as e2:
            print(f"Error downloading alternative: {e2}")

if __name__ == "__main__":
    download_data()