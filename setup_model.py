import os
import shutil
from pathlib import Path

def setup_model():
    print("Finding model in cache...")
    cache_dir = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots")
    target_dir = Path("/workspace/AAIPL/hf_models/Qwen/Qwen3-4B")
    
    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return
        
    snapshots = list(cache_dir.iterdir())
    if not snapshots:
        print("❌ No snapshots found inside the cache!")
        return
        
    # The actual model files are inside a folder named with a commit hash
    latest_snapshot = snapshots[0] 
    print(f"✅ Found model files at: {latest_snapshot}")
    
    # Create the exact target directory required by your script
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying files to {target_dir} (This might take a minute or two...)")
    for item in latest_snapshot.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
            
    print("\n✅ Success! config.json and model files are now flattened and correctly placed.")
    print("You can now run test_answer_model.py!")

if __name__ == "__main__":
    setup_model()