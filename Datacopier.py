import os
import shutil
import requests
import zipfile
import io

# ========================================
# CONFIG
# ========================================

SOURCE_REPO_ZIP = "https://github.com/EGAVSIV/Stock_Scanner_With_ASTA_Parameters/archive/refs/heads/main.zip"

SOURCE_FOLDER_PATH = "Stock_Scanner_With_ASTA_Parameters-main/market_data/sector_index/D"
DESTINATION_FOLDER = "D"


# ========================================
# DOWNLOAD & COPY FUNCTION
# ========================================

def sync_folder():
    print("📥 Downloading source repository...")

    response = requests.get(SOURCE_REPO_ZIP)
    response.raise_for_status()

    print("📦 Extracting files...")
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall("temp_repo")

    source_full_path = os.path.join("temp_repo", SOURCE_FOLDER_PATH)

    if not os.path.exists(source_full_path):
        raise Exception("❌ Source D folder not found!")

    # Remove old D folder if exists
    if os.path.exists(DESTINATION_FOLDER):
        print("🗑 Removing old D folder...")
        shutil.rmtree(DESTINATION_FOLDER)

    print("📁 Copying D folder...")
    shutil.copytree(source_full_path, DESTINATION_FOLDER)

    # Cleanup
    shutil.rmtree("temp_repo")

    print("✅ D folder successfully synced!")


# ========================================
# RUN
# ========================================

if __name__ == "__main__":
    sync_folder()
