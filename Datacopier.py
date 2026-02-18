import os
import shutil
import requests
import zipfile
import io

# ==========================================
# CONFIG
# ==========================================

SOURCE_REPO_ZIP = "https://github.com/EGAVSIV/Stock_Scanner_With_ASTA_Parameters/archive/refs/heads/main.zip"

BASE_SOURCE_PATH = "Stock_Scanner_With_ASTA_Parameters-main"

FOLDERS_TO_SYNC = [
    "stock_data_15",
    "stock_data_1H",
    "stock_data_D",
    "stock_data_M",
    "stock_data_W"
]

DESTINATION_BASE = "."


# ==========================================
# SYNC FUNCTION
# ==========================================

def sync_folders():

    print("📥 Downloading repository...")

    response = requests.get(SOURCE_REPO_ZIP)
    response.raise_for_status()

    print("📦 Extracting repository...")
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall("temp_repo")

    for folder in FOLDERS_TO_SYNC:

        source_path = os.path.join(
            "temp_repo",
            BASE_SOURCE_PATH,
            folder
        )

        destination_path = os.path.join(DESTINATION_BASE, folder)

        if not os.path.exists(source_path):
            print(f"❌ {folder} not found in repo!")
            continue

        if os.path.exists(destination_path):
            print(f"🗑 Removing old {folder}...")
            shutil.rmtree(destination_path)

        print(f"📁 Copying {folder}...")
        shutil.copytree(source_path, destination_path)

    shutil.rmtree("temp_repo")

    print("✅ All folders synced successfully!")


if __name__ == "__main__":
    sync_folders()
