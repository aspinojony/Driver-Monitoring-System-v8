import os
import urllib.request
import zipfile


def download_file(url, target_path):
    print(f"Downloading from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)
    print("Download complete.")


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


if __name__ == "__main__":
    RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Using a small, direct-link open source dataset for demonstration (YawDD sample)
    # The actual massive datasets (few GBs) often require manual browser download.
    print("--- 自动下载实验用的小量样例集 ---")

    sample_url = "https://github.com/mrviaom/Driver-Anomaly-Detection/archive/refs/heads/main.zip"
    sample_zip_path = os.path.join(RAW_DATA_DIR, "DAD_sample.zip")

    try:
        download_file(sample_url, sample_zip_path)
        extract_zip(sample_zip_path, RAW_DATA_DIR)
        print("\n✅ 样例集已下载解压至: data/raw/")
    except Exception as e:
        print(f"\n❌ 下载失败，请手动用浏览器下载: {sample_url}")
        print(f"错误信息: {e}")
