import requests
import os


def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Done.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def main():
    output_dir = "/Volumes/Evo/software/braingraph-pipeline/data/mousley_comparison"
    os.makedirs(output_dir, exist_ok=True)

    files = [
        {
            "url": "https://github.com/data-hcp/lifespan/releases/download/hcp-ya/100307.qsdr.fz",
            "name": "100307.qsdr.fz",
        },
        {
            "url": "https://github.com/data-hcp/lifespan/releases/download/dhcp/sub-CC00063AN06_ses-15102_dwi.qsdr.fz",
            "name": "sub-CC00063AN06_ses-15102_dwi.qsdr.fz",
        },
    ]

    for file_info in files:
        dest_path = os.path.join(output_dir, file_info["name"])
        if not os.path.exists(dest_path):
            download_file(file_info["url"], dest_path)
        else:
            print(f"File {dest_path} already exists.")


if __name__ == "__main__":
    main()
