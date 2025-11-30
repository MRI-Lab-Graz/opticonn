import requests
import os
import re

def list_assets(owner, repo, tag, pattern=r".*\.fz$"):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    response = requests.get(url)
    if response.status_code == 200:
        assets = response.json().get('assets', [])
        print(f"Assets for {tag} (first 5 matching {pattern}):")
        count = 0
        for asset in assets:
            if re.match(pattern, asset['name']):
                print(f"  {asset['name']} - {asset['size'] / 1024 / 1024:.2f} MB - {asset['browser_download_url']}")
                count += 1
                if count >= 5:
                    break
    else:
        print(f"Failed to get assets: {response.status_code}")

if __name__ == "__main__":
    list_assets("data-hcp", "lifespan", "hcp-ya")
    list_assets("data-hcp", "lifespan", "dhcp")
