import requests


def list_tags(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/tags"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        tags = response.json()
        print(f"Tags for {owner}/{repo}:")
        for tag in tags:
            print(f"  {tag['name']}")
    else:
        print(f"Failed to list tags: {response.status_code}")


if __name__ == "__main__":
    list_tags("data-hcp", "lifespan")
