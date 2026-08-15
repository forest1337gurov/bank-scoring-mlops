import requests

API_RELOAD_URL = "http://api:8000/reload"


def main():
    response = requests.post(API_RELOAD_URL, timeout=60)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()