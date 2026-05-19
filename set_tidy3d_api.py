import os
import tidy3d.web as web


API_KEY_ENV = "TIDY3D_API_KEY"


def main():
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV}. In PowerShell, run: $env:{API_KEY_ENV}=\"your_api_key\""
        )

    web.configure(apikey=api_key)
    print("Tidy3D API configured successfully.")
    print(web.get_info())


if __name__ == "__main__":
    main()
