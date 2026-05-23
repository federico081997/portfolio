import base64

import requests
import streamlit as st
from ddgs import DDGS


@st.cache_data(ttl=3600, show_spinner=False)
def get_phone_image_bytes(brand, model_name):
    """
    Searches DuckDuckGo Images, validates image URLs, downloads the
    first usable image, and returns image bytes for Streamlit.

    Parameters
    ----------
    brand : str
        Phone brand name.

    model_name : str
        Phone model name.

    Returns
    -------
    bytes | None
        Image bytes if a valid image is found, otherwise None.
    """
    query = f"{brand} {model_name} smartphone official product photo white background"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": "https://duckduckgo.com/",
    }

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=25))

        for result in results:
            image_url = result.get("image")

            if not image_url or not image_url.startswith("http"):
                continue

            try:
                response = requests.get(
                    image_url,
                    headers=headers,
                    timeout=15,
                )

                content_type = response.headers.get("Content-Type", "")

                if response.status_code != 200:
                    continue

                if not content_type.startswith("image/"):
                    continue

                if len(response.content) < 5000:
                    continue

                return response.content

            except Exception:
                continue

    except Exception as error:
        print(f"Image search failed for {brand} {model_name}: {error}")

    return None


def image_bytes_to_base64(image_bytes):
    """
    Converts image bytes to base64 so the image can be rendered
    inside a fixed-height HTML container.

    Parameters
    ----------
    image_bytes : bytes
        Image bytes.

    Returns
    -------
    str
        Base64 encoded image string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")
