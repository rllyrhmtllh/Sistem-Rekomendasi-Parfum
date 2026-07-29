import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
 
PLACEHOLDER_IMAGE = "https://placehold.co/400x500/1c1c1c/d4af37?text=Parfum"
 
 
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_product_image_url(product_url: str) -> str:
    """Ambil URL gambar (og:image) dari halaman produk. Fallback ke placeholder."""
    if not isinstance(product_url, str) or not product_url.startswith("http"):
        return PLACEHOLDER_IMAGE
 
    try:
        resp = requests.get(product_url, headers=HEADERS, timeout=6)
        if resp.status_code != 200:
            print(f"[IMG ERROR] Status {resp.status_code} untuk {product_url}")
            return PLACEHOLDER_IMAGE
 
        soup = BeautifulSoup(resp.text, "html.parser")
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            return og_image["content"]
 
        # fallback: cari tag <img> pertama yang relevan dengan nama file parfum
        img_tag = soup.find("img", src=re.compile(r"fimgs\.net|perfume", re.IGNORECASE))
        if img_tag and img_tag.get("src"):
            return img_tag["src"]
 
        return PLACEHOLDER_IMAGE
    except Exception as e:
        print(f"Gagal ambil gambar dari {product_url}: {e}")
        return PLACEHOLDER_IMAGE