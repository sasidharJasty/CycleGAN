import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# Path to your Excel file
excel_file = "catalog.xlsx"

# Folder to save downloaded images
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# Load Excel file
df = pd.read_excel(excel_file)

# Column names (adjust if needed)
url_column = "URL"
title_column = "TITLE"
form_column = "FORM"
type_column = "TYPE"
school_column = "SCHOOL"
timeframe_column = "TIMEFRAME"

# === FILTERS FOR RENAISSANCE ART ===
form_filter = ["painting"]
type_filter = []   # empty means accept all types
school_filter = [] # empty means accept all schools

# Renaissance period (approx.)
start_year = 1400
end_year = 1600

headers = {"User-Agent": "Mozilla/5.0"}

def parse_timeframe(tf_str):
    """Extract start and end years from a timeframe string like '1601-1650'."""
    try:
        parts = tf_str.split("-")
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
        return start, end
    except:
        return None, None

for i, row in df.iterrows():
    page_url = row[url_column]
    title = str(row[title_column]).strip()
    form = str(row[form_column]).strip()
    typ = str(row[type_column]).strip()
    school = str(row[school_column]).strip()
    timeframe = str(row[timeframe_column]).strip()

    if pd.isna(page_url) or pd.isna(title):
        continue

    # Apply categorical filters
    if form_filter and form not in form_filter:
        continue
    if type_filter and typ not in type_filter:
        continue
    if school_filter and school not in school_filter:
        continue

    # Apply timeframe filter (Renaissance range)
    if not pd.isna(timeframe) and timeframe != "nan":
        tf_start, tf_end = parse_timeframe(timeframe)
        if tf_start is not None:
            if start_year is not None and tf_end < start_year:
                continue
            if end_year is not None and tf_start > end_year:
                continue

    try:
        # Fetch webpage
        page = requests.get(page_url, headers=headers, timeout=10)
        page.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(page.text, "html.parser")

        # Find high-res image link
        img_url = None
        for a in soup.find_all("a", href=True):
            if a["href"].lower().endswith(".jpg"):
                img_url = urljoin(page_url, a["href"])
                break

        if not img_url:
            print(f"No JPG found at {page_url}")
            continue

        # Download the image
        img_response = requests.get(img_url, headers=headers, timeout=10)
        img_response.raise_for_status()

        # Filename = artwork title + timeframe
        safe_title = title.replace(" ", "_")
        safe_timeframe = timeframe.replace(" ", "_") if timeframe else "unknown"
        filename = f"{safe_title}_{safe_timeframe}.jpg"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, "wb") as f:
            f.write(img_response.content)

        print(f"Downloaded {filename} from {img_url}")

    except Exception as e:
        print(f"Failed to process {page_url}: {e}")
