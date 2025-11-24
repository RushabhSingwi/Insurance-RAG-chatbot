import os
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = (
    "https://irdai.gov.in/circulars"
    "?p_p_id=com_irdai_document_media_IRDAIDocumentMediaPortlet"
    "&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view"
    "&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterToDate=12%2F11%2F2025"
    "&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterFromDate=01%2F01%2F2023"
    "&_com_irdai_document_media_IRDAIDocumentMediaPortlet_delta=20"
    "&_com_irdai_document_media_IRDAIDocumentMediaPortlet_resetCur=false"
    "&_com_irdai_document_media_IRDAIDocumentMediaPortlet_cur={page}"
)

OUTPUT_DIR = "data/raw_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Change to 1 for testing
DEBUG_PAGES = None     # None = run all pages; set to 1 for 1st page only

# Remove Hindi characters and symbols
def clean_english_filename(raw):
    # Remove .pdf anywhere first
    raw = raw.replace(".pdf", "")

    # First, remove any non-English characters that are between Hindi characters
    # This handles cases like "व्यवसाय_विभाग" where _ is between Hindi chars
    # Match Hindi char + any non-English chars + Hindi char, and remove the middle part
    raw = re.sub(r"([\u0900-\u097F])[^A-Za-z\u0900-\u097F]+([\u0900-\u097F])", r"\1\2", raw)

    # Now remove all Hindi & non-English characters (except basic separators for now)
    raw = re.sub(r"[^A-Za-z0-9\s\-\(\),\._/]", "", raw)

    # Look for English text after common separators like '_' or '/'
    # Common pattern: "Hindi text _ English text" or "Hindi / English"
    if " _ " in raw:
        # Split by ' _ ' and take the part with mostly English characters
        parts = raw.split(" _ ")
        for part in parts:
            # Check if this part has mostly English characters
            english_chars = len(re.findall(r"[A-Za-z]", part))
            if english_chars > 5:  # Has substantial English content
                raw = part
                break
    elif " / " in raw:
        # Split by ' / ' separator
        parts = raw.split(" / ")
        for part in parts:
            english_chars = len(re.findall(r"[A-Za-z]", part))
            if english_chars > 5:
                raw = part
                break

    # Final cleanup: remove any remaining non-English characters
    raw = re.sub(r"[^A-Za-z0-9\s\-\(\),\._]", "", raw)

    # Clean up multiple spaces and underscores
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"_+", "_", raw)
    raw = raw.strip().strip("_")  # Remove leading/trailing underscores too

    return raw + ".pdf"


def download_pdf(url, file_path):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(r.content)
            return True
        return False
    except Exception as e:
        print("ERROR downloading:", e, url)
        return False


def scrape_page(page_num):
    url = BASE_URL.format(page=page_num)
    print(f"\nFetching page {page_num}: {url}")

    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    rows = soup.select("tr")  # contains 20 PDF rows
    pdf_count = 0

    for row in rows:
        pdf_link_tag = row.select_one('td.table-col-documents a[href*=".pdf"]')
        if not pdf_link_tag:
            continue

        pdf_url = pdf_link_tag["href"]
        raw_name = pdf_link_tag.get_text(strip=True)

        cleaned_name = clean_english_filename(raw_name)
        output_path = os.path.join(OUTPUT_DIR, cleaned_name)

        pdf_count += 1
        print(f"  → Found PDF {pdf_count}: {cleaned_name}")

        if os.path.exists(output_path):
            print("    (already exists, skipping)")
            continue

        # Download
        ok = download_pdf(pdf_url, output_path)
        if ok:
            print(f"    ✓ Downloaded: {cleaned_name}")
        else:
            print(f"    ✗ Failed: {pdf_url}")

        time.sleep(0.5)  # polite delay

    return pdf_count


def main():
    total_downloaded = 0
    MAX_PAGES = 10   # Limit to 10 pages

    for page in range(1, MAX_PAGES + 1):
        if DEBUG_PAGES and page > DEBUG_PAGES:
            break

        count = scrape_page(page)
        total_downloaded += count

        if count == 0:
            print("No PDFs found → stopping.")
            break

    print("\n✓ Completed. Total PDFs downloaded:", total_downloaded)


if __name__ == "__main__":
    main()
