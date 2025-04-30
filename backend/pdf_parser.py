import fitz  
import pytesseract
from PIL import Image
import io
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def ocr_page_to_text(page):
    pix = page.get_pixmap(dpi=300)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    return pytesseract.image_to_string(image, lang='eng')

def extract_text_from_pdf(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    logging.info(f"Starting PDF extraction: {file_path}")
    doc = fitz.open(file_path)
    full_text = ""

    for i, page in enumerate(doc):
        try:
            logging.info(f"[Page {i+1}] Extracting...")
            text = page.get_text().strip()
            if text:
                try:
                    text = text.encode("utf-8", errors="ignore").decode()
                    logging.info(f"[Page {i+1}] Native text extracted.")
                    full_text += text + "\n"
                except Exception as enc_err:
                    logging.warning(f"[Page {i+1}] Encoding issue: {enc_err}. Using original text.")
                    full_text += text + "\n"
            else:
                logging.warning(f"[Page {i+1}] No native text found. Using OCR.")
                ocr_text = ocr_page_to_text(page)
                full_text += ocr_text + "\n"

        except Exception as e:
            logging.error(f"[Page {i+1}] Error during extraction: {e}")
            continue

    doc.close()
    logging.info("PDF parsing completed.")
    return full_text.strip()
