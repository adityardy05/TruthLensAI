"""
================================================================
OCR PROCESSOR — Multilingual Image-to-English Text Pipeline
================================================================

Pipeline Flow:
    Image Bytes
        |
        v
    [cv2 Preprocessing]  <-- denoise + adaptive threshold
        |
        v
    [Tesseract OCR]      <-- multi-lang: eng+hin+tel+tam+urd+ben+mar
        |
        v
    [Language Detection] <-- lingua (short text) / langdetect (long text)
        |
        v
    [Google Translate]   <-- only if lang != 'en', chunked for long text
        |
        v
    [LangGraph State]    <-- { original_text, detected_language, english_text, input_type }

================================================================
SYSTEM SETUP (One-time)
================================================================

Windows — Install Tesseract:
    https://github.com/UB-Mannheim/tesseract/wiki
    Then add language packs manually or set the path below.

Linux — Install language packs:
    sudo apt install tesseract-ocr
    sudo apt install tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-tam
    sudo apt install tesseract-ocr-urd tesseract-ocr-ben tesseract-ocr-mar

Python packages:
    pip install pytesseract pillow opencv-python langdetect lingua-language-detector deep-translator

================================================================
"""

import io
import re
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    pytesseract = None
    HAS_PYTESSERACT = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    detect = None
    HAS_LANGDETECT = False

try:
    from lingua import Language, LanguageDetectorBuilder
    _lingua_detector = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH,
        Language.HINDI,
        Language.TELUGU,
        Language.TAMIL,
        Language.URDU,
        Language.BENGALI,
        Language.MARATHI,
    ).build()
    HAS_LINGUA = True
except ImportError:
    _lingua_detector = None
    HAS_LINGUA = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    GoogleTranslator = None
    HAS_TRANSLATOR = False

# ----------------------------------------------------------------
# Language Code Reference
# Language  | Tesseract | langdetect | GoogleTranslator
# ----------|-----------|------------|-----------------
# English   | eng       | en         | en
# Hindi     | hin       | hi         | hi
# Telugu    | tel       | te         | te
# Tamil     | tam       | ta         | ta
# Urdu      | urd       | ur         | ur
# Bengali   | ben       | bn         | bn
# Marathi   | mar       | mr         | mr
# ----------------------------------------------------------------

# Tesseract multi-language string — all supported languages combined
TESSERACT_LANG = "eng+hin+tel+tam+urd+ben+mar"


# ================================================================
# STEP 1: IMAGE PREPROCESSING
# ================================================================

def preprocess_image(image: Image.Image) -> Image.Image:
    if not HAS_CV2:
        return image
    img = np.array(image)
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    return Image.fromarray(thresh)

def run_ocr(image_bytes: bytes) -> str:
    # Attempt pytesseract first if module is present
    if HAS_PYTESSERACT:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            preprocessed = preprocess_image(image)
            text = pytesseract.image_to_string(
                preprocessed,
                lang=TESSERACT_LANG,
                config="--psm 6 --oem 3"
            )
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"[OCRProcessor] Local Tesseract OCR failed: {e}. Falling back to OCR.space API...")

    # Fallback to OCR.space free API
    print("[OCRProcessor] Using OCR.space API fallback...")
    try:
        import requests
        payload = {
            'apikey': 'helloworld',
            'language': 'eng',
        }
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'filename': ('image.png', image_bytes, 'image/png')},
            data=payload,
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            parsed_results = result.get("ParsedResults", [])
            if parsed_results:
                text = parsed_results[0].get("ParsedText", "")
                return text.strip()
            else:
                print(f"[OCRProcessor] OCR.space API response parsed results empty: {result}")
        else:
            print(f"[OCRProcessor] OCR.space API returned HTTP {response.status_code}: {response.text}")
    except Exception as ex:
        print(f"[OCRProcessor] OCR.space API fallback failed: {ex}")

    return ""


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "en"
    word_count = len(text.split())
    if word_count < 100 and HAS_LINGUA and _lingua_detector is not None:
        try:
            result = _lingua_detector.detect_language_of(text)
            if result:
                return result.iso_code_639_1.name.lower()
        except Exception:
            pass
    if HAS_LANGDETECT and detect is not None:
        try:
            return detect(text)
        except Exception:
            pass
    return "en"

def translate_to_english(text: str, source_lang: str, chunk_size: int = 4000) -> str:
    if source_lang == "en" or not HAS_TRANSLATOR or GoogleTranslator is None:
        return text
    if not text.strip():
        return text
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        try:
            translated = GoogleTranslator(source=source_lang, target="en").translate(chunk)
            translated_chunks.append(translated)
        except Exception as e:
            print(f"[OCRProcessor] Translation failed for chunk {idx}: {e}")
            translated_chunks.append(chunk)
    return " ".join(translated_chunks)


# ================================================================
# STEP 5: COMBINED PIPELINE FUNCTION
# ================================================================

def process_image_input(image_bytes: bytes) -> dict:
    """
    Full pipeline: image bytes → English claim text.

    Steps:
        1. OCR (multi-language Tesseract)
        2. Language detection (lingua or langdetect)
        3. Translation to English (GoogleTranslator, chunked)

    :param image_bytes: Raw image bytes (from file upload or Telegram bot).
    :return: Dictionary ready to inject into LangGraph state:
        {
            "original_text"    : str  — raw OCR output in source language,
            "detected_language": str  — ISO 639-1 code ('en', 'hi', etc.),
            "english_text"     : str  — English-translated claim text,
            "input_type"       : str  — always "image"
        }

    NOTE: Keep original_text and detected_language in the LangGraph state.
          The final_judgment node should inform the user if a translation
          occurred (e.g., "This claim was originally in Hindi").
    """
    print("[OCRProcessor] Starting image processing pipeline...")

    # Step 1: OCR
    raw_text = run_ocr(image_bytes)
    if not raw_text:
        print("[OCRProcessor] Warning: No text could be extracted from the image.")
        return {
            "original_text": "",
            "detected_language": "en",
            "english_text": "",
            "input_type": "image"
        }

    print(f"[OCRProcessor] OCR extracted {len(raw_text.split())} words.")

    # Step 2: Language Detection
    detected_lang = detect_language(raw_text)
    print(f"[OCRProcessor] Detected language: '{detected_lang}'")

    # Step 3: Translation
    english_text = translate_to_english(raw_text, detected_lang)
    if detected_lang != "en":
        print(f"[OCRProcessor] Translated from '{detected_lang}' to English.")

    return {
        "original_text": raw_text,
        "detected_language": detected_lang,
        "english_text": english_text,
        "input_type": "image"
    }


# ================================================================
# Quick test — run: python src/preprocessing/ocr_processor.py <image_path>
# ================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_processor.py <path_to_image>")
        print("Example: python ocr_processor.py test_poster.png")
        sys.exit(1)

    image_path = sys.argv[1]

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = process_image_input(image_bytes)

    print("\n" + "=" * 60)
    print(f"Detected Language : {result['detected_language']}")
    print(f"Original Text     :\n{result['original_text']}")
    print("-" * 60)
    print(f"English Text      :\n{result['english_text']}")
    print("=" * 60)
