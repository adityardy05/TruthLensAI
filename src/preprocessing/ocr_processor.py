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
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Language Detection
from langdetect import detect, DetectorFactory
from lingua import Language, LanguageDetectorBuilder

# Translation
from deep_translator import GoogleTranslator

# ----------------------------------------------------------------
# IMPORTANT: Uncomment and set this if Tesseract is NOT in PATH
# Windows example:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ----------------------------------------------------------------

# Make langdetect deterministic across runs
DetectorFactory.seed = 0

# ----------------------------------------------------------------
# Lingua detector — best for short OCR text (2–5 lines from a poster)
# Supports all languages used in the project
# ----------------------------------------------------------------
_lingua_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.HINDI,
    Language.TELUGU,
    Language.TAMIL,
    Language.URDU,
    Language.BENGALI,
    Language.MARATHI,
).build()

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
    """
    Applies OpenCV preprocessing to improve OCR accuracy on noisy
    photos, posters, and screenshots:

    1. Convert to numpy array (OpenCV format).
    2. Grayscale — reduces color complexity.
    3. FastNlMeansDenoising — removes photo noise before thresholding.
    4. Adaptive Threshold — handles uneven lighting / shadows.
    5. Return as PIL Image for Tesseract.
    """
    img = np.array(image)

    # Handle images that may already be greyscale (2D array)
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold → sharp black text on white background
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    return Image.fromarray(thresh)


# ================================================================
# STEP 2: TESSERACT OCR
# ================================================================

def run_ocr(image_bytes: bytes) -> str:
    """
    Runs multi-language Tesseract OCR on the given image bytes.

    Supported languages: English, Hindi, Telugu, Tamil, Urdu, Bengali, Marathi.

    :param image_bytes: Raw image bytes (from file upload or Telegram).
    :return: Raw extracted text string (may be non-English).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = preprocess_image(image)

    try:
        # PSM 6: Assume uniform block of text (good for posters/screenshots)
        # OEM 3: Default OCR engine (LSTM + legacy)
        text = pytesseract.image_to_string(
            image,
            lang=TESSERACT_LANG,
            config="--psm 6 --oem 3"
        )
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        print(
            "[OCRProcessor] CRITICAL: Tesseract not found.\n"
            "Install Tesseract and set the path in ocr_processor.py."
        )
        return ""
    except Exception as e:
        print(f"[OCRProcessor] Tesseract OCR failed: {e}")
        return ""


# ================================================================
# STEP 3: LANGUAGE DETECTION
# ================================================================

def detect_language(text: str) -> str:
    """
    Detects the language of the given text.

    Strategy:
    - Short text (< 100 words): Use lingua (more accurate for short text).
    - Long text  (≥ 100 words): Use langdetect (faster for longer content).

    :param text: Extracted OCR text.
    :return: ISO 639-1 language code string (e.g., 'en', 'hi', 'te').
             Defaults to 'en' on failure.
    """
    if not text or not text.strip():
        return "en"

    word_count = len(text.split())

    if word_count < 100:
        # lingua is more accurate for short OCR outputs like posters
        result = _lingua_detector.detect_language_of(text)
        if result:
            return result.iso_code_639_1.name.lower()
        return "en"
    else:
        # langdetect is faster for longer article-style OCR output
        try:
            return detect(text)
        except Exception:
            return "en"


# ================================================================
# STEP 4: TRANSLATION TO ENGLISH
# ================================================================

def translate_to_english(text: str, source_lang: str, chunk_size: int = 4000) -> str:
    """
    Translates the extracted text to English using Google Translate.

    - Skips translation if the detected language is already English.
    - Splits long text into chunks (Google Translate has ~5000 char limit per call).

    :param text: Text to translate.
    :param source_lang: ISO 639-1 source language code (e.g., 'hi').
    :param chunk_size: Max characters per translation chunk (default 4000).
    :return: Translated English text, or original text if translation fails.
    """
    if source_lang == "en":
        return text  # No translation needed

    if not text.strip():
        return text

    # Split into chunks to respect Google Translate's character limit
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []

    for idx, chunk in enumerate(chunks, start=1):
        try:
            translated = GoogleTranslator(source=source_lang, target="en").translate(chunk)
            translated_chunks.append(translated)
        except Exception as e:
            print(f"[OCRProcessor] Translation failed for chunk {idx}: {e}")
            translated_chunks.append(chunk)  # Fallback: use original chunk

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
