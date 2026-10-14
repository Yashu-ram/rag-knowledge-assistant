"""Install local English OCR language data for PyMuPDF's built-in OCR engine."""

from pathlib import Path
from urllib.request import urlretrieve

target = Path(__file__).resolve().parents[1] / "models/tessdata/eng.traineddata"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.is_file():
    urlretrieve("https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata", target)
print(f"OCR language data ready: {target}")
