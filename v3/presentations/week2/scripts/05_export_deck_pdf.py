"""
Export the React deck to a single multi-page PDF.

Usage:
  .venv/bin/python v3/presentations/week2/scripts/05_export_deck_pdf.py

Pre-requisites:
  - Local server running:  python -m http.server 8765 --bind 127.0.0.1
    (run from the repo root, or adjust DECK_URL below).
  - Playwright browsers installed:
      .venv/bin/python -m playwright install chromium

Output:
  v3/presentations/week2/deck.pdf
"""

from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve()
OUT_DIR = HERE.parents[1]
SHOTS_DIR = OUT_DIR / "_pdf_shots"
PDF_PATH = OUT_DIR / "deck.pdf"

DECK_URL = "http://localhost:8765/v3/presentations/week2/react/deck.html"
N_SLIDES = 6
VIEWPORT = {"width": 1920, "height": 1080}


def main():
    SHOTS_DIR.mkdir(exist_ok=True)
    shot_paths = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(DECK_URL, wait_until="networkidle")
        page.wait_for_timeout(800)  # let React mount + fonts settle

        for i in range(N_SLIDES):
            page.wait_for_timeout(500)  # let fade transition finish
            shot = SHOTS_DIR / f"slide_{i + 1:02d}.png"
            page.screenshot(path=str(shot), full_page=False)
            shot_paths.append(shot)
            print(f"  captured slide {i + 1}/{N_SLIDES} → {shot.name}")
            if i < N_SLIDES - 1:
                page.keyboard.press("ArrowRight")

        browser.close()

    # Combine PNGs → multi-page PDF
    imgs = [Image.open(p).convert("RGB") for p in shot_paths]
    imgs[0].save(PDF_PATH, save_all=True, append_images=imgs[1:], resolution=150.0)
    print(f"\nSaved → {PDF_PATH}  ({len(imgs)} pages)")


if __name__ == "__main__":
    main()
