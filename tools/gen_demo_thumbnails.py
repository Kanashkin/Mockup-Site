#!/usr/bin/env python3
"""Re-generate shirt_preview.png / shirt_preview_color.png for the given
mockupN_package dirs using an actual placeholder logo design (not the blank
10x10 transparent image gen_thumbnails.py uses), so the gallery cards show a
demo mockup like the pre-07.09 packages do instead of a bare white shirt.

Usage: python3 tools/gen_demo_thumbnails.py mockup68_package mockup69_package ...
"""
import gc
import os
import sys

from PIL import Image

from server import MockupEngine

THUMB_W = 920
ACCENT = "#2fbf8f"
DEMO_DESIGN_PATH = os.path.join(os.path.dirname(__file__), "..", "demo_design.png")


if __name__ == "__main__":
    pkg_dirs = sys.argv[1:]
    demo = Image.open(DEMO_DESIGN_PATH).convert("RGBA")
    for pd in pkg_dirs:
        name = os.path.basename(pd.rstrip("/"))
        white_path = os.path.join(pd, "shirt_preview.png")
        color_path = os.path.join(pd, "shirt_preview_color.png")
        try:
            eng = MockupEngine(pd)
            img = eng.render(demo, color="#ffffff")
            w, h = img.size
            th = int(h * THUMB_W / w)
            img.resize((THUMB_W, th), Image.LANCZOS).save(white_path, optimize=True)

            img2 = eng.render(demo, color=ACCENT)
            img2.resize((THUMB_W, th), Image.LANCZOS).save(color_path, optimize=True)
            del eng
            gc.collect()
            print(f"OK {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
