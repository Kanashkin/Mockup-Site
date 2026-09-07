#!/usr/bin/env python3
"""Generate shirt_preview.png (default white, blank) and
shirt_preview_color.png (accent color, blank) for one or more
mockupN_package directories — the two thumbnails index.html's gallery
card expects (see previewHTML() in index.html). Run from the repo root
so `import server` picks up MockupEngine and its dependencies.

Usage: python3 tools/gen_thumbnails.py "#2fbf8f" mockup68_package mockup69_package ...
"""
import gc
import os
import sys

from PIL import Image

from server import MockupEngine

THUMB_W = 920  # matches the existing site's preview thumbnail width


def make_blank_design():
    return Image.new("RGBA", (10, 10), (0, 0, 0, 0))


if __name__ == "__main__":
    accent = sys.argv[1]
    pkg_dirs = sys.argv[2:]
    blank = make_blank_design()
    for pd in pkg_dirs:
        name = os.path.basename(pd.rstrip("/"))
        white_path = os.path.join(pd, "shirt_preview.png")
        color_path = os.path.join(pd, "shirt_preview_color.png")
        if os.path.exists(white_path) and os.path.exists(color_path):
            print(f"SKIP {name} (already has previews)")
            continue
        try:
            eng = MockupEngine(pd)
            if not os.path.exists(white_path):
                img = eng.render(blank, color="#ffffff")
                w, h = img.size
                th = int(h * THUMB_W / w)
                img.resize((THUMB_W, th), Image.LANCZOS).save(white_path, optimize=True)
            if not os.path.exists(color_path):
                img2 = eng.render(blank, color=accent)
                w, h = img2.size
                th = int(h * THUMB_W / w)
                img2.resize((THUMB_W, th), Image.LANCZOS).save(color_path, optimize=True)
            del eng
            gc.collect()
            print(f"OK {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
