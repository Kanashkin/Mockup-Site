#!/usr/bin/env python3
"""Generate small JPEG gallery-grid thumbnails from each package's full-size
preview PNGs.

BUG (2026-09-14, "какого размера привью показваешь?"): the homepage gallery
grid, the "N photos" model-group browsing view, and the "You might like" grid
(all three share `previewHTML()` in index.html) were loading `shirt_preview.png`
directly — the SAME file also used as the OG/social-share image. That file is
a full, uncompressed PNG at up to 920x1380px and up to ~2.3MB (mockup1), while
`.gallery-grid img`'s own CSS caps the displayed tile at `minmax(210px,1fr)`
with `aspect-ratio:460/690; object-fit:cover` — so the browser was downloading
a multi-hundred-KB-to-multi-MB image just to show a ~200px-wide cropped
portrait tile. Same category of waste as the 2026-09-14 editor-preview
load-speed pass (PNG for photographic content, full size for a tiny display
slot), just a different, previously-uncovered asset/pipeline.

FIX: this script generates a `_thumb.jpg` sibling next to every `shirt_preview*.png`
(base + each color hoverPreview variant that actually exists in a given
package), pre-cropped to the CSS's own 460:690 aspect ratio (at 480x720, a
comfortable retina-safe size for the grid's largest realistic column width)
using the SAME anchor as the CSS (`object-fit:cover` + `object-position:top`
== resize-to-cover then crop excess from the bottom, never the top), so the
browser doesn't have to crop anything itself and never fetches pixels that
were just going to be cropped away. JPEG quality 82 — these are ordinary
photos, no alpha channel to preserve (matches the reasoning already used for
`shirt_base_preview.jpg` in the editor-preview pass).

`shirt_preview.png` itself is UNTOUCHED — it stays exactly as-is for the
OG/Twitter share image and for `sitemap`/social-card purposes, where full
resolution actually matters.

BUG 2 (2026-09-15, "у тебя там люди не по центру"): many of the 11.09-batch
vendor reference photos are landscape (920x613, ratio 1.5) but the gallery
tile is portrait (460:690, ratio 0.667). Cropping the sides down to width
`h * TARGET_RATIO` and centering that window on the SOURCE's geometric
midpoint assumes the person is dead-center in the frame — but these are
lifestyle/product shots that often leave negative space to one side (for a
logo/text overlay), so a blind center crop can cut the subject off-center
or clip an arm/shoulder while showing a slab of empty background.
FIX: detect faces (`cv2`'s bundled Haar frontal-face cascade) and center the
crop window on the TOPMOST detected face — for a standing subject the head
is almost always the highest point of the body, and picking "topmost" among
multiple candidates is far more robust than picking "largest": the largest
detected box is sometimes a false positive on a busy background (gym
equipment, a paper flower wall) that happens to score bigger than the real,
smaller face box, which "topmost" reliably ignores since backgrounds rarely
sit above a standing person's own head. Falls back to the old geometric
center when no face is found at all (e.g. sunglasses defeat the frontal
cascade) — verified this fallback is no worse than before on a sample with
no face (a man wearing sunglasses, mockup1015). Verified on 6 sample
packages spanning kids/teen/man/woman sets before shipping to all 208:
correct fixes on 2 packages that were visibly bad before (a girl pushed to
the right of frame, a teen almost entirely cropped out except a staircase),
no regression on the other 4 (including two false-positive-prone
backgrounds that "topmost" correctly ignored in favor of the real face).

Usage: python3 tools/gen_gallery_thumbs.py [--force] <package_dir> [<package_dir> ...]
       python3 tools/gen_gallery_thumbs.py --force --all   # every mockup*_package dir here
"""
import os
import sys
import glob
import cv2
from PIL import Image, ImageFile

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# mockup99_package/shirt_preview_color.png is a pre-existing corrupt source
# file on disk (truncated PNG data stream, unrelated to this script) that is
# ALREADY being served as-is by the live site's hover-preview today. Without
# this, PIL refuses to decode it at all and the whole package gets skipped.
# With it, PIL decodes what it can (missing the last ~2% of rows at the very
# bottom of the frame) so a thumbnail still gets produced instead of leaving
# this package with a broken hover image. This does not fix the underlying
# corrupt source file — that's a separate, pre-existing data issue worth
# re-uploading independently.
ImageFile.LOAD_TRUNCATED_IMAGES = True

THUMB_W, THUMB_H = 480, 720  # matches index.html's aspect-ratio:460/690
TARGET_RATIO = THUMB_W / THUMB_H

# Every filename previewHTML() in index.html can possibly reference: the
# default base image, plus every hoverPreview value used anywhere in the
# MOCKUPS array (grep hoverPreview:'... in index.html to keep this in sync).
SOURCE_NAMES = [
    "shirt_preview.png",
    "shirt_preview_color.png",
    "shirt_preview_blue.png",
    "shirt_preview_green.png",
    "shirt_preview_orange.png",
    "shirt_preview_pink.png",
    "shirt_preview_purple.png",
    "shirt_preview_red.png",
    "shirt_preview_teal.png",
    "shirt_preview_yellow.png",
]


def thumb_name(src_name):
    base, _ = os.path.splitext(src_name)
    return base + "_thumb.jpg"


def _detect_face_center_x(src_path):
    """Return the x-center of the TOPMOST detected face, or None if no face
    is found. Topmost (smallest y), not largest, since a standing subject's
    head is reliably the highest point of the body, while the largest box
    among several candidates is sometimes a false positive on a busy
    background that happens to score bigger than the real, smaller face."""
    cv_img = cv2.imread(src_path)
    if cv_img is None:
        return None
    gray = cv2.equalizeHist(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6, minSize=(50, 50)
    )
    if len(faces) == 0:
        return None
    x, y, fw, fh = min(faces, key=lambda f: f[1])
    return x + fw / 2


def make_thumb(src_path, dst_path):
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    src_ratio = w / h
    if src_ratio > TARGET_RATIO:
        # Source is relatively wider than the target box: crop the sides.
        # Center on the detected face when we can find one (see BUG 2 above)
        # since these sources often aren't shot with the subject dead-center;
        # otherwise fall back to the CSS's own default (geometric center).
        new_w = round(h * TARGET_RATIO)
        cx = _detect_face_center_x(src_path)
        if cx is not None:
            x0 = int(round(cx - new_w / 2))
            x0 = max(0, min(w - new_w, x0))
        else:
            x0 = (w - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, h))
    elif src_ratio < TARGET_RATIO:
        # Source is relatively taller/narrower: crop the bottom only, keep
        # the top — matches the CSS's object-position:top.
        new_h = round(w / TARGET_RATIO)
        img = img.crop((0, 0, w, new_h))
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    img.save(dst_path, "JPEG", quality=82, optimize=True)


def process(pkg_dir, force=False):
    name = os.path.basename(pkg_dir.rstrip("/"))
    made = 0
    for src_name in SOURCE_NAMES:
        src_path = os.path.join(pkg_dir, src_name)
        if not os.path.exists(src_path):
            continue
        dst_path = os.path.join(pkg_dir, thumb_name(src_name))
        if os.path.exists(dst_path) and not force:
            continue
        make_thumb(src_path, dst_path)
        made += 1
    print(f"OK {name} ({made} thumb(s))")


if __name__ == "__main__":
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    if "--all" in args:
        targets = sorted(d for d in glob.glob("mockup*_package") if os.path.isdir(d))
    else:
        targets = args
    for pd in targets:
        try:
            process(pd, force=force)
        except Exception as e:
            print(f"FAIL {pd}: {e}")
