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

BUG 3 (2026-09-15, "а че ты их центруешь только при наведении курсора?"): BUG
2's "topmost, not largest" heuristic had no sanity check on a candidate's own
size, so an implausibly huge false-positive box could still beat the real face
in the topmost tie-break if its y happened to be equal or smaller. Concretely,
on `mockup1007_package` (kids2 group's own representative gallery card) the
default photo (`shirt_preview.png`) detected a real 171x171 face and a bogus
444x444 box tied at y=62 — the tie-break happened to keep the real face — but
the hover-color photo (`shirt_preview_color.png`, same shoot/pose, only the
shirt recolored) detected the same real face at y=62 plus a DIFFERENT bogus
box, 479x479 at y=48 — one pixel higher, so "topmost" picked the bogus box
there instead. Since both source photos are the same shot, the crop should
have come out identical for the default and hover thumbnails; instead the
default view centered on the real face while the hover view centered on the
bogus box (~2% of frame width off), which is exactly what read as "only
centers on hover" — the two states genuinely used different crop centers.
FIX: reject any candidate wider than 35% of the image width before picking
topmost (`MAX_FACE_FRAC`) — checked across all 110 face detections in the
11.09 batch, the largest legitimate face tops out at 28% of image width, then
there's a clean jump straight to this one 52%-wide false positive, so 35% is
a safe cutoff that drops only implausible boxes. Falls back to no-face-found
(the existing geometric-center fallback) if every candidate gets filtered out.

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


# BUG 3 (2026-09-15, "а че ты их центруешь только при наведении курсора?"): the
# "topmost, not largest" heuristic above (BUG 2) still had no sanity check on a
# candidate's own size, so it could pick an implausibly huge "face" box if that
# box's y happened to tie/beat the real face's y. Concretely, on
# mockup1007_package (the kids2 group's own representative card) the base photo
# (shirt_preview.png) detected two candidates tied at y=62 — a real 171x171 face
# and a bogus 444x444 box — and min()'s left-to-right tie-break happened to keep
# the real face; the hover-color photo (shirt_preview_color.png, same shoot,
# same pose, only the shirt recolored) detected a real 171x171 face at y=62 AND
# a bogus 479x479 box at y=48 — one pixel higher, so "topmost" picked the bogus
# box instead. Both source images are the same photo, so the correct crop
# should have been identical for the default and hover thumbnails; instead the
# default (base) view came out centered on the real face while the hover view
# came out centered on the bogus box (~16px/920px, ~2% of frame, shifted) —
# which is exactly what looked like "only centers on hover" from the user's
# side, since the two states genuinely used different crop centers.
# FIX: reject any candidate wider than `MAX_FACE_FRAC` of the image width
# before picking "topmost" — a real face in these compositions never gets
# anywhere near that large (checked across all 110 detections in the 11.09
# batch: the next-largest legitimate face tops out at 28% of image width, then
# there's a clean jump straight to this one 52%-wide false positive), so 35% is
# a safe cutoff that keeps every real detection and drops only implausible
# ones. If every candidate gets filtered out, fall back to no-face-found
# (the caller's existing geometric-center fallback), same as an empty
# `detectMultiScale` result.
MAX_FACE_FRAC = 0.35

# BUG 4 (2026-09-16, "ребенка ты так и не отцентровал а самый левый чел ваще
# исчез"): minNeighbors=6 (used since BUG 2) still let through single, isolated
# false-positive detections that aren't oversized (so BUG 3's MAX_FACE_FRAC
# doesn't catch them) but ARE topmost, beating the real face on y alone. Two
# confirmed instances: `mockup105_package` (a building facade's window/cornice
# detail detected as a 91x91 "face" at y=11, one pixel from the very top of the
# frame, versus the real 95x95 face at y=106 — the man's face was cropped out
# of the thumbnail entirely, only his arm remained) and `mockup1000_package`
# (three overlapping detections for the same real face, but the topmost of the
# three was an outlier ~87px off from where the other two agreed the face
# actually was, shifting the crop enough to visibly push the girl off-center).
# FIX: raise minNeighbors from 6 to 8 — this is stricter about how many
# overlapping raw detections must agree before a box is reported at all, and
# empirically drops exactly these two false/outlier detections while leaving
# the real face as the sole (or correctly topmost) survivor. Checked against
# every package in the 11.09 batch to look for the opposite failure (a real
# face that mn=6 found correctly but mn=8 loses or replaces with something
# worse): found one, `mockup1064_package` (a selfie with the phone raised —
# the actual face is weak/marginal at mn=8, a hand+phone-shaped blob is a more
# stable false positive there instead) — for that one case specifically,
# falling back to the plain geometric-center crop (i.e. as if no face were
# found at all) turned out to already look fine, since the subject there
# happens to be reasonably centered in the original shot anyway. No general
# minSize/max-size tweak separated "real but weak" from "false but stable" in
# that one case without breaking others tested alongside it, so rather than
# keep tuning a single global threshold indefinitely, `_MANUAL_NO_FACE`
# below lists this one file explicitly to skip face detection and use the
# geometric-center fallback directly. Add to this list (never invent a new
# global parameter change) if another isolated case like this turns up.
_MIN_NEIGHBORS = 8
_MANUAL_NO_FACE = {
    "mockup1064_package/shirt_preview.png",
    "mockup1064_package/shirt_preview_color.png",
    # BUG 5 (2026-09-17, user: "у тебя блять страница генератор которая
    # показывает что ты обрезал модель и вставил ее не по центру") — reported
    # against mockup1123_package's homepage gallery card (the "8 photos" woman13
    # group tile): a woman on a yacht dock, back turned, head down, both hands
    # raised to her hair — no frontal face is actually visible in the shot, but
    # the cascade still fired on a single, small, confident-looking false
    # positive (a 115x115 box on the boat hull/reflection at the bottom-LEFT of
    # frame, y=403). Being the only detection, "topmost" trivially picked it,
    # centering the crop at x=119.5 of 920 — clamped to x0=0 — which shows
    # almost nothing but boat and water and excludes the woman (who sits at
    # roughly x=290-580) almost entirely. Same false-positive shape in both the
    # base and color-hover variants (boxes at (62,403) and (61,404) respectively
    # — the same background detail, not the model), so both needed the override,
    # matching BUG 3's precedent that a mismatched pair looks like "only wrong on
    # hover"/inconsistent even when it's really wrong on both.
    # Prompted an audit of every OTHER landscape (920x613, the "17.09 batch")
    # package for the same symptom — any face-detected crop clamped to the x0=0
    # or x0=(w-new_w) edge, which is what a real subject positioned at the
    # extreme edge OR a bogus edge-of-frame detection both produce, so each hit
    # needs a visual check to tell those apart. Found 7 more, all genuinely bogus
    # (a wall/skateboard/gym-rig/light-bulb-string/steps false positive, real
    # subject actually roughly centered) — visually confirmed the plain
    # geometric-center fallback looks correct for all 8 before adding them here:
    "mockup1123_package/shirt_preview.png",
    "mockup1123_package/shirt_preview_color.png",
    "mockup1016_package/shirt_preview.png",
    "mockup1016_package/shirt_preview_color.png",
    "mockup1017_package/shirt_preview.png",
    "mockup1017_package/shirt_preview_color.png",
    "mockup1023_package/shirt_preview.png",
    "mockup1023_package/shirt_preview_color.png",
    "mockup1035_package/shirt_preview.png",
    "mockup1035_package/shirt_preview_color.png",
    "mockup1042_package/shirt_preview.png",
    "mockup1042_package/shirt_preview_color.png",
    "mockup1044_package/shirt_preview.png",
    "mockup1044_package/shirt_preview_color.png",
    "mockup1077_package/shirt_preview.png",
    "mockup1077_package/shirt_preview_color.png",
}

# Found in the same 2026-09-16 audit (not user-reported): `mockup1036_package`'s
# hover-color photo shows a man looking off to the side (near-profile), which
# `haarcascade_frontalface_default` doesn't reliably detect at all — it fell
# back to geometric-center, which crops to the dumbbell rack on the left and
# misses him entirely (confirmed: `haarcascade_profileface` DOES find him
# correctly, at x-center ~519 of 920, stable across every minNeighbors tried).
# The base (white-shirt) photo of the same pose already gets a correct
# frontal-cascade detection on its own, so only the recolored hover variant
# needed this. Rather than wire a second cascade into the main path for one
# instance, this is a manual override like `_MANUAL_NO_FACE` above.
_MANUAL_CENTER = {
    "mockup1036_package/shirt_preview_color.png": 519,
    # Found in the 2026-09-17 full-audit of the 17.09 import batch (74 new
    # packages, all landscape 920x613 vendor reference photos — same
    # face-centering exposure as every prior landscape batch). In each case
    # below, `haarcascade_frontalface_default` fires on a busy/textured
    # background (a mottled wall, a window-side pedestrian, a wood-grain
    # pillar) at a smaller y than the real face, so "topmost" — otherwise the
    # right call, see BUG 2/3 above — picks the false positive instead. No
    # single global parameter separated these from the many correctly-handled
    # landscape photos in the same batch, so per BUG 4's precedent these are
    # explicit overrides rather than another global tune. Same crop for both
    # variants in each pair since both are the same photo/pose, just a
    # different shirt color.
    "mockup1083_package/shirt_preview.png": 620,
    "mockup1083_package/shirt_preview_color.png": 620,
    "mockup1106_package/shirt_preview.png": 455,
    "mockup1106_package/shirt_preview_color.png": 455,
    "mockup1115_package/shirt_preview.png": 531,
    "mockup1115_package/shirt_preview_color.png": 531,
    "mockup1142_package/shirt_preview.png": 440,
    "mockup1142_package/shirt_preview_color.png": 440,
    "mockup1143_package/shirt_preview.png": 485,
    "mockup1143_package/shirt_preview_color.png": 485,
}


def _detect_face_center_x(src_path):
    """Return the x-center of the TOPMOST plausible detected face, or None if
    no face is found. Topmost (smallest y), not largest, since a standing
    subject's head is reliably the highest point of the body, while the
    largest box among several candidates is sometimes a false positive on a
    busy background that happens to score bigger than the real, smaller face
    (see BUG 2). Candidates wider than MAX_FACE_FRAC of the image are dropped
    before picking topmost, since an implausibly large box can otherwise still
    win the topmost tie-break over the real, smaller face (see BUG 3)."""
    norm = src_path.replace(os.sep, "/")
    for f, cx in _MANUAL_CENTER.items():
        if norm == f or norm.endswith("/" + f):
            return cx
    if any(norm == f or norm.endswith("/" + f) for f in _MANUAL_NO_FACE):
        return None
    cv_img = cv2.imread(src_path)
    if cv_img is None:
        return None
    h, w = cv_img.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=_MIN_NEIGHBORS, minSize=(50, 50)
    )
    faces = [f for f in faces if f[2] <= MAX_FACE_FRAC * w]
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
