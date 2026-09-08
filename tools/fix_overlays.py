#!/usr/bin/env python3
"""Regenerate overlay_Light.png / overlay_Shadow.png (+ their _preview
downscales) for mockup68-141 directly from the original PSDs.

import_psd_mockups.py originally extracted these "shading" layers with
l.composite(viewport=psd.bbox). Every one of these layers has clipping=True
(clipped to the "PLACE YOUR LOGO" smart object below it in the PSD stack),
and psd-tools' composite() evaluates that clip in isolation from the rest
of the layer stack -- with no clip base in scope for a standalone call, it
silently renders fully transparent. That shipped mockup68-141's overlays as
blank alpha=0 images (verified: alpha std 0.0 across all 74), so recoloring
a shirt or applying a design rendered with zero fabric shading -- a flat
color fill instead of a real garment.

The fix (also applied going forward in import_psd_mockups.py): pull the
layer's own raw pixels via topil() (unaffected by clipping) and paste them
onto a full transparent canvas at the layer's own bbox offset. The render
engine re-clips these overlays itself via shirt_full_mask, so PSD-level
clipping semantics were never needed here in the first place.

Usage (run where the original 07.09 PSD packs and the site's git checkout
are both reachable -- these PSDs are ~100-130MB each, so this runs beside
the source, not against files staged into a container):
    python3 tools/fix_overlays.py 68 69 70 ...        # specific mockup numbers
    python3 tools/fix_overlays.py $(seq 68 141)       # the whole 07.09 batch

Expects:
  SRC_ROOT/<set 1-6>/<n>.psd       -- original PSD packs
  OUT_ROOT/mockupN_package/        -- the site's checked-out packages
"""
import os
import sys
import time

import numpy as np
from PIL import Image
from psd_tools import PSDImage

SRC_ROOT = os.environ.get("PZ_SRC_ROOT", os.path.expanduser("~/mnt/07.09"))
OUT_ROOT = os.environ.get("PZ_OUT_ROOT", os.path.expanduser("~/mnt/Mockup-Site-upload"))
CW, CH = 920, 1380


def build_mapping(start_num=68):
    """mockupN_package -> (set, src_num), mirroring import_psd_mockups.py's
    own sequential numbering: sets "1".."6" in order, each set's PSDs
    numerically sorted."""
    mapping = {}
    num = start_num
    for set_id in ["1", "2", "3", "4", "5", "6"]:
        set_dir = os.path.join(SRC_ROOT, set_id)
        nums = sorted(int(fn[:-4]) for fn in os.listdir(set_dir) if fn.lower().endswith(".psd"))
        for n in nums:
            mapping[num] = (set_id, n)
            num += 1
    return mapping


def extract_layer_full(layer, canvas_w, canvas_h):
    """Raw (uncomposited/unclipped) layer render pasted onto full transparent canvas."""
    bbox = layer.bbox
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return canvas
    img = layer.topil().convert("RGBA")
    canvas.paste(img, (bbox[0], bbox[1]))
    return canvas


def fix_one(mockup_num, set_id, src_num, log):
    out_dir = os.path.join(OUT_ROOT, f"mockup{mockup_num}_package")
    psd_path = os.path.join(SRC_ROOT, set_id, f"{src_num}.psd")
    if not os.path.isdir(out_dir):
        log(f"SKIP {mockup_num}: no out dir")
        return
    if not os.path.exists(psd_path):
        log(f"SKIP {mockup_num}: no psd {psd_path}")
        return
    t0 = time.time()
    psd = PSDImage.open(psd_path)
    cw, ch = psd.size
    by_name = {l.name: l for l in psd}
    for lname, fname in [("Light", "overlay_Light.png"), ("Shadow", "overlay_Shadow.png")]:
        layer = by_name.get(lname)
        if layer is None:
            log(f"  {mockup_num}: no layer {lname}")
            continue
        img = extract_layer_full(layer, cw, ch)
        arr = np.array(img)
        if arr[:, :, 3].std() < 1.0:
            log(f"  {mockup_num}/{lname}: STILL BLANK after fix")
        img.save(os.path.join(out_dir, fname))
        prev_path = os.path.join(out_dir, fname.replace(".png", "_preview.png"))
        img.resize((CW, CH), Image.LANCZOS).save(prev_path, optimize=True)
    del psd
    log(f"OK {mockup_num} (set{set_id}/{src_num}.psd) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    mapping = build_mapping()
    targets = [int(x) for x in sys.argv[1:]]

    def log(msg):
        print(msg, flush=True)

    for n in targets:
        if n not in mapping:
            log(f"SKIP {n}: not in mapping")
            continue
        set_id, src_num = mapping[n]
        try:
            fix_one(n, set_id, src_num, log)
        except Exception as e:
            log(f"FAIL {n}: {e}")
