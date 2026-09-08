#!/usr/bin/env python3
"""Generate the client-side EDITOR preview assets that index.html's
loadMockupAssets() expects and that import_psd_mockups.py never produced:

  - shirt_base_preview.png    (base photo,  stretched to CW x CH = 920x1380)
  - shirt_mask_preview.png    (alpha mask,  stretched to CW x CH)
  - <overlay>_preview.png     (per overlay, stretched to CW x CH)
  - <displace_map>_preview.png (if the package has one, stretched to CW x CH)
  - warp_data.json            (a 256x384 dense lookup grid: for each fixed
    CWxCH canvas cell, the corresponding SOURCE (print-design) x,y — or -1
    where the cell falls outside the garment's valid warp region)

This mirrors the exact math MockupEngine.__init__ uses for the full-res
server-side render (Bezier warp mesh -> perspective inverse -> griddata
registration), just evaluated onto a fixed 256x384 grid tied to the
editor's fixed CW=920,CH=1380 canvas (920/256 == 1380/384 == 3.59375)
instead of onto the mockup's own native canvas resolution.

Usage: python3 tools/gen_editor_previews.py mockup68_package mockup69_package ...
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import griddata

CW, CH = 920, 1380
MAP_W, MAP_H = 256, 384  # = CW/3.59375, CH/3.59375 (matches existing packages)


def _bezier_basis(t):
    return np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3], axis=-1)


def compute_map_xy(pkg):
    """Returns (map_x, map_y) flat lists, row-major over MAP_H x MAP_W,
    in SOURCE (print-design) coordinate space, or -1 for invalid cells."""
    warp = pkg["warp"]
    cw, ch = pkg["canvas"]["width"], pkg["canvas"]["height"]
    src_w = warp["bounds"]["right"]
    src_h = warp["bounds"]["bottom"]
    tx = warp["transform"]
    canvas_corners = np.float32([[tx[0], tx[1]], [tx[2], tx[3]], [tx[4], tx[5]], [tx[6], tx[7]]])
    src_corners = np.float32([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])
    H_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_corners, canvas_corners))
    mx = np.array(warp["mesh_x"]).reshape(4, 4)
    my = np.array(warp["mesh_y"]).reshape(4, 4)

    N = 65
    u = np.linspace(0, 1, N)
    v = np.linspace(0, 1, N)
    Bu = _bezier_basis(u)
    Bv = _bezier_basis(v)
    Sx = Bv @ mx @ Bu.T
    Sy = Bv @ my @ Bu.T
    # Registration targets are NORMALIZED [0,1] fractions of src_w/src_h —
    # this matches the existing packages' warp_data.json convention, since
    # index.html's client compares warpedPixels values directly against
    # pos.cx/pos.cy/pos.size, which are themselves normalized fractions
    # (see loadMockupAssets: cx=(x0+x1)/2/newSrcW etc).
    reg_x = np.tile(u, N)
    reg_y = np.repeat(v, N)
    displaced = np.column_stack([Sx.ravel(), Sy.ravel()])

    # Sample directly at the target MAP_W x MAP_H grid (over the full canvas,
    # same as the editor's computeWarpLookup treats px/CW*mw, py/CH*mh).
    ys_r, xs_r = np.mgrid[0:MAP_H, 0:MAP_W]
    px_r = (xs_r.ravel() + 0.5) / MAP_W * cw
    py_r = (ys_r.ravel() + 0.5) / MAP_H * ch
    cp_h = np.column_stack([px_r, py_r, np.ones_like(px_r)])
    sp_h = (H_inv @ cp_h.T).T
    src_x = sp_h[:, 0] / sp_h[:, 2]
    src_y = sp_h[:, 1] / sp_h[:, 2]
    in_range = (src_x >= -300) & (src_x <= src_w + 300) & (src_y >= -300) & (src_y <= src_h + 300)

    rx = np.full(MAP_W * MAP_H, -1.0)
    ry = np.full(MAP_W * MAP_H, -1.0)
    idx = np.where(in_range)[0]
    sp = np.column_stack([src_x[in_range], src_y[in_range]])
    rxv = griddata(displaced, reg_x, sp, method="linear")
    ryv = griddata(displaced, reg_y, sp, method="linear")
    valid = ~(np.isnan(rxv) | np.isnan(ryv))
    rx[idx[valid]] = rxv[valid]
    ry[idx[valid]] = ryv[valid]
    return rx.tolist(), ry.tolist(), float(src_w), float(src_h)


def preview_name(fname):
    base, ext = os.path.splitext(fname)
    return f"{base}_preview{ext}"


def make_preview(src_path, dst_path):
    if os.path.exists(dst_path):
        return
    img = Image.open(src_path)
    img.resize((CW, CH), Image.LANCZOS).save(dst_path, optimize=True)


def process(pkg_dir):
    name = os.path.basename(pkg_dir.rstrip("/"))
    mockup_json = os.path.join(pkg_dir, "mockup.json")
    warp_data_path = os.path.join(pkg_dir, "warp_data.json")
    with open(mockup_json) as f:
        pkg = json.load(f)

    if not os.path.exists(warp_data_path):
        map_x, map_y, src_w, src_h = compute_map_xy(pkg)
        warp_data = {
            "map_w": MAP_W,
            "map_h": MAP_H,
            "canvas_w": pkg["canvas"]["width"],
            "canvas_h": pkg["canvas"]["height"],
            "src_w": src_w,
            "src_h": src_h,
            "map_x": map_x,
            "map_y": map_y,
        }
        with open(warp_data_path, "w") as f:
            json.dump(warp_data, f)

    make_preview(os.path.join(pkg_dir, "shirt_base.png"), os.path.join(pkg_dir, "shirt_base_preview.png"))
    make_preview(os.path.join(pkg_dir, "shirt_full_mask.png"), os.path.join(pkg_dir, "shirt_mask_preview.png"))
    for ov in pkg.get("overlays", []):
        fname = ov["file"]
        make_preview(os.path.join(pkg_dir, fname), os.path.join(pkg_dir, preview_name(fname)))
    disp = pkg.get("displace")
    if disp:
        fname = disp["map"]
        make_preview(os.path.join(pkg_dir, fname), os.path.join(pkg_dir, preview_name(fname)))

    print(f"OK {name}")


if __name__ == "__main__":
    for pd in sys.argv[1:]:
        try:
            process(pd)
        except Exception as e:
            print(f"FAIL {pd}: {e}")
