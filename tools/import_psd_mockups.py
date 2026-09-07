#!/usr/bin/env python3
"""
Convert the "07.09" raw PSD mockup template packs into the
Mockup-Site mockupN_package/ format (mockup.json + shirt_base.png +
shirt_full_mask.png + overlay_*.png [+ displace map]).

Each source PSD has a consistent layer stack:
  Background            (pixel, base backdrop)
  Don't Touch           (pixel, optional — skin/hair that must show through)
  T-Shirt               (pixel, the garment silhouette/cutout)
  Change T-Shirt Color  (solidcolorfill, clipped — handled by the render
                         engine itself via shirt_full_mask, not exported)
  PLACE YOUR LOGO       (smartobject, clipped — the design placeholder;
                         its Photoshop "Custom Envelope Warp" data maps
                         1:1 onto mockup.json's warp.transform/mesh_x/mesh_y)
  Light, Shadow, ...    (pixel, clipped — shading overlays, exported as-is;
                         the render engine re-clips them with shirt_full_mask
                         so exact PSD clipping semantics don't need to be
                         reproduced here)

A companion "Displacement Maps/<n>.psd" (same number, higher native
resolution) supplies the fabric-fold Displace-filter source map; it is
flattened, resized to the target canvas size, and used only as
mockup.json's displace.map (not blended in as a visible overlay).
"""
import gc
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Tag


def get_warp_data(smart_layer):
    tb = smart_layer._record.tagged_blocks
    pl2 = tb.get_data(Tag.PLACED_LAYER2)
    transform = [float(v) for v in pl2.transform]  # 8 floats, canvas space
    w = pl2.warp
    bounds = w[b"bounds"]
    src_w = float(bounds[b"Rght"])
    src_h = float(bounds[b"Btom"])
    style = w[b"warpStyle"].enum
    if style != b"warpCustom":
        raise ValueError(f"unexpected warpStyle {style!r} (expected warpCustom)")
    mesh = w[b"customEnvelopeWarp"][b"meshPoints"]
    mesh_x = [float(v) for v in mesh[b"Hrzn"]]
    mesh_y = [float(v) for v in mesh[b"Vrtc"]]
    if len(mesh_x) != 16 or len(mesh_y) != 16:
        raise ValueError(f"expected 16-point mesh, got {len(mesh_x)}/{len(mesh_y)}")
    return transform, src_w, src_h, mesh_x, mesh_y


def get_print_zone(smart_layer):
    """The smart object's own embedded content's layer bounding boxes give
    the design placeholder's natural rectangle — used as the default print
    zone in the smart object's own (0..src_w, 0..src_h) space. Uses layer
    metadata (bbox) rather than compositing pixels: the embedded content can
    be very high-resolution (independent of the outer canvas), and a full
    pixel composite of it was OOM-killing the process on some templates."""
    so = smart_layer.smart_object
    inner = PSDImage.open(io.BytesIO(so.data))
    x0 = y0 = None
    x1 = y1 = None
    for l in inner.descendants():
        if l.is_group() or not l.visible:
            continue
        bbox = l.bbox
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        x0 = bbox[0] if x0 is None else min(x0, bbox[0])
        y0 = bbox[1] if y0 is None else min(y0, bbox[1])
        x1 = bbox[2] if x1 is None else max(x1, bbox[2])
        y1 = bbox[3] if y1 is None else max(y1, bbox[3])
    if x0 is None:
        return {"x0": 0, "y0": 0, "x1": inner.width, "y1": inner.height}
    return {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}


def safe_overlay_filename(name, idx):
    base = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    if not base:
        base = f"layer{idx}"
    return f"overlay_{base}.png"


def convert_one(psd_path, disp_path, out_dir):
    psd = PSDImage.open(psd_path)
    canvas_w, canvas_h = psd.size

    layers = list(psd)
    by_name = {l.name: l for l in layers}
    tshirt = by_name.get("T-Shirt")
    bg = by_name.get("Background")
    dont_touch = by_name.get("Don't Touch")
    smart = None
    shading = []
    for l in layers:
        if l.kind == "smartobject":
            smart = l
        elif l.name in ("Background", "Don't Touch", "T-Shirt"):
            continue
        elif l.kind == "solidcolorfill":
            continue  # "Change T-Shirt Color" — engine fills this itself
        elif l.kind == "pixel":
            shading.append(l)

    if tshirt is None or smart is None:
        raise ValueError(f"{psd_path}: missing T-Shirt or smart-object layer")

    os.makedirs(out_dir, exist_ok=True)

    # shirt_base = Background (+ Don't Touch) + T-Shirt, full canvas.
    base_set = {id(x) for x in (bg, dont_touch, tshirt) if x is not None}
    base_img = psd.composite(layer_filter=lambda l: id(l) in base_set)
    base_img.convert("RGB").save(os.path.join(out_dir, "shirt_base.png"))

    # shirt_full_mask = T-Shirt layer's own alpha, full canvas.
    tshirt_img = tshirt.composite(viewport=psd.bbox)
    tshirt_img.convert("RGBA").split()[-1].save(
        os.path.join(out_dir, "shirt_full_mask.png"))

    overlays_meta = []
    for i, l in enumerate(shading):
        fname = safe_overlay_filename(l.name, i)
        img = l.composite(viewport=psd.bbox)
        img.convert("RGBA").save(os.path.join(out_dir, fname))
        overlays_meta.append({
            "name": l.name,
            "file": fname,
            "opacity": round(l.opacity / 255.0, 6),
            "blend_mode": str(l.blend_mode),
        })

    transform, src_w, src_h, mesh_x, mesh_y = get_warp_data(smart)
    print_zone = get_print_zone(smart)

    displace_meta = None
    if disp_path and os.path.exists(disp_path):
        dpsd = PSDImage.open(disp_path)
        dcomp = dpsd.composite()
        if dcomp is not None:
            dcomp = dcomp.convert("L").resize((canvas_w, canvas_h), Image.LANCZOS)
            dcomp.save(os.path.join(out_dir, "displace_map.png"))
            displace_meta = {"map": "displace_map.png", "h_scale": 10, "v_scale": 10}

    mockup_json = {
        "canvas": {"width": canvas_w, "height": canvas_h},
        "warp": {
            "bounds": {"left": 0.0, "top": 0.0, "right": src_w, "bottom": src_h},
            "transform": transform,
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
        },
        "print_zone": print_zone,
        "overlays": overlays_meta,
    }
    if displace_meta:
        mockup_json["displace"] = displace_meta

    with open(os.path.join(out_dir, "mockup.json"), "w") as f:
        json.dump(mockup_json, f, indent=2)

    del psd, base_img, tshirt_img
    gc.collect()

    return {
        "canvas": [canvas_w, canvas_h],
        "n_overlays": len(overlays_meta),
        "has_displace": displace_meta is not None,
        "print_zone": print_zone,
        "src": [src_w, src_h],
    }


if __name__ == "__main__":
    src_root, out_root, start_num = sys.argv[1], sys.argv[2], int(sys.argv[3])
    manifest = []
    num = start_num
    for set_id in ["1", "2", "3", "4", "5", "6"]:
        set_dir = os.path.join(src_root, set_id)
        if not os.path.isdir(set_dir):
            print(f"WARN: missing set dir {set_dir}", file=sys.stderr)
            continue
        nums = sorted(
            (int(fn[:-4]) for fn in os.listdir(set_dir) if fn.lower().endswith(".psd")),
        )
        for n in nums:
            psd_path = os.path.join(set_dir, f"{n}.psd")
            disp_path = os.path.join(set_dir, "Displacement Maps", f"{n}.psd")
            pkg_name = f"mockup{num}_package"
            out_dir = os.path.join(out_root, pkg_name)
            done_marker = os.path.join(out_dir, "mockup.json")
            if os.path.exists(done_marker):
                num += 1
                continue  # resume support: skip packages already converted
            try:
                info = convert_one(psd_path, disp_path, out_dir)
                info.update({"package": pkg_name, "set": set_id, "src_num": n, "ok": True})
            except Exception as e:
                info = {"package": pkg_name, "set": set_id, "src_num": n, "ok": False, "error": str(e)}
            print(json.dumps(info), flush=True)
            manifest.append(info)
            num += 1
    manifest_path = os.path.join(out_root, "_import_manifest.json")
    existing = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    by_pkg = {m["package"]: m for m in existing}
    for m in manifest:
        by_pkg[m["package"]] = m
    with open(manifest_path, "w") as f:
        json.dump(list(by_pkg.values()), f, indent=2)
