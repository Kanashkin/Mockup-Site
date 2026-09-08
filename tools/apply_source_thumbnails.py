#!/usr/bin/env python3
"""Replace shirt_preview.png / shirt_preview_color.png for mockup68-141 with
the actual demo photos the PSD packs shipped with (jpg/обычный = default,
jpg/цвет = hover/color variant), instead of a synthesized placeholder logo.

Mapping mockupN_package -> (set, src_num) is deterministic, mirroring
import_psd_mockups.py's own numbering: sets "1".."6" in order, each set's
PSDs numerically sorted, mockup numbers assigned sequentially starting at
start_num (68).

Usage: python3 tools/apply_source_thumbnails.py <src_root_07.09> <out_root_repo> [start_num]
"""
import os
import sys

from PIL import Image

THUMB_W = 920


def build_mapping(src_root, start_num=68):
    mapping = {}
    num = start_num
    for set_id in ["1", "2", "3", "4", "5", "6"]:
        set_dir = os.path.join(src_root, set_id)
        jpg_dir = os.path.join(set_dir, "jpg", "обычный")
        if not os.path.isdir(jpg_dir):
            print(f"WARN: missing {jpg_dir}", file=sys.stderr)
            continue
        nums = sorted(int(fn[:-4]) for fn in os.listdir(jpg_dir) if fn.lower().endswith(".jpg"))
        for n in nums:
            mapping[f"mockup{num}_package"] = (set_id, n)
            num += 1
    return mapping


def make_preview(src_jpg, dst_png):
    img = Image.open(src_jpg).convert("RGB")
    w, h = img.size
    th = int(h * THUMB_W / w)
    img.resize((THUMB_W, th), Image.LANCZOS).save(dst_png, optimize=True)


if __name__ == "__main__":
    src_root, out_root = sys.argv[1], sys.argv[2]
    start_num = int(sys.argv[3]) if len(sys.argv) > 3 else 68
    mapping = build_mapping(src_root, start_num)
    print(f"Mapped {len(mapping)} packages", file=sys.stderr)
    for pkg_name, (set_id, n) in mapping.items():
        out_dir = os.path.join(out_root, pkg_name)
        if not os.path.isdir(out_dir):
            print(f"SKIP {pkg_name}: no such package dir")
            continue
        normal_jpg = os.path.join(src_root, set_id, "jpg", "обычный", f"{n}.jpg")
        color_jpg = os.path.join(src_root, set_id, "jpg", "цвет", f"{n}.jpg")
        try:
            make_preview(normal_jpg, os.path.join(out_dir, "shirt_preview.png"))
            make_preview(color_jpg, os.path.join(out_dir, "shirt_preview_color.png"))
            print(f"OK {pkg_name} <- set{set_id}/{n}.jpg")
        except Exception as e:
            print(f"FAIL {pkg_name} (set{set_id}/{n}.jpg): {e}")
