#!/usr/bin/env python3
"""Import a single numbered set directory (as used by import_psd_mockups.py's
per-batch folder layout: <set_dir>/<n>.psd, <set_dir>/Displacement Maps/<n>.psd)
with an explicit starting package number, so a batch can be staged and
converted set-by-set (disk-limited) instead of needing the whole 6-set batch
present on disk at once. Package numbering matches what
import_psd_mockups.py would assign if run over the whole batch in one pass:
consecutive numbers starting at start_num, in ascending source-filename order
within this set.

Usage: import_one_set.py <set_dir> <out_root> <start_num> [set_label]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from import_psd_mockups import convert_one

if __name__ == "__main__":
    set_dir, out_root, start_num = sys.argv[1], sys.argv[2], int(sys.argv[3])
    set_label = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(set_dir.rstrip("/"))
    nums = sorted(
        int(fn[:-4]) for fn in os.listdir(set_dir) if fn.lower().endswith(".psd")
    )
    manifest_path = os.path.join(out_root, "_import_manifest.json")
    existing = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    by_pkg = {m["package"]: m for m in existing}

    num = start_num
    for n in nums:
        psd_path = os.path.join(set_dir, f"{n}.psd")
        disp_path = os.path.join(set_dir, "Displacement Maps", f"{n}.psd")
        pkg_name = f"mockup{num}_package"
        out_dir = os.path.join(out_root, pkg_name)
        done_marker = os.path.join(out_dir, "mockup.json")
        if os.path.exists(done_marker):
            num += 1
            continue
        try:
            info = convert_one(psd_path, disp_path if os.path.exists(disp_path) else None, out_dir)
            info.update({"package": pkg_name, "set": set_label, "src_num": n, "ok": True})
        except Exception as e:
            info = {"package": pkg_name, "set": set_label, "src_num": n, "ok": False, "error": str(e)}
        print(json.dumps(info), flush=True)
        by_pkg[pkg_name] = info
        num += 1
        # Write after every file, not just at the end — a long set can hit
        # the shell timeout mid-loop, and a manifest only flushed at the end
        # would silently lose every already-converted package's record.
        with open(manifest_path, "w") as f:
            json.dump(list(by_pkg.values()), f, indent=2)

    print(f"NEXT_START_NUM={num}", flush=True)
