import os, sys, json
import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import griddata

def compute_corrected_zone(mockup_dir):
    with open(os.path.join(mockup_dir, "mockup.json")) as f:
        pkg = json.load(f)
    cw, ch = pkg["canvas"]["width"], pkg["canvas"]["height"]
    warp = pkg["warp"]
    sw = warp["bounds"]["right"]; sh = warp["bounds"]["bottom"]

    mask = np.array(Image.open(os.path.join(mockup_dir, "shirt_full_mask.png")).convert("L")).astype(np.float32)/255
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None, "empty mask"
    x0m, x1m = xs.min(), xs.max()
    y0m, y1m = ys.min(), ys.max()
    mask_w = x1m - x0m
    mask_h = y1m - y0m
    mask_cx = (x0m + x1m) / 2

    # target chest rectangle in canvas space: centered horizontally on mask,
    # vertically starting ~18% down from mask top, spanning ~32% of mask height,
    # width ~38% of mask width -- typical centered chest print area
    tx0 = mask_cx - 0.19*mask_w
    tx1 = mask_cx + 0.19*mask_w
    ty0 = y0m + 0.18*mask_h
    ty1 = y0m + 0.50*mask_h

    # Build forward mesh (source -> canvas) same as MockupEngine.__init__, then
    # invert via griddata to sample canvas->source at our 4 target corners.
    tx = warp["transform"]
    canvas_corners = np.float32([[tx[0],tx[1]],[tx[2],tx[3]],[tx[4],tx[5]],[tx[6],tx[7]]])
    src_corners    = np.float32([[0,0],[sw,0],[sw,sh],[0,sh]])
    H_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_corners, canvas_corners))

    mx = np.array(warp["mesh_x"]).reshape(4,4); my = np.array(warp["mesh_y"]).reshape(4,4)
    def _bezier_basis(t):
        return np.stack([(1-t)**3, 3*t*(1-t)**2, 3*t**2*(1-t), t**3], axis=-1)
    _N = 65
    _u = np.linspace(0,1,_N); _v = np.linspace(0,1,_N)
    _Bu = _bezier_basis(_u); _Bv = _bezier_basis(_v)
    _Sx = _Bv @ mx @ _Bu.T; _Sy = _Bv @ my @ _Bu.T
    reg_x = np.tile(_u,_N)*sw; reg_y = np.repeat(_v,_N)*sh
    displaced = np.column_stack([_Sx.ravel(),_Sy.ravel()])

    corners = np.float32([[tx0,ty0],[tx1,ty0],[tx1,ty1],[tx0,ty1]])
    cp_h = np.column_stack([corners[:,0], corners[:,1], np.ones(4)])
    sp_h = (H_inv @ cp_h.T).T
    src_x = sp_h[:,0]/sp_h[:,2]; src_y = sp_h[:,1]/sp_h[:,2]

    rxv = griddata(displaced, reg_x, np.column_stack([src_x,src_y]), method="linear")
    ryv = griddata(displaced, reg_y, np.column_stack([src_x,src_y]), method="linear")

    if np.any(np.isnan(rxv)) or np.any(np.isnan(ryv)):
        return None, "nan in mapping"

    nx0, nx1 = float(min(rxv)), float(max(rxv))
    ny0, ny1 = float(min(ryv)), float(max(ryv))
    if nx0 < 0 or ny0 < 0 or nx1 > sw or ny1 > sh or nx1 <= nx0 or ny1 <= ny0:
        return None, f"result out of bounds: {nx0,ny0,nx1,ny1} vs sw,sh={sw},{sh}"

    return {"x0": int(round(nx0)), "y0": int(round(ny0)), "x1": int(round(nx1)), "y1": int(round(ny1))}, None

if __name__ == "__main__":
    targets = sys.argv[1:]
    for name in targets:
        pz, err = compute_corrected_zone(name)
        if err:
            print(f"FAIL {name}: {err}")
        else:
            print(f"OK {name}: {pz}")
