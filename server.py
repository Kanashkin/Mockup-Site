"""
Mockup API Server
"""

import io
import os
import json
import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import griddata
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCKUP_DIR = os.path.join(os.path.dirname(__file__), "mockup_package")

BLEND_FNS = {
    "BlendMode.SOFT_LIGHT": lambda b, bl: np.clip(np.where(
        bl <= 0.5,
        b - (1 - 2*bl)*b*(1 - b),
        b + (2*bl - 1)*(np.where(b <= 0.25, ((16*b-12)*b+4)*b,
                                  np.sqrt(np.maximum(b, 0))) - b)), 0, 1),
    "BlendMode.MULTIPLY": lambda b, bl: b * bl,
    "BlendMode.SCREEN":   lambda b, bl: 1 - (1-b)*(1-bl),
    "BlendMode.OVERLAY":  lambda b, bl: np.where(b<=0.5, 2*b*bl, 1-2*(1-b)*(1-bl)),
}


class MockupEngine:
    def __init__(self, mockup_dir: str):
        print("Loading mockup package...")
        with open(os.path.join(mockup_dir, "mockup.json")) as f:
            pkg = json.load(f)

        self.canvas_w = pkg["canvas"]["width"]
        self.canvas_h = pkg["canvas"]["height"]
        warp = pkg["warp"]

        # Используем ВСЮ зону варпа — полная футболка
        src_w = warp["bounds"]["right"]
        src_h = warp["bounds"]["bottom"]
        self.print_zone = {"x0": 0, "y0": 0, "x1": int(src_w), "y1": int(src_h)}

        self.shirt_arr = (np.array(Image.open(os.path.join(mockup_dir, "shirt_base.png"))
                          .convert("RGB")).astype(np.float32) / 255)
        self.mask_arr  = (np.array(Image.open(os.path.join(mockup_dir, "tshirt_mask.png"))
                          .convert("L")).astype(np.float32) / 255)

        self.overlays = []
        for ov in pkg["overlays"]:
            img = (np.array(Image.open(os.path.join(mockup_dir, ov["file"]))
                   .convert("RGB")).astype(np.float32) / 255)
            self.overlays.append((img, ov["opacity"], ov["blend_mode"]))

        self.src_w = src_w
        self.src_h = src_h
        tx = warp["transform"]
        canvas_corners = np.float32([[tx[0],tx[1]], [tx[2],tx[3]], [tx[4],tx[5]], [tx[6],tx[7]]])
        src_corners    = np.float32([[0,0],[src_w,0],[src_w,src_h],[0,src_h]])
        H_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_corners, canvas_corners))

        u_norm = np.array([0, 1/3, 2/3, 1])
        v_norm = np.array([0, 1/3, 2/3, 1])
        reg_x  = np.tile(u_norm, 4) * src_w
        reg_y  = np.repeat(v_norm, 4) * src_h
        mx = np.array(warp["mesh_x"]).reshape(4,4)
        my = np.array(warp["mesh_y"]).reshape(4,4)
        displaced = np.column_stack([mx.ravel(), my.ravel()])

        print("Precomputing warp map...")
        cw, ch = self.canvas_w, self.canvas_h
        ys, xs = np.mgrid[0:ch, 0:cw]
        cp  = np.column_stack([xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)])
        cp_h = np.concatenate([cp, np.ones((len(cp),1))], axis=1)
        sp_h = (H_inv @ cp_h.T).T
        src_x = sp_h[:,0] / sp_h[:,2]
        src_y = sp_h[:,1] / sp_h[:,2]
        in_r  = ((src_x >= -300) & (src_x <= src_w+300) &
                 (src_y >= -300) & (src_y <= src_h+300))
        ridx  = np.where(in_r)[0]
        rsp   = np.column_stack([src_x[in_r], src_y[in_r]])
        self._rx   = griddata(displaced, reg_x, rsp, method="cubic")
        self._ry   = griddata(displaced, reg_y, rsp, method="cubic")
        self._ridx = ridx
        print("Ready.")

    def render(self, design_img: Image.Image) -> Image.Image:
        pz = self.print_zone
        pw = pz["x1"] - pz["x0"]
        ph = pz["y1"] - pz["y0"]

        src_canvas = Image.new("RGBA", (int(self.src_w), int(self.src_h)), (0,0,0,0))
        src_canvas.paste(design_img.resize((pw, ph), Image.LANCZOS), (pz["x0"], pz["y0"]))
        dw, dh = src_canvas.size
        design_arr = np.array(src_canvas).astype(np.float32)

        nx = self._rx / self.src_w * dw
        ny = self._ry / self.src_h * dh
        valid = ((~np.isnan(nx)) & (~np.isnan(ny)) &
                 (nx >= 0) & (nx < dw-1) & (ny >= 0) & (ny < dh-1))

        warped = np.zeros((self.canvas_h, self.canvas_w, 4), dtype=np.float32)
        vi = np.where(valid)[0]
        fi = self._ridx[vi]
        warped[fi//self.canvas_w, fi%self.canvas_w] = design_arr[
            ny[valid].astype(int), nx[valid].astype(int)]

        warped[:,:,3] = warped[:,:,3] * self.mask_arr

        alpha      = warped[:,:,3:4] / 255
        design_rgb = warped[:,:,:3]  / 255
        result     = self.shirt_arr * (1-alpha) + self.shirt_arr * design_rgb * alpha

        for img, opacity, blend_mode in self.overlays:
            fn = BLEND_FNS.get(blend_mode)
            if fn:
                result = result*(1-opacity) + fn(result, img)*opacity

        return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))


engine = MockupEngine(MOCKUP_DIR)


@app.post("/render")
async def render_mockup(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20MB)")

    try:
        design = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        raise HTTPException(400, "Could not read image file")

    result = engine.render(design)

    buf = io.BytesIO()
    result.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=mockup.png"}
    )


@app.get("/health")
def health():
    return {"status": "ok", "canvas": f"{engine.canvas_w}x{engine.canvas_h}"}


app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")
