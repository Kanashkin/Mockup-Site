import io, os, json, datetime
import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import Response, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session

import paypal
import google_oauth
from db import init_db, get_db, User, Subscription
from auth import (
    hash_password, verify_password, validate_email, validate_password,
    create_user_session, clear_user_session, get_current_user, require_user,
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# SECRET_KEY signs the session cookie — set a real random value on Railway.
# The fallback below is only so the app still boots (with sessions reset
# every deploy) if someone forgets to set it; never rely on it in production.
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY", "dev-insecure-secret-key"))

def public_base_url(request: Request) -> str:
    """request.base_url reflects the scheme Railway's internal proxy used to
    reach this process, which is plain http even though the site is only
    ever visited over https — using it as-is breaks anything that must
    match an exact registered URL (Google OAuth's redirect_uri, PayPal's
    return/cancel URLs). X-Forwarded-Proto carries the scheme the browser
    actually used, so prefer that when the proxy sets it."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{proto}://{host}"


BASE_DIR = os.path.dirname(__file__)
init_db()

def soft_light(b,bl): return np.clip(np.where(bl<=0.5,b-(1-2*bl)*b*(1-b),b+(2*bl-1)*(np.where(b<=0.25,((16*b-12)*b+4)*b,np.sqrt(np.maximum(b,0)))-b)),0,1)
def blend_multiply(b,bl): return b*bl
def blend_screen(b,bl):   return 1-(1-b)*(1-bl)
def blend_overlay(b,bl):  return np.where(b<=0.5,2*b*bl,1-2*(1-b)*(1-bl))
def blend_hard_light(b,bl): return np.where(bl<=0.5,2*b*bl,1-2*(1-b)*(1-bl))
BLEND_FNS = {
    "BlendMode.SOFT_LIGHT": soft_light,
    "BlendMode.MULTIPLY":   blend_multiply,
    "BlendMode.SCREEN":     blend_screen,
    "BlendMode.OVERLAY":    blend_overlay,
    "BlendMode.HARD_LIGHT": blend_hard_light,
}


def _warp_triangle_into(dst_arr, src_arr, tri_src, tri_dst):
    """Affine-warp the triangular region `tri_src` of `src_arr` onto
    `tri_dst`'s location in `dst_arr` (mutated in place), masked to the
    triangle. Both arrays are HxWx4 uint8. Operates only on each triangle's
    small bounding box (not the full canvas), so this stays cheap even when
    called ~500 times for a fine warp mesh (see _build_src_canvas_warp)."""
    rx, ry, rw, rh = cv2.boundingRect(tri_dst.astype(np.float32))
    if rw <= 0 or rh <= 0:
        return
    sx, sy, sw_, sh_ = cv2.boundingRect(tri_src.astype(np.float32))
    if sw_ <= 0 or sh_ <= 0:
        return
    src_crop = src_arr[sy:sy + sh_, sx:sx + sw_]
    if src_crop.size == 0:
        return
    tri_src_local = (tri_src - [sx, sy]).astype(np.float32)
    tri_dst_local = (tri_dst - [rx, ry]).astype(np.float32)
    M = cv2.getAffineTransform(tri_src_local, tri_dst_local)
    warped = cv2.warpAffine(src_crop, M, (rw, rh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, tri_dst_local.astype(np.int32), 255)
    dst_h, dst_w = dst_arr.shape[:2]
    x0, y0 = max(0, rx), max(0, ry)
    x1, y1 = min(dst_w, rx + rw), min(dst_h, ry + rh)
    if x1 <= x0 or y1 <= y0:
        return
    wx0, wy0 = x0 - rx, y0 - ry
    wx1, wy1 = wx0 + (x1 - x0), wy0 + (y1 - y0)
    region_mask = mask[wy0:wy1, wx0:wx1] > 0
    dst_arr[y0:y1, x0:x1][region_mask] = warped[wy0:wy1, wx0:wx1][region_mask]


def _build_src_canvas_distort(design_img, sw, sh, corners):
    """Place `design_img` onto a full (sw,sh) transparent canvas via a single
    perspective (homography) warp from the design's own rectangle onto the
    4 given corners — corners: [[x,y]x4] in absolute source-pixel
    coordinates, order TL,TR,BR,BL (matches the client's quad convention)."""
    dimg = design_img.convert("RGBA")
    dw0, dh0 = dimg.size
    src_pts = np.float32([[0, 0], [dw0, 0], [dw0, dh0], [0, dh0]])
    dst_pts = np.float32(corners)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    arr = np.array(dimg)
    warped = cv2.warpPerspective(arr, M, (sw, sh), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return Image.fromarray(warped, mode="RGBA")


def _build_src_canvas_warp(design_img, sw, sh, mesh):
    """Place `design_img` onto a full (sw,sh) transparent canvas via a 4x4
    cubic-Bezier control mesh (mesh: 16 [x,y] absolute source-pixel points,
    row-major, same convention as the garment's own warp.mesh_x/mesh_y) —
    evaluated on a fine regular grid and rasterized triangle by triangle,
    same technique as the client-side live preview."""
    dimg = design_img.convert("RGBA")
    dw0, dh0 = dimg.size
    mesh = np.array(mesh, dtype=np.float64)
    mesh_x = mesh[:, 0].reshape(4, 4)
    mesh_y = mesh[:, 1].reshape(4, 4)

    def _bezier_basis(t):
        return np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3], axis=-1)

    N = 17  # grid resolution for triangulation — smooth enough, cheap enough
    u = np.linspace(0, 1, N); v = np.linspace(0, 1, N)
    Bu = _bezier_basis(u); Bv = _bezier_basis(v)
    grid_x = Bv @ mesh_x @ Bu.T
    grid_y = Bv @ mesh_y @ Bu.T

    canvas = np.zeros((sh, sw, 4), dtype=np.uint8)
    src_arr = np.array(dimg)
    for j in range(N - 1):
        for i in range(N - 1):
            s0, s1 = i / (N - 1) * dw0, (i + 1) / (N - 1) * dw0
            t0, t1 = j / (N - 1) * dh0, (j + 1) / (N - 1) * dh0
            quad_src = np.float32([[s0, t0], [s1, t0], [s1, t1], [s0, t1]])
            quad_dst = np.float32([
                [grid_x[j, i], grid_y[j, i]], [grid_x[j, i + 1], grid_y[j, i + 1]],
                [grid_x[j + 1, i + 1], grid_y[j + 1, i + 1]], [grid_x[j + 1, i], grid_y[j + 1, i]],
            ])
            _warp_triangle_into(canvas, src_arr, quad_src[[0, 1, 2]], quad_dst[[0, 1, 2]])
            _warp_triangle_into(canvas, src_arr, quad_src[[0, 2, 3]], quad_dst[[0, 2, 3]])
    return Image.fromarray(canvas, mode="RGBA")

class MockupEngine:
    def __init__(self, mockup_dir):
        print(f"Loading {mockup_dir}...")
        with open(os.path.join(mockup_dir, "mockup.json")) as f:
            pkg = json.load(f)

        self.canvas_w = pkg["canvas"]["width"]
        self.canvas_h = pkg["canvas"]["height"]
        warp = pkg["warp"]

        # Background photo — kept as uint8 (not the /255 float copy) so each
        # engine's resident memory stays ~4x smaller; render() converts to
        # float on the fly for the (infrequent, one-per-request) compositing.
        self.bg_u8 = np.array(Image.open(os.path.join(mockup_dir,"shirt_base.png")).convert("RGBA"))

        # T-Shirt group mask (shirt silhouette)
        self.shirt_mask = np.array(Image.open(os.path.join(mockup_dir,"shirt_full_mask.png")).convert("L")).astype(np.float32)/255
        self.shirt_mask_3d = self.shirt_mask[:,:,np.newaxis]

        # Shirt color (white by default = 255,255,255)
        self.shirt_color = np.array([1.0, 1.0, 1.0])

        # Overlay layers — also kept as uint8 for the same memory reason.
        self.overlays = []
        for ov in pkg["overlays"]:
            img_u8 = np.array(Image.open(os.path.join(mockup_dir,ov["file"])).convert("RGBA"))
            self.overlays.append((img_u8, ov["opacity"], ov["blend_mode"]))

        # Displacement map (Photoshop Filter > Distort > Displace, applied to the
        # design after it's warped onto the shirt, in canvas space). Optional.
        self.displace = None
        disp = pkg.get("displace")
        if disp:
            dmap = np.array(Image.open(os.path.join(mockup_dir,disp["map"])).convert("L")).astype(np.float32)/255
            self.displace = {
                "map": dmap,
                "h_scale": disp.get("h_scale", 10),
                "v_scale": disp.get("v_scale", 10),
            }

        # Warp setup
        src_w = warp["bounds"]["right"]
        src_h = warp["bounds"]["bottom"]
        self.src_w = src_w; self.src_h = src_h
        self.print_zone = pkg["print_zone"]

        tx = warp["transform"]
        canvas_corners = np.float32([[tx[0],tx[1]],[tx[2],tx[3]],[tx[4],tx[5]],[tx[6],tx[7]]])
        src_corners    = np.float32([[0,0],[src_w,0],[src_w,src_h],[0,src_h]])
        H_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_corners, canvas_corners))

        # Photoshop's Custom Envelope Warp mesh is a 4x4 tensor-product CUBIC
        # BEZIER control net — the surface only actually touches the 4 corner
        # points; the other 12 are Bezier tangent handles that shape curvature
        # without being touched. Feeding all 16 into a scattered interpolator
        # (the old approach) forces the surface through every handle too,
        # which produced wavy, escalating distortion that didn't match
        # Photoshop's real (smooth) render. Instead, evaluate the true
        # bicubic Bezier surface on a dense regular grid, then invert that
        # dense/smooth sampling.
        mx = np.array(warp["mesh_x"]).reshape(4,4); my = np.array(warp["mesh_y"]).reshape(4,4)

        def _bezier_basis(t):
            return np.stack([(1-t)**3, 3*t*(1-t)**2, 3*t**2*(1-t), t**3], axis=-1)

        _N = 65
        _u = np.linspace(0,1,_N); _v = np.linspace(0,1,_N)
        _Bu = _bezier_basis(_u); _Bv = _bezier_basis(_v)
        _Sx = _Bv @ mx @ _Bu.T; _Sy = _Bv @ my @ _Bu.T
        reg_x = np.tile(_u,_N)*src_w; reg_y = np.repeat(_v,_N)*src_h
        displaced = np.column_stack([_Sx.ravel(),_Sy.ravel()])

        print("Precomputing warp map...")
        cw,ch = self.canvas_w,self.canvas_h

        # Compute the warp on a reduced grid (cheap: a few hundred thousand
        # points regardless of canvas size) instead of per-pixel cubic
        # griddata over the full canvas (which can be many millions of
        # points for a large canvas/warp region and was blowing up memory
        # and taking minutes on some mockups). The reduced grid is then
        # upsampled to full canvas resolution with cheap bilinear resize —
        # visually indistinguishable since the warp itself is a smooth,
        # low-frequency deformation defined by just 16 control points.
        rw,rh = 512,768
        ys_r,xs_r = np.mgrid[0:rh,0:rw]
        px_r = (xs_r.ravel()+0.5)/rw*cw
        py_r = (ys_r.ravel()+0.5)/rh*ch
        cp_h_r = np.column_stack([px_r,py_r,np.ones_like(px_r)])
        sp_h_r = (H_inv@cp_h_r.T).T
        src_x_r = sp_h_r[:,0]/sp_h_r[:,2]; src_y_r = sp_h_r[:,1]/sp_h_r[:,2]
        in_r_r = (src_x_r>=-300)&(src_x_r<=src_w+300)&(src_y_r>=-300)&(src_y_r<=src_h+300)
        rx_r = np.full(rw*rh,np.nan); ry_r = np.full(rw*rh,np.nan)
        ridx_r = np.where(in_r_r)[0]
        rsp_r = np.column_stack([src_x_r[in_r_r],src_y_r[in_r_r]])
        rxv = griddata(displaced,reg_x,rsp_r,method="linear")
        ryv = griddata(displaced,reg_y,rsp_r,method="linear")
        valid_r = ~(np.isnan(rxv)|np.isnan(ryv))
        rx_r[ridx_r[valid_r]] = rxv[valid_r]
        ry_r[ridx_r[valid_r]] = ryv[valid_r]
        rx_grid = rx_r.reshape(rh,rw); ry_grid = ry_r.reshape(rh,rw)
        valid_grid = (~np.isnan(rx_grid)).astype(np.float32)

        rx_full = cv2.resize(np.nan_to_num(rx_grid,nan=0.0),(cw,ch),interpolation=cv2.INTER_LINEAR)
        ry_full = cv2.resize(np.nan_to_num(ry_grid,nan=0.0),(cw,ch),interpolation=cv2.INTER_LINEAR)
        valid_full = cv2.resize(valid_grid,(cw,ch),interpolation=cv2.INTER_LINEAR) > 0.5

        ridx = np.where(valid_full.ravel())[0]
        self._rx = rx_full.ravel()[ridx]
        self._ry = ry_full.ravel()[ridx]
        self._ridx = ridx
        print(f"Ready. Canvas: {cw}x{ch}")

    def _apply_displace(self, rgba):
        """Photoshop Filter > Distort > Displace, 'Stretch To Fit' + 'Repeat Edge
        Pixels': map value 128 (0.5) = no shift, 0 = -scale, 255 = +scale, same
        channel drives both axes for a grayscale map. Backward-sampling (pull)."""
        d = self.displace
        h, w = rgba.shape[0], rgba.shape[1]
        dmap = d["map"]
        dx = (dmap - 0.5) * 2 * d["h_scale"]
        dy = (dmap - 0.5) * 2 * d["v_scale"]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        src_x = np.clip(xs - dx, 0, w - 1)
        src_y = np.clip(ys - dy, 0, h - 1)
        out = np.empty_like(rgba)
        for c in range(rgba.shape[2]):
            out[:, :, c] = map_coordinates(rgba[:, :, c], [src_y, src_x], order=1, mode="nearest")
        return out

    def render(self, design_img, x=0, y=0, w=None, h=None, color="#ffffff",
               distort_corners=None, warp_mesh=None):
        sw,sh = int(self.src_w),int(self.src_h)
        pz = self.print_zone
        if w is None: w = pz['x1']-pz['x0']
        if h is None: h = pz['y1']-pz['y0']
        if x == 0 and y == 0:
            x, y = pz['x0'], pz['y0']

        # === Step 1: Background ===
        result = self.bg_u8.astype(np.float32)/255

        # === Step 2: Shirt color fill (clipped to shirt mask) ===
        r = int(color[1:3],16)/255
        g = int(color[3:5],16)/255
        b = int(color[5:7],16)/255
        color_arr = np.zeros_like(result)
        color_arr[:,:,0] = r; color_arr[:,:,1] = g; color_arr[:,:,2] = b
        color_arr[:,:,3] = 1.0
        # Composite color fill over background, clipped by shirt mask
        m = self.shirt_mask_3d
        result[:,:,:3] = result[:,:,:3]*(1-m) + color_arr[:,:,:3]*m
        result[:,:,3] = np.maximum(result[:,:,3], self.shirt_mask)

        # === Step 3: Warp design onto shirt (clipped by shirt_mask) ===
        # distort_corners/warp_mesh (new, optional — client-side "Distort"/
        # "Warp" tools) replace the plain resize+paste with a perspective or
        # mesh warp of the design itself, in this SAME absolute source-pixel
        # space, before the existing garment warp below (self._rx/self._ry)
        # ever runs — that part doesn't know or care how src_canvas was
        # built, so this only changes step 3's placement, nothing else.
        if warp_mesh is not None:
            src_canvas = _build_src_canvas_warp(design_img, sw, sh, warp_mesh)
        elif distort_corners is not None:
            src_canvas = _build_src_canvas_distort(design_img, sw, sh, distort_corners)
        else:
            src_canvas = Image.new("RGBA",(sw,sh),(0,0,0,0))
            src_canvas.paste(design_img.resize((w,h),Image.LANCZOS),(x,y))
        dw,dh = src_canvas.size
        design_arr = np.array(src_canvas).astype(np.float32)

        nx = self._rx/self.src_w*dw; ny = self._ry/self.src_h*dh
        valid = (~np.isnan(nx))&(~np.isnan(ny))&(nx>=0)&(nx<dw-1)&(ny>=0)&(ny<dh-1)
        warped = np.zeros((self.canvas_h,self.canvas_w,4),dtype=np.float32)
        vi = np.where(valid)[0]; fi = self._ridx[vi]
        warped[fi//self.canvas_w,fi%self.canvas_w] = design_arr[ny[valid].astype(int),nx[valid].astype(int)]
        # Clip to shirt mask
        warped[:,:,3] = warped[:,:,3] * self.shirt_mask

        # Fabric-texture displacement (Photoshop Displace filter equivalent)
        if self.displace:
            warped = self._apply_displace(warped)

        # Alpha-composite the design onto the shirt with its own true colors —
        # NOT multiplied against the (possibly colored) shirt underneath, which
        # was tinting the whole design toward whatever shirt color was picked
        # (e.g. a green shirt turning a black/white design green). Fabric
        # folds/shading still come through afterward via the overlay layers
        # below, which apply on top of the design too.
        alpha = warped[:,:,3:4]/255
        design_rgb = warped[:,:,:3]/255
        result[:,:,:3] = result[:,:,:3]*(1-alpha) + design_rgb*alpha

        # === Step 4: Overlay layers (clipped to shirt mask) ===
        for img_u8,opacity,blend_mode in self.overlays:
            fn = BLEND_FNS.get(blend_mode)
            if fn:
                img = img_u8.astype(np.float32)/255
                blended = fn(result[:,:,:3], img[:,:,:3])
                effective = opacity * img[:,:,3:4] * self.shirt_mask_3d
                result[:,:,:3] = result[:,:,:3]*(1-effective) + blended*effective

        return Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)).convert("RGB")

# Mockup registry — NOT eagerly loaded. With 60+ packages (some at large
# 2000x3000 canvas size), constructing every MockupEngine at startup held
# them all resident in memory simultaneously and OOM-killed the container.
# Instead we just record which package names exist here, and load/cache
# engines lazily (see get_engine below), bounding memory to a handful of
# recently-used mockups instead of all of them at once.
import collections
MOCKUP_NAMES = []
for name in ["mockup1_package","mockup14_package","mockup3_package","mockup4_package",
             "mockup5_package","mockup6_package","mockup7_package","mockup8_package",
             "mockup9_package","mockup10_package","mockup11_package","mockup12_package",
             "mockup13_package","mockup15_package","mockup16_package","mockup17_package",
             "mockup18_package","mockup19_package","mockup20_package","mockup21_package",
             "mockup22_package","mockup23_package","mockup24_package","mockup25_package",
             "mockup26_package","mockup27_package","mockup28_package","mockup29_package",
             "mockup30_package","mockup31_package","mockup32_package","mockup33_package",
             "mockup34_package","mockup35_package","mockup36_package","mockup37_package",
             "mockup38_package","mockup39_package","mockup40_package","mockup41_package",
             "mockup42_package","mockup43_package","mockup44_package","mockup45_package",
             "mockup46_package","mockup47_package","mockup48_package","mockup49_package",
             "mockup50_package","mockup51_package","mockup52_package","mockup53_package",
             "mockup54_package","mockup55_package","mockup56_package","mockup57_package",
             "mockup58_package","mockup59_package","mockup60_package","mockup61_package",
             "mockup62_package","mockup63_package","mockup64_package","mockup65_package",
             "mockup66_package","mockup67_package",
             # 07.09 import — 6 raw PSD template packs converted via
             # tools/import_psd_mockups.py (see that script's docstring for
             # how the PSD Smart-Object warp data maps onto mockup.json).
             "mockup68_package","mockup69_package","mockup70_package","mockup71_package",
             "mockup72_package","mockup73_package","mockup74_package","mockup75_package",
             "mockup76_package","mockup77_package","mockup78_package","mockup79_package",
             "mockup80_package","mockup81_package","mockup82_package","mockup83_package",
             "mockup84_package","mockup85_package","mockup86_package","mockup87_package",
             "mockup88_package","mockup89_package","mockup90_package","mockup91_package",
             "mockup92_package","mockup93_package","mockup94_package","mockup95_package",
             "mockup96_package","mockup97_package","mockup98_package","mockup99_package",
             "mockup100_package","mockup101_package","mockup102_package","mockup103_package",
             "mockup104_package","mockup105_package","mockup106_package","mockup107_package",
             "mockup108_package","mockup109_package","mockup110_package","mockup111_package",
             "mockup112_package","mockup113_package","mockup114_package","mockup115_package",
             "mockup116_package","mockup117_package","mockup118_package","mockup119_package",
             "mockup120_package","mockup121_package","mockup122_package","mockup123_package",
             "mockup124_package","mockup125_package","mockup126_package","mockup127_package",
             "mockup128_package","mockup129_package","mockup130_package","mockup131_package",
             "mockup132_package","mockup133_package","mockup134_package","mockup135_package",
             "mockup136_package","mockup137_package","mockup138_package","mockup139_package",
             "mockup140_package","mockup141_package"]:
    d = os.path.join(BASE_DIR, name)
    if os.path.exists(os.path.join(d, "mockup.json")):
        MOCKUP_NAMES.append(name)

_ENGINE_CACHE = collections.OrderedDict()
_ENGINE_CACHE_MAX = int(os.environ.get("ENGINE_CACHE_MAX", "4"))

def get_engine(name):
    """Load a MockupEngine on first use and keep only the last
    _ENGINE_CACHE_MAX resident, evicting the least-recently-used one once
    the cache is full — bounds memory regardless of how many mockup
    packages the site has."""
    if name in _ENGINE_CACHE:
        _ENGINE_CACHE.move_to_end(name)
        return _ENGINE_CACHE[name]
    if name not in MOCKUP_NAMES:
        return None
    d = os.path.join(BASE_DIR, name)
    engine = MockupEngine(d)
    _ENGINE_CACHE[name] = engine
    _ENGINE_CACHE.move_to_end(name)
    while len(_ENGINE_CACHE) > _ENGINE_CACHE_MAX:
        _ENGINE_CACHE.popitem(last=False)
    return engine


# ── Auth ─────────────────────────────────────────────────────────────────
@app.post("/api/register")
def register(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    email = validate_email(email)
    validate_password(password)
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "An account with this email already exists")
    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    create_user_session(request, user.id)
    return {"email": user.email}


@app.post("/api/login")
def login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    email = validate_email(email)
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    create_user_session(request, user.id)
    return {"email": user.email}


@app.post("/api/logout")
def logout(request: Request):
    clear_user_session(request)
    return {"ok": True}


@app.get("/api/auth/google/login")
def google_login(request: Request):
    if not google_oauth.configured():
        raise HTTPException(503, "Google sign-in is not configured yet")
    redirect_uri = f"{public_base_url(request)}/api/auth/google/callback"
    state = google_oauth.new_state()
    request.session["google_oauth_state"] = state
    return RedirectResponse(google_oauth.authorize_url(redirect_uri, state))


@app.get("/api/auth/google/callback")
def google_callback(request: Request, code: str = None, state: str = None, error: str = None,
                     db: Session = Depends(get_db)):
    if error:
        return RedirectResponse(f"/?auth=error")
    expected_state = request.session.pop("google_oauth_state", None)
    if not code or not state or not expected_state or state != expected_state:
        return RedirectResponse(f"/?auth=error")
    redirect_uri = f"{public_base_url(request)}/api/auth/google/callback"
    try:
        tokens = google_oauth.exchange_code(code, redirect_uri)
        info = google_oauth.get_userinfo(tokens["access_token"])
    except Exception:
        return RedirectResponse(f"/?auth=error")

    google_id = info.get("sub")
    email = (info.get("email") or "").strip().lower()
    if not google_id or not email:
        return RedirectResponse(f"/?auth=error")

    user = db.query(User).filter(User.google_id == google_id).first()
    if not user:
        # Link to an existing email/password account if one matches, so
        # someone who registered manually can also use "Continue with
        # Google" for the same account instead of getting a duplicate.
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.google_id = google_id
        else:
            user = User(email=email, google_id=google_id, password_hash=None)
            db.add(user)
        db.commit()
        db.refresh(user)

    create_user_session(request, user.id)
    return RedirectResponse("/?auth=success")


@app.get("/api/me")
def me(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return {"logged_in": False}
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    return {
        "logged_in": True,
        "email": user.email,
        "subscription": {
            "status": sub.status if sub else "none",
            "current_period_end": sub.current_period_end.isoformat() if sub and sub.current_period_end else None,
        },
    }


# ── Subscription / PayPal ───────────────────────────────────────────────
@app.post("/api/subscription/create")
def create_subscription(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    plan_id = os.environ.get("PAYPAL_PLAN_ID")
    if not paypal.configured() or not plan_id:
        raise HTTPException(503, "Payments are not configured yet")
    base = public_base_url(request)
    result = paypal.create_subscription(
        plan_id=plan_id,
        return_url=f"{base}/api/subscription/return",
        cancel_url=f"{base}/?subscribe=cancelled",
        custom_id=str(user.id),
    )
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    if not sub:
        sub = Subscription(user_id=user.id)
        db.add(sub)
    sub.status = "pending"
    sub.paypal_subscription_id = result["id"]
    sub.plan_id = plan_id
    db.commit()
    return {"approve_url": result["approve_url"]}


@app.get("/api/subscription/return")
def subscription_return(subscription_id: str, db: Session = Depends(get_db)):
    """PayPal redirects the browser here after the user approves the
    subscription on PayPal's site. We double-check the status directly with
    PayPal's API (never trust query params alone) before marking it active."""
    sub = db.query(Subscription).filter(Subscription.paypal_subscription_id == subscription_id).first()
    if not sub:
        return RedirectResponse("/?subscribe=error")
    info = paypal.get_subscription(subscription_id)
    if info.get("status") == "ACTIVE":
        sub.status = "active"
        next_billing = info.get("billing_info", {}).get("next_billing_time")
        if next_billing:
            sub.current_period_end = datetime.datetime.fromisoformat(next_billing.replace("Z", "+00:00"))
        db.commit()
        return RedirectResponse("/?subscribe=success")
    return RedirectResponse("/?subscribe=pending")


@app.post("/api/subscription/cancel")
def cancel_subscription_endpoint(user: User = Depends(require_user), db: Session = Depends(get_db)):
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    if not sub or not sub.paypal_subscription_id:
        raise HTTPException(400, "No active subscription")
    paypal.cancel_subscription(sub.paypal_subscription_id)
    sub.status = "cancelled"
    db.commit()
    return {"ok": True}


@app.post("/api/paypal/webhook")
async def paypal_webhook(request: Request, db: Session = Depends(get_db)):
    body = await request.body()
    if not paypal.verify_webhook_signature(dict(request.headers), body):
        raise HTTPException(400, "Invalid webhook signature")
    event = json.loads(body)
    event_type = event.get("event_type", "")
    resource = event.get("resource", {})
    paypal_sub_id = resource.get("id")
    if not paypal_sub_id:
        return {"ok": True}
    sub = db.query(Subscription).filter(Subscription.paypal_subscription_id == paypal_sub_id).first()
    if not sub:
        return {"ok": True}
    if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
        sub.status = "active"
    elif event_type in ("BILLING.SUBSCRIPTION.CANCELLED", "BILLING.SUBSCRIPTION.SUSPENDED"):
        sub.status = "cancelled"
    elif event_type == "BILLING.SUBSCRIPTION.EXPIRED":
        sub.status = "expired"
    elif event_type == "PAYMENT.SALE.COMPLETED":
        sub.status = "active"
    db.commit()
    return {"ok": True}


@app.post("/render")
async def render_mockup(
    file: UploadFile = File(...),
    x: int = Form(0), y: int = Form(0),
    w: int = Form(None), h: int = Form(None),
    rot: float = Form(0),
    color: str = Form("#ffffff"),
    mockup: str = Form("mockup3_package"),
    # Optional — set only when the client's "Distort" (4 corners) or "Warp"
    # (4x4 mesh) tool is active; JSON-encoded [[x,y],...] in absolute
    # source-pixel coordinates (same space as x/y/w/h above). See
    # MockupEngine.render / _build_src_canvas_distort / _build_src_canvas_warp.
    distort_corners: str = Form(None),
    warp_mesh: str = Form(None),
    # TEMP: login requirement disabled for testing — re-enable before launch
    # user: User = Depends(require_user),
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # TEMP: subscription check disabled for testing — re-enable before launch
    # sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    # if not sub or not sub.is_active():
    #     raise HTTPException(402, "Active subscription required to generate high-res downloads")
    engine = get_engine(mockup) or get_engine(MOCKUP_NAMES[0])
    if not file.content_type.startswith("image/"):
        raise HTTPException(400,"File must be an image")
    data = await file.read()
    if len(data) > 20*1024*1024:
        raise HTTPException(400,"File too large")
    try:
        design = Image.open(io.BytesIO(data)).convert("RGBA")
    except:
        raise HTTPException(400,"Could not read image")

    parsed_corners = None
    parsed_mesh = None
    try:
        if warp_mesh:
            parsed_mesh = json.loads(warp_mesh)
        elif distort_corners:
            parsed_corners = json.loads(distort_corners)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid distort_corners/warp_mesh")

    if rot and not parsed_corners and not parsed_mesh:
        # Distort/Warp corners already encode any rotation directly, so the
        # client skips sending rot (leaves it 0) whenever either is active —
        # this pre-rotation only applies to the plain move/resize placement.
        # Negated to match the client canvas's rotation direction (canvas
        # ctx.rotate() is clockwise-positive, PIL's rotate() is
        # counter-clockwise-positive) so the live preview and the final
        # high-res render look the same for a given slider value.
        design = design.rotate(-rot, expand=True, resample=Image.BICUBIC)

    result = engine.render(design,x=x,y=y,w=w,h=h,color=color,
                            distort_corners=parsed_corners,warp_mesh=parsed_mesh)
    buf = io.BytesIO()
    result.save(buf,format="PNG",optimize=True)
    buf.seek(0)
    return Response(content=buf.read(),media_type="image/png",
                    headers={"Content-Disposition":"attachment; filename=mockup.png"})

@app.get("/health")
def health():
    return {"status":"ok","mockups":MOCKUP_NAMES}

app.mount("/",StaticFiles(directory=BASE_DIR,html=True),name="static")
