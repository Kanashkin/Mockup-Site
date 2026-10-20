import sys, numpy as np
sys.path.insert(0,'tools')
from import_psd_mockups import get_warp_data, get_print_zone
from psd_tools import PSDImage
import cv2
from scipy.interpolate import griddata

def _bezier_basis(t):
    return np.stack([(1-t)**3,3*t*(1-t)**2,3*t**2*(1-t),t**3],axis=-1)

def full_pipeline_footprint_test(path):
    psd = PSDImage.open(path)
    layers = list(psd)
    smart = [l for l in layers if l.kind=='smartobject'][0]
    transform, src_w, src_h, mesh_x, mesh_y = get_warp_data(smart)
    pz = get_print_zone(smart)
    true_bbox = smart.bbox

    canvas_w, canvas_h = psd.size
    tx = transform
    canvas_corners = np.float32([[tx[0],tx[1]],[tx[2],tx[3]],[tx[4],tx[5]],[tx[6],tx[7]]])
    src_corners = np.float32([[0,0],[src_w,0],[src_w,src_h],[0,src_h]])
    H_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_corners, canvas_corners))

    mx = np.array(mesh_x).reshape(4,4); my = np.array(mesh_y).reshape(4,4)
    N=33
    u=np.linspace(0,1,N); v=np.linspace(0,1,N)
    Bu=_bezier_basis(u); Bv=_bezier_basis(v)
    Sx=Bv@mx@Bu.T; Sy=Bv@my@Bu.T
    reg_x=np.tile(u,N)*src_w; reg_y=np.repeat(v,N)*src_h
    displaced=np.column_stack([Sx.ravel(),Sy.ravel()])

    RW,RH=200,150
    ys_r,xs_r = np.mgrid[0:RH,0:RW]
    px_r=(xs_r.ravel()+0.5)/RW*canvas_w; py_r=(ys_r.ravel()+0.5)/RH*canvas_h
    cp_h=np.column_stack([px_r,py_r,np.ones_like(px_r)])
    sp_h=(H_inv@cp_h.T).T
    src_x=sp_h[:,0]/sp_h[:,2]; src_y=sp_h[:,1]/sp_h[:,2]
    in_range=(src_x>=-300)&(src_x<=src_w+300)&(src_y>=-300)&(src_y<=src_h+300)
    rxv=griddata(displaced,reg_x,np.column_stack([src_x[in_range],src_y[in_range]]),method='linear')
    ryv=griddata(displaced,reg_y,np.column_stack([src_x[in_range],src_y[in_range]]),method='linear')
    valid=~(np.isnan(rxv)|np.isnan(ryv))
    in_pz = valid & (rxv>=pz['x0'])&(rxv<=pz['x1'])&(ryv>=pz['y0'])&(ryv<=pz['y1'])
    idx = np.where(in_range)[0][in_pz]
    if idx.size==0:
        return None, true_bbox, 0.0
    fx = px_r[idx]; fy = py_r[idx]
    footprint_bbox = (fx.min(),fy.min(),fx.max(),fy.max())
    ax0,ay0,ax1,ay1 = footprint_bbox
    bx0,by0,bx1,by1 = true_bbox
    ix0,iy0=max(ax0,bx0),max(ay0,by0)
    ix1,iy1=min(ax1,bx1),min(ay1,by1)
    iw,ih=max(0,ix1-ix0),max(0,iy1-iy0)
    inter=iw*ih
    areaA=(ax1-ax0)*(ay1-ay0); areaB=(bx1-bx0)*(by1-by0)
    iou = inter/(areaA+areaB-inter+1e-6)
    return footprint_bbox, true_bbox, iou

if __name__=='__main__':
    for path in sys.argv[1:]:
        try:
            fb, tb, iou = full_pipeline_footprint_test(path)
            print(path, 'footprint', fb, 'true', tb, 'iou', round(iou,3) if fb else 0.0)
        except Exception as e:
            print(path, 'ERROR', e)
