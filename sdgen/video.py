import os, numpy as np, math
from typing import List
from PIL import Image

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio


def build_video(frames: List[str], out_path: str, fps: int, blend_mode: str, blend_steps: int, target_duration: float = 0.0) -> str:
    base_frames = [Image.open(p).convert('RGB') for p in frames]
    if len(base_frames) < 2:
        raise ValueError('Not enough frames for video')
    if target_duration and target_duration > 0 and fps <= 0:
        # fps so wählen, dass Gesamtdauer nahe target_duration (~frames_out/fps)
        # first estimate without blends
        est_frames = len(base_frames) + max(0, (len(base_frames)-1) * blend_steps)
        fps = max(1, min(60, round(est_frames / target_duration)))

    def pil_to_np(im): return np.array(im)
    def np_to_pil(arr): return Image.fromarray(arr.astype('uint8'))

    out_frames = []
    have_flow = False
    if blend_mode == 'flow':
        try:
            import cv2  # type: ignore
            have_flow = True
        except Exception:
            blend_mode = 'linear'

    if blend_mode == 'none' or blend_steps <= 0:
        out_frames = base_frames
    else:
        for i in range(len(base_frames)-1):
            A = base_frames[i]
            B = base_frames[i+1]
            out_frames.append(A)
            A_np = pil_to_np(A)
            B_np = pil_to_np(B)
            if have_flow:
                import cv2  # type: ignore
                h, w, _ = A_np.shape
                scale = 512 / max(h, w)
                if scale < 1:
                    As = cv2.resize(A_np, (int(w*scale), int(h*scale)))
                    Bs = cv2.resize(B_np, (int(w*scale), int(h*scale)))
                else:
                    As, Bs = A_np, B_np
                grayA = cv2.cvtColor(As, cv2.COLOR_RGB2GRAY)
                grayB = cv2.cvtColor(Bs, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(grayA, grayB, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                fx = cv2.resize(flow[...,0], (w, h))
                fy = cv2.resize(flow[...,1], (w, h))
            for s in range(1, blend_steps+1):
                t = s / (blend_steps + 1)
                if have_flow:
                    import cv2  # type: ignore
                    h, w, _ = A_np.shape
                    yy, xx = np.mgrid[0:h, 0:w]
                    map_x = (xx + fx * t).astype('float32')
                    map_y = (yy + fy * t).astype('float32')
                    warpedA = cv2.remap(A_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    blended = warpedA * (1 - t) + B_np * t
                else:
                    blended = A_np * (1 - t) + B_np * t
                out_frames.append(np_to_pil(blended.clip(0,255)))
        out_frames.append(base_frames[-1])

    frames_np = [pil_to_np(f) for f in out_frames]
    imageio.mimwrite(out_path, frames_np, fps=fps, quality=8)
    return out_path
