"""COLMAP 前景掩码的 CPU 回归测试：python utils/_test_mask.py"""
import os
import sys
import tempfile

import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mask_utils import load_colmap_foreground_mask

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"{name}: OK")
    else:
        print(f"{name}: FAIL {detail}")
        failures.append(name)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        images = os.path.join(tmp, "images")
        masks = os.path.join(tmp, "masks")
        os.makedirs(images)
        os.makedirs(masks)
        H, W = 6, 8
        image = np.full((H, W, 3), 0.5, dtype=np.float32)

        # 1. 无 mask：全 1，图不变
        m = load_colmap_foreground_mask(images, "0001.png", H, W)
        check("no-mask all ones", m.shape == (H, W) and np.all(m == 1.0))
        check("no-mask image unchanged", np.allclose(image * m[..., None], image))

        # 2. 灰度 mask：左上 3x4 为前景
        gray = np.zeros((H, W), dtype=np.uint8)
        gray[:3, :4] = 255
        imageio.imwrite(os.path.join(masks, "0001.png"), gray)
        m = load_colmap_foreground_mask(images, "0001.png", H, W)
        check("gray binary", set(np.unique(m).tolist()) == {0.0, 1.0})
        check("gray foreground", np.all(m[:3, :4] == 1.0) and np.all(m[3:] == 0.0) and np.all(m[:, 4:] == 0.0))
        masked = image * m[..., None]
        check("gray background zeroed", np.all(masked[3:] == 0.0))
        check("gray foreground kept", np.allclose(masked[:3, :4], 0.5))

        # 3. 抗锯齿灰度值：127 以下仍视为背景
        soft = np.full((H, W), 127, dtype=np.uint8)
        soft[0, 0] = 128
        imageio.imwrite(os.path.join(masks, "0001.png"), soft)
        m = load_colmap_foreground_mask(images, "0001.png", H, W)
        check("threshold 127/128", m[0, 0] == 1.0 and np.all(m[0, 1:] == 0.0))

        # 4. RGBA：取 alpha
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = 255          # RGB 全白但 alpha 为 0，应视为背景
        rgba[2:4, 2:6, 3] = 255
        imageio.imwrite(os.path.join(masks, "0002.png"), rgba)
        m = load_colmap_foreground_mask(images, "0002.png", H, W)
        check("rgba uses alpha", np.all(m[2:4, 2:6] == 1.0) and m[0, 0] == 0.0)

        # 5. 分辨率不一致：缩放回图片尺寸
        small = np.full((3, 4), 255, dtype=np.uint8)
        imageio.imwrite(os.path.join(masks, "0003.png"), small)
        m = load_colmap_foreground_mask(images, "0003.png", H, W)
        check("resize to image", m.shape == (H, W) and np.all(m == 1.0))

    print(f"\nFAILURES: {len(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
