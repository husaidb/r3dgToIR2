import os
import cv2
import imageio
import numpy as np


def load_colmap_foreground_mask(images_folder, image_name, height, width):
    """COLMAP 路径的前景掩码：<images 同级的 masks>/<图片文件名>，白=前景=1，背景=0。

    找不到 mask 时返回全 1（不抠背景）。支持灰度、二值、RGBA（取 alpha），
    尺寸不一致时按最近邻缩放到图片尺寸。
    """
    mask_path = os.path.join(os.path.dirname(images_folder), "masks", image_name)
    if not os.path.exists(mask_path):
        return np.ones((height, width), dtype=np.float32)

    m = imageio.imread(mask_path)
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 else cv2.cvtColor(m[..., :3], cv2.COLOR_RGB2GRAY)
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    if m.shape[:2] != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return (m > 0.5).astype(np.float32)
