"""Classical image-processing helpers shared by training and inference."""
import cv2
import numpy as np

BODY_T = 100        # gray < BODY_T  -> fly body only (wings/legs are lighter)
WING_T = 195        # gray < WING_T  -> body + wings (+ legs, removed by erosion)
MIN_AREA = 20
K3 = np.ones((3, 3), np.uint8)

try:
    from skimage.feature import hog as _hog

    def hog(patch):
        """Histogram of Oriented Gradients: 9 bins, 8x8-pixel cells, 2x2-cell blocks.
        128x64 patch -> 3780-d vector, 96x80 patch -> 3564-d vector."""
        return _hog(patch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm="L2-Hys", feature_vector=True)
except ImportError:
    def hog(patch):
        """Pure NumPy Histogram of Oriented Gradients matching skimage default."""
        image = np.asarray(patch, dtype=np.float64)
        s_row, s_col = image.shape
        c_row, c_col = (8, 8)
        b_row, b_col = (2, 2)
        orientations = 9

        g_row = np.zeros_like(image)
        g_col = np.zeros_like(image)
        g_row[1:-1, :] = image[2:, :] - image[:-2, :]
        g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]

        magnitude = np.hypot(g_col, g_row)
        orientation = (np.rad2deg(np.arctan2(g_row, g_col)) % 180.0)

        n_cells_row = s_row // c_row
        n_cells_col = s_col // c_col

        h_crop = n_cells_row * c_row
        w_crop = n_cells_col * c_col
        mag = magnitude[:h_crop, :w_crop]
        ori = orientation[:h_crop, :w_crop]

        mag_cells = mag.reshape(n_cells_row, c_row, n_cells_col, c_col).transpose(0, 2, 1, 3)
        ori_cells = ori.reshape(n_cells_row, c_row, n_cells_col, c_col).transpose(0, 2, 1, 3)

        bin_width = 180.0 / orientations
        bin_idx = np.clip(np.floor(ori_cells / bin_width).astype(int), 0, orientations - 1)

        hist = np.zeros((n_cells_row, n_cells_col, orientations), dtype=np.float64)
        for i in range(orientations):
            hist[:, :, i] = np.sum(np.where(bin_idx == i, mag_cells, 0.0), axis=(2, 3)) / (c_row * c_col)

        n_blocks_row = n_cells_row - b_row + 1
        n_blocks_col = n_cells_col - b_col + 1
        eps = 1e-5

        normalized_blocks = np.zeros((n_blocks_row, n_blocks_col, b_row, b_col, orientations), dtype=np.float64)
        for r in range(n_blocks_row):
            for c in range(n_blocks_col):
                block = hist[r:r + b_row, c:c + b_col, :]
                val = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
                val = np.minimum(val, 0.2)
                val = val / np.sqrt(np.sum(val ** 2) + eps ** 2)
                normalized_blocks[r, c] = val

        return normalized_blocks.ravel()


def get_contours(gray):
    """Threshold the frame to the fly bodies and return external contours."""
    bw = (gray < BODY_T).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, K3)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) >= MIN_AREA]


def contour_stats(c):
    """Area, centroid, principal-axis angle (image moments) and aspect ratio."""
    m = cv2.moments(c)
    a = m["m00"]
    if a < 1:
        return None
    cx, cy = m["m10"] / a, m["m01"] / a
    mu20, mu02, mu11 = m["mu20"] / a, m["mu02"] / a, m["mu11"] / a
    phi = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))   # axis, ambiguous by 180
    root = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = 0.5 * (mu20 + mu02) + 0.5 * root, 0.5 * (mu20 + mu02) - 0.5 * root
    aspect = float(np.sqrt(l1 / max(l2, 1e-6)))
    return {"area": float(a), "cx": cx, "cy": cy, "phi": float(phi), "aspect": aspect}


def upright(img, cx, cy, heading_deg, scale, out_w, out_h, ox, oy):
    """Rotate/scale so that `heading_deg` points up and (cx, cy) lands on (ox, oy)."""
    M = cv2.getRotationMatrix2D((cx, cy), heading_deg + 90.0, scale)
    M[0, 2] += ox - cx
    M[1, 2] += oy - cy
    return cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def masked_fly(gray, c, pad=4):
    """Crop around one contour and white-out everything except that fly."""
    H, W = gray.shape
    x, y, w, h = cv2.boundingRect(c)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    sub = gray[y0:y1, x0:x1]
    m = np.zeros(sub.shape, np.uint8)
    cv2.drawContours(m, [c], -1, 255, -1, offset=(-x0, -y0))
    m = cv2.dilate(m, K3)
    return np.where(m > 0, sub, 255).astype(np.uint8), x0, y0


def orientation_patch(gray, c, st):
    """Body-only 128x64 patch with the fly's long axis vertical (head up OR down)."""
    sub, x0, y0 = masked_fly(gray, c)
    return upright(sub, st["cx"] - x0, st["cy"] - y0, st["phi"], 2.0, 64, 128, 32, 64)


def orientation_feature(patch):
    return hog(patch)


def wing_patches(gray, cx, cy, heading_deg):
    """Return (right_patch, left_patch_mirrored), each 96x80, after blur/threshold/erode."""
    p = upright(gray, cx, cy, heading_deg, 1.5, 160, 96, 80, 40)
    p = cv2.medianBlur(p, 3)
    bw = (p < WING_T).astype(np.uint8)
    _, lab = cv2.connectedComponents(bw, connectivity=8)
    l = lab[40, 80]
    keep = (lab == l).astype(np.uint8) if l > 0 else bw
    keep = cv2.erode(keep, K3)                       # removes thin legs, keeps wings
    out = np.where(keep > 0, p, 255).astype(np.uint8)
    right = out[:, 77:157]
    left = out[:, 4:84][:, ::-1]
    return np.ascontiguousarray(right), np.ascontiguousarray(left)


def wing_feature(patch):
    return hog(patch)


def ang_diff(a, b):
    """Signed smallest difference a-b in degrees."""
    return (a - b + 180.0) % 360.0 - 180.0
