import cv2
import numpy as np
from typing import Tuple


class FeatureRegistrator:
    """基于 SIFT / ORB 算法的自动图像配准"""

    def __init__(self, method: str = "SIFT"):
        self.method = method.upper()

    def compute_affine_matrix(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        max_features: int = 5000,
    ) -> np.ndarray:
        src_gray = (
            cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            if len(source_img.shape) == 3
            else source_img
        )
        tgt_gray = (
            cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            if len(target_img.shape) == 3
            else target_img
        )

        detector = (
            cv2.SIFT_create(nfeatures=max_features)
            if self.method == "SIFT"
            else cv2.ORB_create(max_features)
        )

        kp1, des1 = detector.detectAndCompute(src_gray, None)
        kp2, des2 = detector.detectAndCompute(tgt_gray, None)

        if des1 is None or des2 is None:
            raise ValueError("未能提取出足够的特征点。")

        if self.method == "SIFT":
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=50)
            )
            matches = matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            good = sorted(matches, key=lambda x: x.distance)[:50]

        if len(good) < 3:
            raise ValueError("匹配特征点不足，无法计算仿射矩阵。")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )
        tgt_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(
            -1, 1, 2
        )

        matrix, _ = cv2.estimateAffine2D(
            src_pts, tgt_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )
        if matrix is None:
            raise RuntimeError("仿射矩阵估算失败。")

        return matrix