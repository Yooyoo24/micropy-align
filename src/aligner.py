import cv2
import numpy as np
from typing import Tuple, Optional, List


class ImageAligner:
    """
    MicroPy-Align 图像配准核心引擎
    支持：
    1. 基于 SIFT + RANSAC 的自动仿射变换配准 (Automatic Affine Registration)
    2. 基于自定义标记点对 (Fiducial Points) 的精准坐标对齐 (Manual Keypoint Matching)
    """

    def __init__(self, method: str = "SIFT"):
        self.method = method.upper()

    def align_automatic(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        max_features: int = 5000,
        good_match_percent: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        利用 SIFT/ORB 自动进行跨模态图像对齐
        :param source_img: 待配准图像 (如荧光图，RGB 或灰度)
        :param target_img: 基准图像 (如电镜图，灰度)
        :return: (配准后对齐的图像, 2x3 仿射变换矩阵 Matrix)
        """
        # 转为灰度图进行特征检测
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

        # 1. 提取特征点与描述子
        if self.method == "SIFT":
            detector = cv2.SIFT_create(nfeatures=max_features)
        else:
            detector = cv2.ORB_create(max_features)

        keypoints1, descriptors1 = detector.detectAndCompute(src_gray, None)
        keypoints2, descriptors2 = detector.detectAndCompute(tgt_gray, None)

        if descriptors1 is None or descriptors2 is None:
            raise ValueError("未能在图像中检测到足够的特征点。")

        # 2. 特征匹配 (FLANN 或 BFMatcher)
        if self.method == "SIFT":
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=50)
            )
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            # Lowe's ratio test 筛选优质匹配点
            good_matches = [
                m for m, n in matches if m.distance < 0.7 * n.distance
            ]
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(descriptors1, descriptors2)
            matches.sort(key=lambda x: x.distance, reverse=False)
            num_good_matches = int(len(matches) * good_match_percent)
            good_matches = matches[:num_good_matches]

        if len(good_matches) < 3:
            raise ValueError(
                f"配准失败：有效匹配特征点不足（当前仅找到 {len(good_matches)} 个，至少需要 3 个）。"
            )

        # 3. 提取特征点坐标
        src_pts = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        tgt_pts = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        # 4. 使用 RANSAC 计算仿射变换矩阵 (2x3 Affine Matrix)
        matrix, inliers = cv2.estimateAffine2D(
            src_pts, tgt_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if matrix is None:
            raise RuntimeError("无法估算有效的仿射变换矩阵。")

        # 5. 对源图像进行空间几何变换 Warp
        height, width = target_img.shape[:2]
        aligned_img = cv2.warpAffine(
            source_img, matrix, (width, height), flags=cv2.INTER_LINEAR
        )

        return aligned_img, matrix

    @staticmethod
    def align_from_points(
        source_img: np.ndarray,
        src_points: List[Tuple[float, float]],
        tgt_points: List[Tuple[float, float]],
        target_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于手动选择的标记点对 (Fiducials) 计算仿射矩阵并对齐
        :param source_img: 待配准图像
        :param src_points: 源图像上的控制点 [(x1,y1), (x2,y2), ...]
        :param tgt_points: 目标图像上的对应控制点 [(x1',y1'), (x2',y2'), ...]
        :param target_shape: 目标图像尺寸 (height, width)
        :return: (配准后图像, 2x3 仿射矩阵)
        """
        if len(src_points) < 3 or len(tgt_points) < 3:
            raise ValueError("计算仿射变换至少需要 3 对对应标记点。")

        pts1 = np.float32(src_points)
        pts2 = np.float32(tgt_points)

        # 如果正好 3 个点用 getAffineTransform，多于 3 个点用 estimateAffine2D (最小二乘优化)
        if len(src_points) == 3:
            matrix = cv2.getAffineTransform(pts1, pts2)
        else:
            matrix, _ = cv2.estimateAffine2D(pts1, pts2)

        height, width = target_shape[:2]
        aligned_img = cv2.warpAffine(
            source_img, matrix, (width, height), flags=cv2.INTER_LINEAR
        )

        return aligned_img, matrix