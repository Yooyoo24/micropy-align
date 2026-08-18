import cv2
import numpy as np
from typing import Tuple


class AffineTransformer:
    """处理图像的空间几何变换与 Warp 操作"""

    @staticmethod
    def apply_warp(
        image: np.ndarray, matrix: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """根据 2x3 仿射变换矩阵对图像进行空间变换"""
        height, width = target_shape[:2]
        return cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR
        )