import numpy as np
from typing import Tuple, List


class CoordinateMapper:
    """基于仿射矩阵的跨模态 2D 坐标系映射"""

    def __init__(self, affine_matrix: np.ndarray):
        if affine_matrix.shape != (2, 3):
            raise ValueError("仿射矩阵形状必须为 (2, 3)")
        self.M = affine_matrix.astype(np.float64)

        # 计算逆矩阵
        M_3x3 = np.vstack([self.M, [0, 0, 1]])
        M_inv_3x3 = np.linalg.inv(M_3x3)
        self.M_inv = M_inv_3x3[:2, :]

    def transform_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """正向映射: (x, y) -> (x', y')"""
        src_vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
        tgt_vec = np.dot(self.M, src_vec)
        return float(tgt_vec[0]), float(tgt_vec[1])

    def inverse_transform_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """反向映射: (x', y') -> (x, y)"""
        tgt_vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
        src_vec = np.dot(self.M_inv, tgt_vec)
        return float(src_vec[0]), float(src_vec[1])