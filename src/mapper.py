import numpy as np
import cv2
from typing import Tuple, List, Union


class CoordinateMapper:
    """
    MicroPy-Align 坐标映射引擎
    实现跨模态（荧光 ↔ 电镜）图像的坐标几何变换与载物台（Stage）导航点计算
    """

    def __init__(self, affine_matrix: np.ndarray):
        """
        初始化坐标映射器
        :param affine_matrix: 2x3 仿射变换矩阵 (通常由 ImageAligner 计算得出)
        """
        if affine_matrix.shape != (2, 3):
            raise ValueError("输入的仿射变换矩阵必须是 2x3 维度的 NumPy 数组。")
        self.M = affine_matrix.astype(np.float64)

        # 计算逆矩阵 (用于反向坐标映射)
        # 将 2x3 补充为 3x3 方阵以求解逆矩阵
        M_3x3 = np.vstack([self.M, [0, 0, 1]])
        try:
            M_inv_3x3 = np.linalg.inv(M_3x3)
            self.M_inv = M_inv_3x3[:2, :]
        except np.linalg.LinAlgError:
            raise ValueError("仿射变换矩阵不可逆，无法建立反向坐标映射。")

    def transform_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        正向坐标映射：将源图像（荧光图）点 $(x, y)$ 转换为目标图像（电镜图）点 $(x', y')$
        数学公式: [x', y']^T = M * [x, y, 1]^T
        """
        x, y = point
        src_vector = np.array([x, y, 1.0], dtype=np.float64)
        target_vector = np.dot(self.M, src_vector)
        return float(target_vector[0]), float(target_vector[1])

    def inverse_transform_point(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        反向坐标映射：将目标图像（电镜图）点 $(x', y')$ 还原回源图像（荧光图）点 $(x, y)$
        """
        x_prime, y_prime = point
        tgt_vector = np.array([x_prime, y_prime, 1.0], dtype=np.float64)
        src_vector = np.dot(self.M_inv, tgt_vector)
        return float(src_vector[0]), float(src_vector[1])

    def transform_batch_points(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        批量坐标映射
        :param points: 源图像点列表 [(x1, y1), (x2, y2), ...]
        :return: 转换后的目标图像点列表 [(x1', y1'), (x2', y2'), ...]
        """
        if not points:
            return []
        pts_np = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        transformed_pts = cv2.transform(pts_np, self.M)
        return [
            (float(p[0][0]), float(p[0][1]))
            for p in transformed_pts
        ]

    def draw_mapped_marker(
        self,
        image: np.ndarray,
        target_point: Tuple[float, float],
        label: str = "POI",
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 8,
    ) -> np.ndarray:
        """
        在目标图像（如电镜图）上高亮标出映射到的感兴趣区域（Point of Interest, POI）
        """
        marked_img = image.copy()
        pt_x, pt_y = int(round(target_point[0])), int(round(target_point[1]))

        # 绘制准星交叉线与圆圈标记
        cv2.circle(marked_img, (pt_x, pt_y), radius, color, 2)
        cv2.drawMarker(
            marked_img,
            (pt_x, pt_y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=radius * 2,
            thickness=2,
        )

        # 标注文本坐标
        text = f"{label} ({pt_x}, {pt_y})"
        cv2.putText(
            marked_img,
            text,
            (pt_x + radius + 4, pt_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        return marked_img