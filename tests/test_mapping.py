import numpy as np
import pytest
from src.mapping.coordinate import CoordinateMapper


def test_coordinate_transformation():
    # 构造一个简单的平移矩阵 (X+10, Y+20)
    matrix = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]])

    mapper = CoordinateMapper(matrix)

    # 正向变换测试
    src_pt = (50.0, 50.0)
    tgt_pt = mapper.transform_point(src_pt)
    assert tgt_pt == (60.0, 70.0)

    # 逆向变换测试
    recovered_pt = mapper.inverse_transform_point(tgt_pt)
    assert pytest.approx(recovered_pt[0]) == 50.0
    assert pytest.approx(recovered_pt[1]) == 50.0