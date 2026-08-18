import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 使用重构后的新子模块路径
from src.registration.automatic import FeatureRegistrator
from src.mapping.coordinate import CoordinateMapper
from src.transform.affine import AffineTransformer

# 页面基础配置
st.set_page_config(
    page_title="MicroPy-Align | CLEM Image Registration",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 MicroPy-Align: Correlative Light & Electron Microscopy Registration")
st.markdown("""
*Automatic Image Registration, Cross-Modal Coordinate Mapping, and Fusion Visualization for CLEM Workflows.*
""")

# 侧边栏：上传与参数设置
st.sidebar.header("1. Upload Images")
src_file = st.sidebar.file_uploader("Fluorescence Image (Source)", type=["png", "jpg", "jpeg", "tif", "tiff"])
tgt_file = st.sidebar.file_uploader("Electron Microscopy Image (Target)", type=["png", "jpg", "jpeg", "tif", "tiff"])

st.sidebar.header("2. Registration Parameters")
reg_method = st.sidebar.selectbox("Registration Method", ["SIFT", "ORB"])

if src_file and tgt_file:
    # 1. 加载并转换图像
    src_pil = Image.open(src_file).convert("RGB")
    tgt_pil = Image.open(tgt_file).convert("RGB")

    src_img = np.array(src_pil)
    tgt_img = np.array(tgt_pil)

    # 显示原始图像
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fluorescence View")
        st.image(src_img, width="stretch")
    with col2:
        st.subheader("Electron Microscopy View")
        st.image(tgt_img, width="stretch")

    st.divider()
    st.header("⚡ Image Registration & Alignment")

    registrator = FeatureRegistrator(method=reg_method)
    aligned_img = None
    matrix = None

    try:
        with st.spinner("Calculating affine transformation matrix..."):
            # 1. 计算 2x3 仿射变换矩阵
            matrix = registrator.compute_affine_matrix(src_img, tgt_img)
            # 2. 对源图像施加 Warp 几何变换
            aligned_img = AffineTransformer.apply_warp(src_img, matrix, tgt_img.shape)
        st.success("✅ Automatic Affine Registration Successful!")
    except Exception as e:
        st.error(f"❌ Registration Failed: {str(e)}")

    # 2. 坐标映射与融合预览
    if aligned_img is not None and matrix is not None:
        mapper = CoordinateMapper(matrix)

        st.divider()
        st.header("🎯 Coordinate Mapping & Stage Navigation")

        col_coord, col_res = st.columns([1, 2])

        with col_coord:
            st.subheader("Input Fluorescence Coordinates")
            x_in = st.number_input("Fluorescence X (px)", min_value=0, max_value=src_img.shape[1],
                                   value=int(src_img.shape[1] / 2))
            y_in = st.number_input("Fluorescence Y (px)", min_value=0, max_value=src_img.shape[0],
                                   value=int(src_img.shape[0] / 2))

            # 计算目标坐标
            target_x, target_y = mapper.transform_point((x_in, y_in))

            st.metric(label="Mapped EM Target X (px)", value=f"{target_x:.2f}")
            st.metric(label="Mapped EM Target Y (px)", value=f"{target_y:.2f}")

        with col_res:
            st.subheader("EM Target Location Overlay")
            marked_em = tgt_img.copy()
            pt_x, pt_y = int(round(target_x)), int(round(target_y))
            cv2.circle(marked_em, (pt_x, pt_y), 10, (0, 0, 255), 2)
            cv2.drawMarker(marked_em, (pt_x, pt_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20,
                           thickness=2)
            st.image(marked_em, width="stretch")

        st.divider()
        st.header("🎛️ Dual-Channel Fusion View (Alpha Overlay)")

        alpha = st.slider("Fluorescence Channel Opacity (Alpha)", 0.0, 1.0, 0.5, 0.05)

        # 融合图计算
        fusion_img = cv2.addWeighted(aligned_img, alpha, tgt_img, 1 - alpha, 0)
        st.image(fusion_img, caption=f"Overlay Fusion (Alpha: {alpha:.2f})", width="stretch")

else:
    st.info("👈 Please upload both a Fluorescence and an EM image from the sidebar to start alignment.")