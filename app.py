import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.aligner import ImageAligner
from src.mapper import CoordinateMapper

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
reg_method = st.sidebar.selectbox("Registration Method", ["SIFT", "ORB", "Manual Points"])

if src_file and tgt_file:
    # 1. 加载并转换图像
    src_pil = Image.open(src_file).convert("RGB")
    tgt_pil = Image.open(tgt_file).convert("RGB")

    src_img = np.array(src_pil)
    tgt_img = np.array(tgt_pil)

    # 显示原始图像 (已修改参数为 use_container_width)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fluorescence View")
        st.image(src_img, use_container_width=True)
    with col2:
        st.subheader("Electron Microscopy View")
        st.image(tgt_img, use_container_width=True)

    st.divider()
    st.header("⚡ Image Registration & Alignment")

    aligner = ImageAligner(method=reg_method)
    aligned_img = None
    matrix = None

    if reg_method in ["SIFT", "ORB"]:
        try:
            with st.spinner("Calculating affine transformation matrix..."):
                aligned_img, matrix = aligner.align_automatic(src_img, tgt_img)
            st.success("✅ Automatic Affine Registration Successful!")
        except Exception as e:
            st.error(f"❌ Registration Failed: {str(e)}")
            st.info("Tip: Try switching to 'Manual Points' mode if cross-modal contrast difference is too high.")
    else:
        st.subheader("Manual Control Points (Fiducial Markers)")
        st.info("Provide at least 3 pairs of matching coordinates (x, y) between Fluorescence and EM images.")

        # 默认示例 3 点
        col_pts1, col_pts2 = st.columns(2)
        with col_pts1:
            p1_src = st.text_input("Fluorescence Points (x,y)", "50,50; 200,50; 100,200")
        with col_pts2:
            p1_tgt = st.text_input("EM Points (x',y')", "60,55; 210,52; 108,205")

        if st.button("Run Manual Registration"):
            try:
                src_pts = [tuple(map(float, p.split(','))) for p in p1_src.split(';')]
                tgt_pts = [tuple(map(float, p.split(','))) for p in p1_tgt.split(';')]
                aligned_img, matrix = aligner.align_from_points(src_img, src_pts, tgt_pts, tgt_img.shape)
                st.success("✅ Manual Keypoint Registration Successful!")
            except Exception as e:
                st.error(f"Error parsing points: {str(e)}")

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
            marked_em = mapper.draw_mapped_marker(tgt_img, (target_x, target_y), label="POI")
            st.image(marked_em, use_container_width=True)

        st.divider()
        st.header("🎛️ Dual-Channel Fusion View (Alpha Overlay)")

        alpha = st.slider("Fluorescence Channel Opacity (Alpha)", 0.0, 1.0, 0.5, 0.05)

        # 融合图计算
        fusion_img = cv2.addWeighted(aligned_img, alpha, tgt_img, 1 - alpha, 0)
        st.image(fusion_img, caption=f"Overlay Fusion (Alpha: {alpha:.2f})", use_container_width=True)

else:
    st.info("👈 Please upload both a Fluorescence and an EM image from the sidebar to start alignment.")