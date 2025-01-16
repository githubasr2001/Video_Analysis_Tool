import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from zipfile import ZipFile
import tempfile
import shutil
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def check_file_size(file):
    """Check if file size is within limit (200MB)"""
    MAX_SIZE = 200 * 1024 * 1024  # 200MB in bytes
    
    if hasattr(file, 'size'):
        file_size = file.size
    else:
        # Fallback method if size attribute is not available
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
    return file_size <= MAX_SIZE, file_size / (1024 * 1024)  # Return size in MB

class VideoAnalyzer:
    # [Previous VideoAnalyzer class code remains exactly the same]
    def __init__(self):
        self.features = None
        self.reduced_features = None
        self.frames_data = []
        self.motion_data = []
        self.scene_changes = []
        self.color_histograms = []
        
    def extract_frames(self, video_path, temp_dir):
        """Extract colored frames from video and save to temporary directory"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        prev_frame = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Keep original color frame for saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save colored frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            # Calculate motion if we have a previous frame
            if prev_frame is not None:
                motion = self.calculate_motion(prev_frame, frame)
                self.motion_data.append(motion)
            
            # Calculate color histogram
            hist = self.calculate_color_histogram(frame_rgb)
            self.color_histograms.append(hist)
            
            # Detect scene changes
            if frame_count > 0:
                diff = np.mean(np.abs(frame_rgb - prev_frame))
                if diff > 30:  # Threshold for scene change
                    self.scene_changes.append(frame_count)
            
            self.frames_data.append(frame_rgb)
            prev_frame = frame_rgb
            frame_count += 1
            
        cap.release()
        return frame_count, fps

    def calculate_motion(self, prev_frame, curr_frame):
        """Calculate motion between consecutive frames"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)

    def calculate_color_histogram(self, frame):
        """Calculate color histogram for RGB channels"""
        hist_r = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([frame], [2], None, [256], [0, 256])
        return np.concatenate([hist_r, hist_g, hist_b]).flatten()

    def extract_features(self):
        """Extract comprehensive features from frames"""
        features = []
        
        for frame in self.frames_data:
            # Basic statistics
            mean_rgb = np.mean(frame, axis=(0, 1))
            std_rgb = np.std(frame, axis=(0, 1))
            
            # Edge features
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_intensity = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            # Texture features using GLCM
            gray = np.uint8(gray)
            glcm = self.calculate_glcm(gray)
            contrast = np.mean(glcm * np.square(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])))
            
            features.append([
                *mean_rgb,  # RGB means
                *std_rgb,   # RGB standard deviations
                edge_intensity,
                contrast
            ])
        
        self.features = np.array(features)
        columns = [
            'Mean R', 'Mean G', 'Mean B',
            'Std R', 'Std G', 'Std B',
            'Edge Intensity', 'Contrast'
        ]
        return pd.DataFrame(features, columns=columns)

    def calculate_glcm(self, image, distance=1, angles=[0]):
        """Calculate Gray Level Co-occurrence Matrix"""
        glcm = np.zeros((256, 256))
        for angle in angles:
            dx = int(distance * np.cos(angle))
            dy = int(distance * np.sin(angle))
            for i in range(image.shape[0] - dx):
                for j in range(image.shape[1] - dy):
                    glcm[image[i, j], image[i + dx, j + dy]] += 1
        return glcm / glcm.sum()

    def cluster_scenes(self, n_clusters=5):
        """Cluster similar frames using K-means"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(self.features)

    def create_analysis_dashboard(self, fps):
        """Create comprehensive analysis visualizations"""
        # 1. PCA visualization
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.features)
        
        # Create subplot figure
        fig = plt.figure(figsize=(20, 10))
        
        # PCA plot
        ax1 = plt.subplot(231)
        scatter = ax1.scatter(reduced_features[:, 0], reduced_features[:, 1],
                            c=np.arange(len(reduced_features)), cmap='viridis')
        ax1.set_title('Frame PCA Analysis')
        plt.colorbar(scatter, ax=ax1, label='Frame Number')
        
        # Motion analysis
        ax2 = plt.subplot(232)
        ax2.plot(self.motion_data)
        ax2.set_title('Motion Analysis')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Motion Magnitude')
        
        # Scene changes
        ax3 = plt.subplot(233)
        ax3.vlines(self.scene_changes, 0, 1, color='r', label='Scene Changes')
        ax3.set_title('Scene Changes')
        ax3.set_xlabel('Frame Number')
        ax3.legend()
        
        # Color distribution
        ax4 = plt.subplot(234)
        color_means = np.array([frame.mean(axis=(0, 1)) for frame in self.frames_data])
        ax4.plot(color_means[:, 0], 'r', label='Red')
        ax4.plot(color_means[:, 1], 'g', label='Green')
        ax4.plot(color_means[:, 2], 'b', label='Blue')
        ax4.set_title('Color Distribution Over Time')
        ax4.legend()
        
        # Cluster analysis
        ax5 = plt.subplot(235)
        clusters = self.cluster_scenes()
        scatter = ax5.scatter(reduced_features[:, 0], reduced_features[:, 1],
                            c=clusters, cmap='tab10')
        ax5.set_title('Scene Clustering')
        plt.colorbar(scatter, ax=ax5, label='Cluster')
        
        # Edge intensity
        ax6 = plt.subplot(236)
        ax6.plot(self.features[:, 6])
        ax6.set_title('Edge Intensity Over Time')
        ax6.set_xlabel('Frame Number')
        ax6.set_ylabel('Edge Intensity')
        
        plt.tight_layout()
        return fig

def main():
    st.set_page_config(page_title="Advanced Video Analysis", layout="wide")
    
    # Sidebar for settings
    st.sidebar.title("Analysis Settings")
    st.sidebar.write("Configure your analysis parameters")
    
    scene_threshold = st.sidebar.slider("Scene Change Sensitivity", 10, 50, 30)
    cluster_count = st.sidebar.slider("Number of Scene Clusters", 2, 10, 5)
    
    # Main content
    st.title("🎥 Advanced Video Frame Analysis")
    
    # Description in an expandable section
    with st.expander("ℹ️ About this tool"):
        st.write("""
        This advanced video analysis tool provides:
        - Frame extraction in full color
        - Motion detection and analysis
        - Scene change detection
        - Color distribution analysis
        - Texture and edge analysis
        - Scene clustering
        - Comprehensive visualization dashboard
        
        **Note:** Maximum file size limit is 200MB
        """)
    
    # File upload section
    st.header("📤 Upload Video")
    video_file = st.file_uploader("Upload your video file", type=['mp4', 'avi', 'mov', 'mkv', 'wmv'])
    
    if video_file is not None:
        # Check file size
        size_ok, file_size = check_file_size(video_file)
        if not size_ok:
            st.error(f"⚠️ File size ({file_size:.1f}MB) exceeds the 200MB limit. Please upload a smaller file.")
            return
            
        st.info(f"📁 File size: {file_size:.1f}MB")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            temp_video_path = os.path.join(temp_dir, "input_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getvalue())
            
            # Initialize analyzer
            analyzer = VideoAnalyzer()
            
            # Process video with progress tracking
            with st.spinner("🎬 Extracting frames from video..."):
                frame_count, fps = analyzer.extract_frames(temp_video_path, temp_dir)
                st.success(f"✅ Successfully extracted {frame_count} frames at {fps:.1f} FPS!")
            
            # Create downloads section
            st.header("📥 Downloads")
            col1, col2 = st.columns(2)
            
            # Frames download
            with col1:
                frames_zip = BytesIO()
                with ZipFile(frames_zip, 'w') as zip_file:
                    for frame_file in os.listdir(temp_dir):
                        if frame_file.startswith("frame_"):
                            zip_file.write(os.path.join(temp_dir, frame_file), frame_file)
                st.download_button(
                    "📦 Download Extracted Frames",
                    frames_zip.getvalue(),
                    "frames.zip",
                    "application/zip"
                )
            
            # Feature extraction and download
            with col2:
                features_df = analyzer.extract_features()
                csv_buffer = BytesIO()
                features_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📊 Download Features CSV",
                    csv_buffer.getvalue(),
                    "features.csv",
                    "text/csv"
                )
            
            # Analysis section
            st.header("📈 Video Analysis Dashboard")
            
            # Create and display visualization dashboard
            with st.spinner("🔄 Generating analysis dashboard..."):
                fig = analyzer.create_analysis_dashboard(fps)
                st.pyplot(fig)
                
                # Download visualization
                viz_buffer = BytesIO()
                fig.savefig(viz_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "💾 Download Complete Analysis",
                    viz_buffer.getvalue(),
                    "video_analysis.png",
                    "image/png"
                )
            
            # Summary statistics
            st.header("📋 Analysis Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Frames", frame_count)
            with col2:
                st.metric("Scene Changes", len(analyzer.scene_changes))
            with col3:
                st.metric("Average Motion", f"{np.mean(analyzer.motion_data):.2f}")

if __name__ == "__main__":
    main()
