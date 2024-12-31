import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile
import tempfile
import shutil

class VideoAnalyzer:
    def __init__(self):
        self.features = None
        self.reduced_features = None
        
    def extract_frames(self, video_path, temp_dir):
        """Extract frames from video and save to temporary directory"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        frames_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, gray_frame)
            
            frames_data.append(gray_frame)
            frame_count += 1
            
        cap.release()
        return frame_count, frames_data

    def extract_features(self, frames_data):
        """Extract features from frames"""
        features = []
        
        for frame in frames_data:
            # Calculate intensity features
            mean_intensity = np.mean(frame)
            std_intensity = np.std(frame)
            
            # Calculate edge features
            sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
            edge_intensity = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            features.append([mean_intensity, std_intensity, edge_intensity])
        
        self.features = np.array(features)
        return pd.DataFrame(features, columns=["Mean Intensity", "Std Intensity", "Edge Intensity"])

    def reduce_dimensions(self):
        """Reduce feature dimensions using PCA"""
        if self.features is None:
            raise ValueError("Features must be extracted before dimensionality reduction")
            
        pca = PCA(n_components=2)
        self.reduced_features = pca.fit_transform(self.features)
        return self.reduced_features

    def create_visualization(self):
        """Create PCA visualization"""
        if self.reduced_features is None:
            raise ValueError("Must perform dimensionality reduction first")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            self.reduced_features[:, 0],
            self.reduced_features[:, 1],
            c=np.arange(len(self.reduced_features)),
            cmap='viridis'
        )
        plt.colorbar(scatter, label='Frame Number')
        ax.set_title('Video Frame Analysis')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        
        return fig

def check_file_size(file):
    """Check if file size is within limit (200MB)"""
    MAX_SIZE = 200 * 1024 * 1024  # 200MB in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size <= MAX_SIZE, file_size / (1024 * 1024)  # Return size in MB

def main():
    st.set_page_config(page_title="Video Analysis", layout="wide")
    
    st.title("Video Frame Analysis")
    st.write("""
    This application analyzes video files through the following steps:
    1. Frame extraction from the video
    2. Feature extraction from frames (intensity and edge detection)
    3. Dimensionality reduction and visualization
    
    **Note:** Maximum file size limit is 200MB
    """)
    
    # File uploader with type validation
    video_file = st.file_uploader("Upload your video file", type=['mp4', 'avi', 'mov', 'mkv', 'wmv'])
    
    if video_file is not None:
        # Check file size
        size_ok, file_size = check_file_size(video_file)
        if not size_ok:
            st.error(f"File size ({file_size:.1f}MB) exceeds the 200MB limit. Please upload a smaller file.")
            return
            
        st.info(f"File size: {file_size:.1f}MB")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video to temporary file
            temp_video_path = os.path.join(temp_dir, "input_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getvalue())
            
            # Initialize analyzer
            analyzer = VideoAnalyzer()
            
            # Extract frames
            with st.spinner("Extracting frames from video..."):
                frame_count, frames_data = analyzer.extract_frames(temp_video_path, temp_dir)
                st.success(f"Successfully extracted {frame_count} frames!")
            
            # Create ZIP file of frames
            frames_zip = BytesIO()
            with ZipFile(frames_zip, 'w') as zip_file:
                for frame_file in os.listdir(temp_dir):
                    if frame_file.startswith("frame_"):
                        zip_file.write(
                            os.path.join(temp_dir, frame_file),
                            frame_file
                        )
            st.download_button(
                "Download Extracted Frames",
                frames_zip.getvalue(),
                "frames.zip",
                "application/zip"
            )
            
            # Extract features
            with st.spinner("Extracting features..."):
                features_df = analyzer.extract_features(frames_data)
                st.success("Feature extraction complete!")
            
            # Download features
            csv_buffer = BytesIO()
            features_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Features CSV",
                csv_buffer.getvalue(),
                "features.csv",
                "text/csv"
            )
            
            # Create visualization
            with st.spinner("Creating visualization..."):
                analyzer.reduce_dimensions()
                fig = analyzer.create_visualization()
                st.pyplot(fig)
                
                # Option to download the visualization
                viz_buffer = BytesIO()
                fig.savefig(viz_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "Download Visualization",
                    viz_buffer.getvalue(),
                    "visualization.png",
                    "image/png"
                )

if __name__ == "__main__":
    main()