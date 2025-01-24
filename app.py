import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tempfile
from io import BytesIO
from zipfile import ZipFile

def check_file_size(file):
    """Check if file size is within limit (800MB)"""
    MAX_SIZE = 800 * 1024 * 1024  # 800MB in bytes
    
    if hasattr(file, 'size'):
        file_size = file.size
    else:
        # Fallback method if size attribute is not available
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
    return file_size <= MAX_SIZE, file_size / (1024 * 1024)  # Return size in MB

class VideoAnalyzer:
    def __init__(self, sample_rate=5):
        """
        Initialize VideoAnalyzer with optional frame sampling
        
        :param sample_rate: Extract every nth frame to reduce processing time
        """
        self.sample_rate = sample_rate
        self.features = None
        self.frames_data = []
        self.motion_data = []
        self.scene_changes = []
        self.frame_paths = []  # Store frame file paths
        
    def extract_frames(self, video_path, temp_dir):
        """
        Extract frames from video more efficiently
        
        :param video_path: Path to the video file
        :param temp_dir: Temporary directory to save frames
        :return: Tuple of frame count and fps
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames to reduce processing time
            if frame_count % self.sample_rate == 0:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame
                frame_filename = f"frame_{saved_frame_count:04d}.jpg"
                frame_path = os.path.join(temp_dir, frame_filename)
                cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                
                # Store frame path for ZIP creation
                self.frame_paths.append(frame_path)
                
                # Calculate motion if possible
                if prev_frame is not None:
                    motion = self.calculate_motion(prev_frame, frame_rgb)
                    self.motion_data.append(motion)
                
                # Detect scene changes
                if saved_frame_count > 0:
                    diff = np.mean(np.abs(frame_rgb.astype(np.float32) - prev_frame.astype(np.float32)))
                    if diff > 30:  # Threshold for scene change
                        self.scene_changes.append(saved_frame_count)
                
                self.frames_data.append(frame_rgb)
                prev_frame = frame_rgb
                saved_frame_count += 1
            
            frame_count += 1
        
        cap.release()
        return saved_frame_count, fps
    
    def create_frames_zip(self):
        """
        Create ZIP file of extracted frames
        
        :return: BytesIO object containing ZIP file
        """
        frames_zip = BytesIO()
        with ZipFile(frames_zip, 'w') as zip_file:
            for frame_path in self.frame_paths:
                zip_file.write(frame_path, os.path.basename(frame_path))
        frames_zip.seek(0)
        return frames_zip

    def calculate_motion(self, prev_frame, curr_frame):
        """Calculate motion between consecutive frames"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        # Use more efficient optical flow calculation
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)

    def extract_features(self):
        """Extract comprehensive features from frames with reduced computation"""
        features = []
        
        for frame in self.frames_data:
            # Basic color statistics
            mean_rgb = np.mean(frame, axis=(0, 1))
            std_rgb = np.std(frame, axis=(0, 1))
            
            # Simplified edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_intensity = np.mean(edges)
            
            features.append([
                *mean_rgb,  # RGB means
                *std_rgb,   # RGB standard deviations
                edge_intensity
            ])
        
        self.features = np.array(features)
        columns = [
            'Mean R', 'Mean G', 'Mean B',
            'Std R', 'Std G', 'Std B',
            'Edge Intensity'
        ]
        return pd.DataFrame(features, columns=columns)

    def create_analysis_dashboard(self, fps):
        """Create visualization dashboard with reduced complexity"""
        plt.figure(figsize=(15, 10))
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.features)
        
        # Plotting
        plt.subplot(221)
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                    c=np.arange(len(reduced_features)), cmap='viridis')
        plt.title('Frame PCA Analysis')
        
        plt.subplot(222)
        plt.plot(self.motion_data)
        plt.title('Motion Analysis')
        
        plt.subplot(223)
        plt.vlines(self.scene_changes, 0, 1, color='r')
        plt.title('Scene Changes')
        
        plt.subplot(224)
        color_means = np.array([frame.mean(axis=(0, 1)) for frame in self.frames_data])
        plt.plot(color_means[:, 0], 'r', label='Red')
        plt.plot(color_means[:, 1], 'g', label='Green')
        plt.plot(color_means[:, 2], 'b', label='Blue')
        plt.title('Color Distribution')
        plt.legend()
        
        plt.tight_layout()
        return plt

def main():
    st.set_page_config(page_title="Video Frame Analyzer", layout="wide")
    
    # Sidebar for optimization settings
    st.sidebar.title("Analysis Settings")
    sample_rate = st.sidebar.slider("Frame Sampling Rate", 1, 30, 5, 
                                    help="Extract every nth frame to reduce processing time")
    
    # Main content
    st.title("🎥 Efficient Video Frame Analyzer")
    
    # File upload
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        # Check file size
        size_ok, file_size = check_file_size(video_file)
        if not size_ok:
            st.error(f"File size ({file_size:.1f}MB) exceeds 800MB limit")
            return
        
        # Temporary processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save video
            temp_video_path = os.path.join(temp_dir, "input_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getvalue())
            
            # Initialize analyzer with sampling
            analyzer = VideoAnalyzer(sample_rate=sample_rate)
            
            # Process video
            with st.spinner("Extracting frames..."):
                frame_count, fps = analyzer.extract_frames(temp_video_path, temp_dir)
                st.success(f"Extracted {frame_count} frames")
            
            # Downloads section
            st.header("Downloads")
            col1, col2 = st.columns(2)
            
            # Frame ZIP download
            with col1:
                frames_zip = analyzer.create_frames_zip()
                st.download_button(
                    "Download Frames (ZIP)",
                    frames_zip.getvalue(),
                    "video_frames.zip",
                    "application/zip"
                )
            
            # Feature extraction
            with st.spinner("Extracting features..."):
                features_df = analyzer.extract_features()
                
                # Download features
                with col2:
                    csv_buffer = BytesIO()
                    features_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "Download Features (CSV)",
                        csv_buffer.getvalue(),
                        "video_features.csv",
                        "text/csv"
                    )
            
            # Visualization
            with st.spinner("Creating analysis dashboard..."):
                fig = analyzer.create_analysis_dashboard(fps)
                st.pyplot(fig)
                
                # Download visualization
                viz_buffer = BytesIO()
                fig.savefig(viz_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "Download Analysis Plot",
                    viz_buffer.getvalue(),
                    "video_analysis.png",
                    "image/png"
                )
            
            # Summary statistics
            st.header("Analysis Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Frames", frame_count)
            with col2:
                st.metric("Scene Changes", len(analyzer.scene_changes))

if __name__ == "__main__":
    main()
