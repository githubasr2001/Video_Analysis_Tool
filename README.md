# Video Frame Analysis Application

A Streamlit-based web application for analyzing video files through frame extraction, feature analysis, and dimensionality reduction visualization.

## Features

- Frame extraction from various video formats (MP4, AVI, MOV, MKV, WMV)
- Automatic conversion of frames to grayscale
- Feature extraction including:
  - Mean intensity
  - Standard deviation of intensity
  - Edge detection intensity
- PCA-based dimensionality reduction
- Interactive visualization of frame analysis
- Downloadable outputs:
  - Extracted frames (ZIP)
  - Feature data (CSV)
  - Analysis visualization (PNG)

## Requirements

```
streamlit
opencv-python
numpy
pandas
scikit-learn
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-analysis-app.git
cd video-analysis-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload a video file through the web interface
   - Supported formats: MP4, AVI, MOV, MKV, WMV
   - Current size limit: 200MB

4. Wait for the analysis to complete

5. Download the results:
   - Extracted frames as ZIP
   - Feature data as CSV
   - Visualization as PNG

## Technical Details

### Video Processing
- Frames are extracted sequentially from the video
- Each frame is converted to grayscale for consistent analysis
- Temporary files are automatically cleaned up after processing

### Feature Extraction
The application extracts the following features from each frame:
- Mean Intensity: Average brightness of the frame
- Standard Deviation of Intensity: Measure of contrast
- Edge Intensity: Average intensity of detected edges using Sobel operators

### Visualization
- Uses Principal Component Analysis (PCA) to reduce feature dimensions to 2D
- Frames are plotted as points with color indicating frame sequence
- Interactive plot allows zooming and hovering

## Limitations

- Maximum file size: 200MB
- Memory usage scales with video length and resolution
- Processing time depends on video length and complexity

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- OpenCV for video processing
- Streamlit for the web interface
- scikit-learn for PCA implementation

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
