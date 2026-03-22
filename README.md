# Real-Time Emotion Detection System

A comprehensive Python application that combines computer vision and audio processing to detect emotions in real-time using both facial expressions and speech patterns.

## Features

- **Multi-modal Emotion Detection**: Combines facial expression analysis and audio emotion recognition
- **Real-time Processing**: Low-latency processing for smooth real-time performance
- **Multiple Fusion Strategies**: Supports majority voting, weighted average, and confidence-weighted fusion
- **Modular Architecture**: Well-structured, easily extensible codebase
- **Pre-trained Models**: Uses CNN for facial emotion detection and LSTM/CNN for audio emotion detection
- **Performance Monitoring**: Built-in FPS tracking and performance statistics

## System Requirements

- Python 3.8+
- Webcam
- Microphone
- GPU (recommended for better performance)

## Installation

1. Clone or download the project files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the required datasets (optional, for training):
   - FER-2013 dataset for facial emotion detection
   - RAVDESS dataset for audio emotion detection

## Quick Start

### Using Pre-trained Models

If you have pre-trained models, place them in the `models/` directory:
- `models/video_emotion_model.h5` - Video emotion detection model
- `models/audio_emotion_model.h5` - Audio emotion detection model

Then run:

```bash
python real_time_emotion_detection.py
```

### Training Your Own Models

1. Download the datasets:
   - FER-2013: [Download here](https://www.kaggle.com/datasets/msambare/fer2013)
   - RAVDESS: [Download here](https://zenodo.org/record/1188976)

2. Update the paths in `train_models.py`:
   ```python
   fer2013_path = "path/to/fer2013.csv"
   ravdess_path = "path/to/RAVDESS"
   ```

3. Train the models:
   ```bash
   python train_models.py
   ```

## Usage

### Basic Usage

```python
from real_time_emotion_detection import RealTimeEmotionDetection

# Create emotion detector
detector = RealTimeEmotionDetection(
    camera_index=0,
    fusion_strategy="weighted_average"
)

# Run the system
detector.run()
```

### Advanced Configuration

```python
from config import *

# Modify configuration
FUSION_STRATEGY = "majority_voting"
VIDEO_WEIGHT = 0.7
AUDIO_WEIGHT = 0.3
FPS_TARGET = 30
```

## Architecture

### Core Components

1. **VideoProcessor**: Handles video capture, face detection, and facial emotion prediction
2. **AudioProcessor**: Manages audio capture, feature extraction, and audio emotion prediction
3. **EmotionFusion**: Combines predictions from multiple modalities
4. **ModelLoader**: Loads and manages pre-trained models
5. **RealTimeEmotionDetection**: Main application orchestrating all components

### Data Flow

```
Video Input → Face Detection → CNN Model → Video Emotion
     ↓
Audio Input → Feature Extraction → LSTM/CNN Model → Audio Emotion
     ↓
Fusion Module → Final Emotion Prediction → Display
```

## Configuration

The system can be configured through `config.py`:

- **FPS_TARGET**: Target frames per second (default: 30)
- **AUDIO_CHUNK_DURATION**: Audio processing window (default: 2 seconds)
- **FUSION_STRATEGY**: Emotion fusion method
- **VIDEO_WEIGHT/AUDIO_WEIGHT**: Weights for weighted average fusion
- **EMOTION_LABELS**: Supported emotion categories

## Fusion Strategies

1. **Majority Voting**: Uses the most frequent emotion from recent predictions
2. **Weighted Average**: Combines predictions based on confidence scores
3. **Confidence Weighted**: Uses the prediction with higher confidence
4. **Temporal Fusion**: Considers recent prediction history

## Performance Optimization

- **GPU Acceleration**: Enable GPU for faster model inference
- **Batch Processing**: Process multiple frames simultaneously
- **Model Quantization**: Reduce model size for faster inference
- **Feature Caching**: Cache extracted features to avoid recomputation

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and device index
2. **Audio not working**: Verify microphone permissions and audio drivers
3. **Low FPS**: Reduce video resolution or enable GPU acceleration
4. **Model loading errors**: Check model file paths and compatibility

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## File Structure

```
├── config.py                          # Configuration settings
├── video_processor.py                 # Video processing module
├── audio_processor.py                 # Audio processing module
├── emotion_fusion.py                  # Emotion fusion strategies
├── model_loader.py                    # Model loading and management
├── real_time_emotion_detection.py     # Main application
├── train_models.py                    # Model training script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── models/                            # Pre-trained models directory
    ├── video_emotion_model.h5
    └── audio_emotion_model.h5
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FER-2013 dataset for facial emotion recognition
- RAVDESS dataset for audio emotion recognition
- OpenCV for computer vision
- TensorFlow/Keras for deep learning
- Librosa for audio processing

## Future Improvements

- [ ] Support for multiple faces
- [ ] Real-time model fine-tuning
- [ ] Web interface
- [ ] Mobile app integration
- [ ] Cloud deployment
- [ ] Advanced fusion strategies
- [ ] Emotion intensity estimation
- [ ] Temporal emotion tracking
