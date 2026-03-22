# Model Training Results

## Training Script Features

The updated `train_models.py` script now includes:

1. **Accuracy Reporting**: Displays training and validation accuracy after each model training
2. **Synthetic Data Generation**: Automatically generates synthetic training data if real datasets are not available
3. **Detailed Output**: Shows:
   - Training samples count
   - Validation samples count
   - Training accuracy (%)
   - Validation accuracy (%)
   - Training loss
   - Validation loss
   - Best epoch accuracies

## How to Train

Run the training script:
```bash
python train_models.py
```

Or use the quick training script (5 epochs for fast demo):
```bash
python quick_train.py
```

## Expected Output Format

When training completes, you will see output like:

```
============================================================
VIDEO MODEL TRAINING RESULTS
============================================================
Training Accuracy:   XX.XX%
Training Loss:       X.XXXX
Validation Accuracy: XX.XX%
Validation Loss:     X.XXXX

Best Training Accuracy:   XX.XX%
Best Validation Accuracy: XX.XX%
============================================================

============================================================
AUDIO MODEL TRAINING RESULTS
============================================================
Training Accuracy:   XX.XX%
Training Loss:       X.XXXX
Validation Accuracy: XX.XX%
Validation Loss:     X.XXXX

Best Training Accuracy:   XX.XX%
Best Validation Accuracy: XX.XX%
============================================================
```

## Model Locations

- Video Model: `models/video_emotion_model.h5`
- Audio Model: `models/audio_emotion_model.h5`

## Notes

- Training with synthetic data will have lower accuracy than real datasets
- For production use, download FER-2013 and RAVDESS datasets
- Recommended epochs: 20-50 for synthetic data, 50-100 for real datasets
- Training time depends on your hardware (CPU/GPU)



