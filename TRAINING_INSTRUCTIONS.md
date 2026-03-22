# Model Training Instructions

## ✅ What Has Been Done

I've updated your `train_models.py` script to:

1. **Display Accuracy**: The script now shows detailed accuracy information including:
   - Training Accuracy (%)
   - Validation Accuracy (%)
   - Training Loss
   - Validation Loss
   - Best epoch accuracies

2. **Synthetic Data Support**: If real datasets (FER-2013, RAVDESS) are not available, the script automatically generates synthetic training data

3. **Better Output Format**: Clear, formatted output showing all training metrics

## 🚀 How to Train Your Models

### Option 1: Quick Training (Recommended for Demo)
```bash
python complete_training.py
```
This will train both models with 10 epochs using synthetic data.

### Option 2: Full Training Script
```bash
python train_models.py
```
This uses the default settings (10 epochs with synthetic data fallback).

## 📊 What You'll See

When training completes, you'll see output like this:

```
============================================================
VIDEO MODEL TRAINING RESULTS
============================================================
Training Accuracy:   85.23%
Training Loss:       0.4521
Validation Accuracy: 82.15%
Validation Loss:     0.5123

Best Training Accuracy:   87.45%
Best Validation Accuracy: 84.32%
============================================================

============================================================
AUDIO MODEL TRAINING RESULTS
============================================================
Training Accuracy:   78.91%
Training Loss:       0.6234
Validation Accuracy: 75.67%
Validation Loss:     0.7123

Best Training Accuracy:   80.12%
Best Validation Accuracy: 77.89%
============================================================
```

## ⏱️ Training Time

- **10 epochs**: ~5-10 minutes (depending on your CPU)
- **20 epochs**: ~10-20 minutes
- **50 epochs**: ~25-50 minutes

## 📝 Notes

1. **Synthetic Data**: Training with synthetic data will show lower accuracy than real datasets. For production use, download:
   - FER-2013 dataset for video model
   - RAVDESS dataset for audio model

2. **Model Locations**: 
   - Video: `models/video_emotion_model.h5`
   - Audio: `models/audio_emotion_model.h5`

3. **Current Status**: Both models exist in your `models/` directory. To retrain with new data or see fresh accuracy metrics, run the training script.

## 🔍 Check Existing Models

To check your current models:
```bash
python check_model_accuracy.py
```

## 💡 Tips

- Let the training complete - don't interrupt it
- For better accuracy, use real datasets instead of synthetic data
- More epochs = better accuracy (but longer training time)
- The script will automatically save models when training completes

