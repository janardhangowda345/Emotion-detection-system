"""
Complete training script that trains both models and shows accuracy
"""
import sys
import os

# Ensure output is visible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

print("="*70, flush=True)
print("COMPLETE MODEL TRAINING", flush=True)
print("="*70, flush=True)
print("Training both video and audio models with synthetic data", flush=True)
print("This will take a few minutes...", flush=True)
print("="*70 + "\n", flush=True)

try:
    from train_models import ModelTrainer
    from config import VIDEO_MODEL_PATH, AUDIO_MODEL_PATH
    
    trainer = ModelTrainer()
    
    # Train both models with 10 epochs (good balance of speed and accuracy)
    print("Starting training process...\n", flush=True)
    
    trainer.train_both_models(
        fer2013_path=None,  # Will use synthetic
        ravdess_path=None,  # Will use synthetic
        use_synthetic=True,
        epochs=30  # More epochs for 85%+ accuracy
    )
    
    print("\n" + "="*70, flush=True)
    print("✓ TRAINING SUCCESSFULLY COMPLETED!", flush=True)
    print("="*70, flush=True)
    print(f"\nModels saved to:", flush=True)
    print(f"  📹 Video Model: {VIDEO_MODEL_PATH}", flush=True)
    print(f"  🎤 Audio Model: {AUDIO_MODEL_PATH}", flush=True)
    print("\n" + "="*70, flush=True)
    print("You can now use these models in your ed2.py application!", flush=True)
    print("="*70 + "\n", flush=True)
    
except KeyboardInterrupt:
    print("\n\n⚠ Training interrupted by user.", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"\n❌ ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

