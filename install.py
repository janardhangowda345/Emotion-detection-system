"""
Installation Script for Real-Time Emotion Detection System
Automatically installs dependencies and sets up the system
"""

import subprocess
import sys
import os
import platform

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    # Read requirements from file
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().splitlines()
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False
    
    # Install each package
    failed_packages = []
    for package in requirements:
        if package.strip() and not package.startswith("#"):
            print(f"Installing {package}...")
            if not install_package(package.strip()):
                failed_packages.append(package)
                print(f"❌ Failed to install {package}")
            else:
                print(f"✓ Installed {package}")
    
    if failed_packages:
        print(f"\n⚠️ Failed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually or check for compatibility issues")
        return False
    
    print("✓ All dependencies installed successfully")
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "models",
        "data",
        "logs"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check platform
    system = platform.system()
    print(f"✓ Operating system: {system}")
    
    # Check for camera and microphone (basic check)
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera detected")
            cap.release()
        else:
            print("⚠️ No camera detected")
    except ImportError:
        print("⚠️ OpenCV not available (will be installed)")
    
    try:
        import sounddevice as sd
        print("✓ Audio system available")
    except ImportError:
        print("⚠️ SoundDevice not available (will be installed)")
    
    return True

def download_sample_models():
    """Download sample models if available"""
    print("Checking for sample models...")
    
    # This would typically download pre-trained models
    # For now, we'll just create placeholder files
    model_files = [
        "models/video_emotion_model.h5",
        "models/audio_emotion_model.h5"
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            try:
                # Create a placeholder file
                with open(model_file, "w") as f:
                    f.write("# Placeholder model file\n")
                    f.write("# Replace with actual trained model\n")
                print(f"✓ Created placeholder: {model_file}")
            except Exception as e:
                print(f"❌ Failed to create {model_file}: {e}")
        else:
            print(f"✓ Model file exists: {model_file}")

def main():
    """Main installation function"""
    print("=" * 60)
    print("Real-Time Emotion Detection System - Installation")
    print("=" * 60)
    
    try:
        # Check system requirements
        if not check_system_requirements():
            print("\n❌ System requirements not met")
            return False
        
        # Create directories
        if not create_directories():
            print("\n❌ Failed to create directories")
            return False
        
        # Install dependencies
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            return False
        
        # Download sample models
        download_sample_models()
        
        print("\n" + "=" * 60)
        print("Installation completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run 'python test_system.py' to verify installation")
        print("2. Run 'python demo.py' to start the demo")
        print("3. Run 'python real_time_emotion_detection.py' to start the system")
        print("\nFor training your own models:")
        print("1. Download FER-2013 and RAVDESS datasets")
        print("2. Update paths in train_models.py")
        print("3. Run 'python train_models.py'")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\nInstallation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
