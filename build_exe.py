"""
Build script for creating executable from yysls-opencv-template
"""
import os
import subprocess
import sys

def build_exe():
    """Build executable using PyInstaller for yysls-opencv-template"""
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Use custom spec file for manifest support
    cmd = [
        "pyinstaller",
        "yysls-opencv-template.spec"  # Use custom spec file with manifest
    ]
    
    print("Building executable...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ Build completed successfully!")
        print("📁 Executable location: dist/yysls-opencv-template-v2.exe")
        print("\n📋 Usage:")
        print("1. Copy dist/yysls-opencv-template-v2.exe to your desired location")
        print("2. Copy crop_config.json to the same folder")
        print("3. Copy templates/ folder to the same location")
        print("4. Run yysls-opencv-template-v2.exe")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    build_exe()
