"""
Build script for creating executable from yysls-opencv-template
"""
import os
import subprocess
import sys
import json
from datetime import datetime

# Set console encoding to UTF-8 for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def build_exe():
    """Build executable using PyInstaller for yysls-opencv-template"""
    
    # Load version info if available
    version_info = {}
    if os.path.exists("version.txt"):
        with open("version.txt", "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    version_info[key] = value
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Use custom spec file for manifest support
    spec_file = "yysls-opencv-template.spec"
    if not os.path.exists(spec_file):
        print(f"Warning: {spec_file} not found, using default PyInstaller options")
        cmd = [
            "pyinstaller",
            "--onefile",
            "--windowed",
            "--name=燕云十六声 剧情模式QTE助手",
            "--add-data=templates;templates",
            "--add-data=crop_config.json;.",
            "--add-data=app.manifest;.",
            "--hidden-import=cv2",
            "--hidden-import=numpy",
            "--hidden-import=mss",
            "--hidden-import=psutil",
            "--hidden-import=pydirectinput",
            "--hidden-import=win32gui",
            "--hidden-import=win32process",
            "--hidden-import=pystray",
            "--hidden-import=PIL",
            "--hidden-import=PIL.Image",
            "--hidden-import=PIL.ImageDraw",
            "main.py"
        ]
    else:
        cmd = [
            "pyinstaller",
            spec_file
        ]
    
    print("Building executable...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.check_call(cmd)
        print("\nBuild completed successfully!")
        
        # Check if executable was created
        exe_name = "燕云十六声 剧情模式QTE助手.exe"
        exe_path = os.path.join("dist", exe_name)
        
        # List all files in dist directory to debug
        if os.path.exists("dist"):
            dist_files = os.listdir("dist")
            print(f"Files in dist directory: {dist_files}")
            
            # Check if any .exe file exists
            exe_files = [f for f in dist_files if f.endswith('.exe')]
            if exe_files:
                print(f"Found executable files: {exe_files}")
                if exe_name in exe_files:
                    print(f"Executable created: dist/{exe_name}")
                else:
                    print(f"Expected: {exe_name}")
                    print(f"Found: {exe_files[0]}")
                    # Update exe_path to the actual file found
                    exe_path = os.path.join("dist", exe_files[0])
            else:
                print("No executable files found in dist directory")
                sys.exit(1)
        else:
            print("dist directory does not exist")
            sys.exit(1)
        
        # Get version info for output
        version = version_info.get("VERSION", "unknown")
        build_date = version_info.get("BUILD_DATE", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"Version: {version}")
        print(f"Build Date: {build_date}")
        print(f"Executable location: {exe_path}")
        print("\nUsage:")
        print(f"1. Copy {exe_path} to your desired location")
        print("2. Copy crop_config.json to the same folder")
        print("3. Copy templates/ folder to the same location")
        print(f"4. Run {os.path.basename(exe_path)}")
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)
    
    return True

if __name__ == "__main__":
    build_exe()
