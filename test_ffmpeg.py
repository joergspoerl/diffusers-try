#!/usr/bin/env python3
"""Test FFmpeg detection"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_ffmpeg_direct():
    """Test FFmpeg detection without Qt"""
    import shutil
    
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Priority order: Local project FFmpeg first, then system installations
    possible_paths = [
        # Local project FFmpeg (highest priority)
        os.path.join(project_root, "ffmpeg", "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"),
        os.path.join(project_root, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(project_root, "bin", "ffmpeg.exe"),
        os.path.join(project_root, "ffmpeg.exe"),
        
        # System PATH
        "ffmpeg",  # Unix/Linux in PATH
        "ffmpeg.exe",  # Windows in PATH
        
        # Common Windows system installations
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe", 
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        
        # Current working directory (legacy)
        os.path.join(os.getcwd(), "ffmpeg.exe"),
        os.path.join(os.getcwd(), "bin", "ffmpeg.exe"),
    ]
    
    print(f"🔍 Suche FFmpeg im Projekt: {project_root}")
    print("="*60)
    
    for i, path in enumerate(possible_paths, 1):
        print(f"{i:2d}. Prüfe: {path}")
        
        # Check if file exists directly (for absolute paths)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"    ✅ GEFUNDEN! Größe: {size:,} bytes ({size/1024/1024:.1f} MB)")
            return path
        # Check if command exists in PATH (for relative commands)
        elif shutil.which(path):
            found_path = shutil.which(path)
            print(f"    ✅ GEFUNDEN in PATH: {found_path}")
            return found_path
        else:
            print(f"    ❌ Nicht gefunden")
            
    print("\n❌ FFmpeg nicht gefunden!")
    
    # List what's actually in the ffmpeg directory
    ffmpeg_dir = os.path.join(project_root, "ffmpeg")
    
    if os.path.exists(ffmpeg_dir):
        print(f"\n🔍 Inhalt von {ffmpeg_dir}:")
        for item in os.listdir(ffmpeg_dir):
            item_path = os.path.join(ffmpeg_dir, item)
            if os.path.isdir(item_path):
                print(f"  � {item}/")
                # Show subdirectories too
                if item.startswith("ffmpeg"):
                    subdir = os.path.join(ffmpeg_dir, item)
                    for subitem in os.listdir(subdir):
                        subitem_path = os.path.join(subdir, subitem)
                        if os.path.isdir(subitem_path):
                            print(f"    📁 {subitem}/")
                            if subitem == "bin":
                                # Show bin contents
                                bin_dir = os.path.join(subdir, subitem)
                                for binfile in os.listdir(bin_dir):
                                    print(f"      📄 {binfile}")
            else:
                print(f"  📄 {item}")
    else:
        print(f"\n❌ FFmpeg-Verzeichnis nicht gefunden: {ffmpeg_dir}")

if __name__ == "__main__":
    test_ffmpeg_direct()
