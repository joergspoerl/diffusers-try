#!/usr/bin/env python3
"""
Ultra-Fast Video Exporter
=========================

Optimized for speed - creates videos in seconds, not minutes!
Uses aggressive optimization and GPU acceleration when available.

Usage:
    python fast_exporter.py input_folder output.mp4 [--fps 30] [--quality medium]
"""

import argparse
import os
import sys
import glob
import subprocess
import tempfile
import shutil
from pathlib import Path
import time

def find_ffmpeg():
    """Find FFmpeg executable"""
    # Check local project folder first
    local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg", "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    
    # Check system PATH
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return 'ffmpeg'
    except:
        pass
    
    return None

def check_gpu_support(ffmpeg_path):
    """Quick GPU support check"""
    try:
        result = subprocess.run([ffmpeg_path, "-encoders"], capture_output=True, text=True, timeout=5)
        return "h264_nvenc" in result.stdout
    except:
        return False

def create_fast_video(input_folder, output_path, fps=30, quality="medium"):
    """Create video with maximum speed optimization"""
    
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        print("❌ Error: FFmpeg not found!")
        return False
    
    # Find image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    image_files.sort()
    
    if not image_files:
        print(f"❌ Error: No images found in {input_folder}")
        return False
    
    print(f"🚀 FAST MODE: Creating video from {len(image_files)} images")
    print(f"📁 Input: {input_folder}")
    print(f"🎥 Output: {output_path}")
    
    # Check GPU support
    use_gpu = check_gpu_support(ffmpeg_path)
    if use_gpu:
        print("🚀 Using NVIDIA GPU acceleration")
    else:
        print("💻 Using CPU (ultrafast preset)")
    
    try:
        start_time = time.time()
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory(prefix="fast_export_") as temp_dir:
            print(f"📁 Temp dir: {temp_dir}")
            
            # Copy images with sequential naming (faster than symlinks on Windows)
            print("📋 Preparing images...")
            for i, img_path in enumerate(image_files):
                temp_name = f"frame_{i+1:06d}.png"
                temp_path = os.path.join(temp_dir, temp_name)
                shutil.copy2(img_path, temp_path)
                
                if i % 20 == 0:
                    print(f"📋 {i+1}/{len(image_files)}", end='\r')
            
            print(f"\n📋 Prepared {len(image_files)} images")
            
            # Build ultra-fast FFmpeg command
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")
            
            if use_gpu:
                # GPU-accelerated command
                cmd = [
                    ffmpeg_path, "-y", "-hwaccel", "cuda",
                    "-framerate", str(fps), "-i", input_pattern,
                    "-c:v", "h264_nvenc", "-preset", "fast",
                    "-cq", "23", "-pix_fmt", "yuv420p",
                    output_path
                ]
            else:
                # CPU ultra-fast command
                cmd = [
                    ffmpeg_path, "-y",
                    "-framerate", str(fps), "-i", input_pattern,
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "23", "-pix_fmt", "yuv420p", 
                    "-threads", "0",  # Use all CPU cores
                    output_path
                ]
            
            print(f"🎬 Command: {' '.join(cmd[:8])}...")
            
            # Run FFmpeg
            print("🎬 Encoding video...")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Simple progress
            while process.poll() is None:
                elapsed = time.time() - start_time
                print(f"🎬 Encoding... {elapsed:.1f}s", end='\r')
                time.sleep(0.2)
            
            stdout, stderr = process.communicate()
            total_time = time.time() - start_time
            
            if process.returncode == 0:
                print(f"\n✅ Video created in {total_time:.1f}s!")
                
                # Show stats
                if os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    fps_processed = len(image_files) / total_time
                    print(f"📊 Size: {size_mb:.1f} MB")
                    print(f"⚡ Speed: {fps_processed:.1f} images/second")
                
                return True
            else:
                print(f"\n❌ FFmpeg failed: {stderr.decode()[:200]}")
                return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Video Exporter")
    parser.add_argument("input_folder", help="Input image folder")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--fps", type=float, default=30, help="Framerate (default: 30)")
    parser.add_argument("--quality", choices=["high", "medium", "low"], default="medium", 
                       help="Quality (default: medium)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_folder):
        print(f"❌ Input folder not found: {args.input_folder}")
        sys.exit(1)
    
    print("⚡ Ultra-Fast Video Exporter")
    print("=" * 30)
    
    success = create_fast_video(args.input_folder, args.output, args.fps, args.quality)
    
    if success:
        print("🎉 Success!")
        sys.exit(0)
    else:
        print("💥 Failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
