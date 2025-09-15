#!/usr/bin/env python3
"""
Standalone Video Exporter with FFmpeg-based Interpolation
=========================================================

A clean, command-line video exporter that creates smooth videos from image sequences
with optional frame interpolation using FFmpeg's advanced filters.

Usage:
    python video_exporter.py input_folder output.mp4 [options]

Features:
- FFmpeg-based frame interpolation (minterpolate filter)
- Multiple quality presets
- Upscaling support
- Automatic codec selection
- Progress tracking
- Robust error handling
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

class VideoExporter:
    """Standalone video exporter with FFmpeg interpolation"""
    
    def __init__(self):
        self.ffmpeg_path = None
        self.verbose = False
        
    def find_ffmpeg(self):
        """Find FFmpeg executable in various locations"""
        if self.ffmpeg_path and os.path.exists(self.ffmpeg_path):
            return self.ffmpeg_path
            
        # Check local project folder first
        local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg", "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe")
        if os.path.exists(local_ffmpeg):
            self.log(f"✅ Found local FFmpeg: {local_ffmpeg}")
            self.ffmpeg_path = local_ffmpeg
            return local_ffmpeg
        
        # Check system PATH
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.log("✅ Found FFmpeg in system PATH")
                self.ffmpeg_path = 'ffmpeg'
                return 'ffmpeg'
        except FileNotFoundError:
            pass
        
        # Check common Windows locations
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.log(f"✅ Found FFmpeg at: {path}")
                self.ffmpeg_path = path
                return path
        
        return None
    
    def check_gpu_support(self):
        """Check if NVIDIA GPU encoding is available"""
        if not self.ffmpeg_path:
            return False
            
        try:
            # Check if NVENC is available
            result = subprocess.run([self.ffmpeg_path, "-encoders"], 
                                  capture_output=True, text=True, timeout=10)
            if "h264_nvenc" in result.stdout:
                self.log("✅ NVIDIA GPU encoding (NVENC) available")
                return True
            else:
                self.log("ℹ️ NVIDIA GPU encoding not available, using CPU")
                return False
        except (subprocess.TimeoutExpired, Exception) as e:
            self.log(f"⚠️ GPU check failed: {e}")
            return False
    
    def log(self, message):
        """Print log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def get_unique_filename(self, output_path):
        """Generate unique filename if file already exists"""
        if not os.path.exists(output_path):
            return output_path
        
        base, ext = os.path.splitext(output_path)
        counter = 1
        
        while True:
            new_path = f"{base}_{counter:02d}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1
    
    def find_images(self, input_folder):
        """Find all image files in the input folder"""
        supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tga']
        image_files = []
        
        for format_pattern in supported_formats:
            pattern = os.path.join(input_folder, format_pattern)
            files = glob.glob(pattern, recursive=False)
            image_files.extend(files)
        
        # Sort files naturally
        image_files.sort()
        
        self.log(f"📸 Found {len(image_files)} images in {input_folder}")
        return image_files
    
    def create_video_with_interpolation(self, input_folder, output_path, fps=30, quality="medium", 
                                      upscale="1x", codec="libx264", interpolation_fps=None, 
                                      interpolation_mode="mci"):
        """Create video with FFmpeg-based frame interpolation"""
        
        ffmpeg_path = self.find_ffmpeg()
        if not ffmpeg_path:
            print("❌ Error: FFmpeg not found!")
            print("Please install FFmpeg or place it in the project folder.")
            return False
        
        # Find image files
        image_files = self.find_images(input_folder)
        if not image_files:
            print(f"❌ Error: No image files found in {input_folder}")
            return False
        
        print(f"🎬 Creating video from {len(image_files)} images")
        print(f"📁 Input: {input_folder}")
        print(f"🎥 Output: {output_path}")
        print(f"⚙️ Settings: {fps}fps, {quality} quality, {upscale} scale, {codec} codec")
        
        if interpolation_fps:
            print(f"🔄 Interpolation: {fps}fps → {interpolation_fps}fps using {interpolation_mode}")
        
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory(prefix="video_export_") as temp_dir:
                self.log(f"📁 Created temp directory: {temp_dir}")
                
                # Step 1: Copy/link images to temp directory with sequential naming
                self.log("📋 Preparing image sequence...")
                temp_images = []
                for i, img_path in enumerate(image_files):
                    temp_name = f"frame_{i+1:06d}.png"
                    temp_path = os.path.join(temp_dir, temp_name)
                    shutil.copy2(img_path, temp_path)
                    temp_images.append(temp_path)
                    
                    if i % 50 == 0:
                        print(f"📋 Prepared {i+1}/{len(image_files)} images...")
                
                # Step 2: Build FFmpeg command
                input_pattern = os.path.join(temp_dir, "frame_%06d.png")
                output_path = self.get_unique_filename(output_path)
                
                cmd = [ffmpeg_path, "-y", "-framerate", str(fps), "-i", input_pattern]
                
                # Add interpolation filter if requested
                if interpolation_fps and interpolation_fps > fps:
                    if interpolation_mode == "mci":
                        # Motion Compensated Interpolation
                        filter_str = f"minterpolate=fps={interpolation_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
                    elif interpolation_mode == "blend":
                        # Simple blending interpolation
                        filter_str = f"minterpolate=fps={interpolation_fps}:mi_mode=blend"
                    else:
                        # Duplicate frames (fastest)
                        filter_str = f"minterpolate=fps={interpolation_fps}:mi_mode=dup"
                    
                    # Add upscaling to filter chain if needed
                    if upscale != "1x":
                        upscale_factor = self.parse_upscale_factor(upscale)
                        if upscale_factor != 1:
                            scale_filter = f"scale=iw*{upscale_factor}:ih*{upscale_factor}:flags=lanczos"
                            filter_str = f"{filter_str},{scale_filter}"
                    
                    cmd.extend(["-vf", filter_str])
                else:
                    # No interpolation, just upscaling if needed
                    if upscale != "1x":
                        upscale_factor = self.parse_upscale_factor(upscale)
                        if upscale_factor != 1:
                            cmd.extend(["-vf", f"scale=iw*{upscale_factor}:ih*{upscale_factor}:flags=lanczos"])
                
                # Add codec and quality settings
                cmd.extend(["-c:v", codec])
                
                # Try GPU acceleration first (NVIDIA)
                if codec == "libx264":
                    # Check if we should use GPU encoding
                    if self.check_gpu_support():
                        cmd[-1] = "h264_nvenc"  # Replace libx264 with NVENC
                        self.log("🚀 Using NVIDIA GPU acceleration (NVENC)")
                
                # Quality settings with faster presets
                if quality == "youtube":
                    if "nvenc" in cmd:
                        cmd.extend(["-cq", "18", "-preset", "fast"])
                    else:
                        cmd.extend(["-crf", "18", "-preset", "ultrafast"])
                elif quality == "high":
                    if "nvenc" in cmd:
                        cmd.extend(["-cq", "20", "-preset", "fast"])
                    else:
                        cmd.extend(["-crf", "20", "-preset", "veryfast"])
                elif quality == "medium":
                    if "nvenc" in cmd:
                        cmd.extend(["-cq", "23", "-preset", "fast"])
                    else:
                        cmd.extend(["-crf", "23", "-preset", "ultrafast"])
                elif quality == "low":
                    if "nvenc" in cmd:
                        cmd.extend(["-cq", "28", "-preset", "fast"])
                    else:
                        cmd.extend(["-crf", "28", "-preset", "ultrafast"])
                
                # Output settings
                cmd.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path])
                
                print(f"🎬 FFmpeg command: {' '.join(cmd)}")
                
                # Step 3: Run FFmpeg with progress tracking
                print("🎬 Starting video creation...")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Simple progress tracking
                total_frames = len(image_files)
                if interpolation_fps and interpolation_fps > fps:
                    total_frames = int(total_frames * (interpolation_fps / fps))
                
                # Monitor process
                start_time = time.time()
                while process.poll() is None:
                    elapsed = time.time() - start_time
                    print(f"🎬 Processing... {elapsed:.1f}s elapsed", end='\r')
                    time.sleep(0.5)
                
                # Get final output
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    elapsed = time.time() - start_time
                    print(f"\n✅ Video created successfully in {elapsed:.1f}s!")
                    print(f"📄 Output: {output_path}")
                    
                    # Show file size
                    if os.path.exists(output_path):
                        size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        print(f"📊 File size: {size_mb:.1f} MB")
                    
                    return True
                else:
                    print(f"\n❌ FFmpeg failed with return code {process.returncode}")
                    print(f"Error output: {stderr}")
                    return False
        
        except Exception as e:
            print(f"❌ Error during video creation: {e}")
            return False
    
    def parse_upscale_factor(self, upscale_str):
        """Parse upscale string to numeric factor"""
        if upscale_str.endswith('x'):
            try:
                return float(upscale_str[:-1])
            except ValueError:
                return 1.0
        try:
            return float(upscale_str)
        except ValueError:
            return 1.0
    
    def create_simple_video(self, input_folder, output_path, fps=30, quality="medium", 
                           upscale="1x", codec="libx264"):
        """Create video without interpolation (faster)"""
        return self.create_video_with_interpolation(
            input_folder, output_path, fps, quality, upscale, codec, 
            interpolation_fps=None
        )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Standalone Video Exporter with FFmpeg Interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic video creation
  python video_exporter.py input_folder output.mp4
  
  # High quality with upscaling
  python video_exporter.py input_folder output.mp4 --fps 60 --quality high --upscale 2x
  
  # With frame interpolation
  python video_exporter.py input_folder output.mp4 --fps 30 --interpolation-fps 60 --interpolation-mode mci
  
  # Custom codec and settings
  python video_exporter.py input_folder output.mp4 --codec libx265 --quality youtube
        """
    )
    
    # Required arguments
    parser.add_argument("input_folder", help="Folder containing input images")
    parser.add_argument("output", help="Output video file path")
    
    # Video settings
    parser.add_argument("--fps", type=float, default=30.0, 
                       help="Output framerate (default: 30)")
    parser.add_argument("--quality", choices=["youtube", "high", "medium", "low"], 
                       default="medium", help="Video quality preset (default: medium)")
    parser.add_argument("--upscale", default="1x", 
                       help="Upscale factor, e.g., '2x', '1.5x' (default: 1x)")
    parser.add_argument("--codec", default="libx264", 
                       help="Video codec (default: libx264)")
    
    # Interpolation settings
    parser.add_argument("--interpolation-fps", type=float, 
                       help="Target FPS for interpolation (must be higher than --fps)")
    parser.add_argument("--interpolation-mode", choices=["mci", "blend", "dup"], 
                       default="mci", help="Interpolation mode (default: mci)")
    
    # Other options
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--ffmpeg-path", help="Custom path to FFmpeg executable")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.isdir(args.input_folder):
        print(f"❌ Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    if args.interpolation_fps and args.interpolation_fps <= args.fps:
        print(f"❌ Error: Interpolation FPS ({args.interpolation_fps}) must be higher than base FPS ({args.fps})")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize exporter
    exporter = VideoExporter()
    exporter.verbose = args.verbose
    if args.ffmpeg_path:
        exporter.ffmpeg_path = args.ffmpeg_path
    
    # Run export
    print("🎬 Standalone Video Exporter v1.0")
    print("=" * 40)
    
    if args.interpolation_fps:
        success = exporter.create_video_with_interpolation(
            args.input_folder, args.output, args.fps, args.quality,
            args.upscale, args.codec, args.interpolation_fps, args.interpolation_mode
        )
    else:
        success = exporter.create_simple_video(
            args.input_folder, args.output, args.fps, args.quality,
            args.upscale, args.codec
        )
    
    if success:
        print("🎉 Export completed successfully!")
        sys.exit(0)
    else:
        print("💥 Export failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
