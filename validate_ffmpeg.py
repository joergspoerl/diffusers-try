#!/usr/bin/env python3
"""Validate FFmpeg integration"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_ffmpeg_simple():
    print("🔍 FFmpeg Integration Test")
    print("="*40)
    
    try:
        from sdgen.utils import find_ffmpeg
        
        ffmpeg_path = find_ffmpeg()
        
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            size = os.path.getsize(ffmpeg_path)
            print(f"✅ SUCCESS: FFmpeg found at {ffmpeg_path}")
            print(f"   Size: {size:,} bytes ({size/1024/1024:.1f} MB)")
            
            # Quick version test
            print(f"   Testing execution...")
            import subprocess
            try:
                result = subprocess.run([ffmpeg_path, "-version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    print(f"   ✅ {version_line}")
                else:
                    print(f"   ❌ FFmpeg execution failed")
            except Exception as e:
                print(f"   ⚠️ Version test failed: {e}")
                
        else:
            print("❌ FAIL: FFmpeg not found or not accessible")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        
    print("\n🧪 Component Integration:")
    
    # Test imports
    components = [
        ("export_dialog", "SimpleVideoExportDialog"),
        ("preview_viewer", "PreviewViewer")
    ]
    
    for module_name, class_name in components:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"   ✅ {module_name}.{class_name}")
            else:
                print(f"   ⚠️ {module_name} imported but {class_name} not found")
        except Exception as e:
            print(f"   ❌ {module_name}: {e}")

if __name__ == "__main__":
    test_ffmpeg_simple()
