#!/usr/bin/env python3
"""
Diffusers Preview Viewer - Standalone Image Viewer for Output Folders
"""

import sys
import os
import time
import glob
import subprocess
import tempfile
import shutil
from typing import List, Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSlider, QSpinBox, QCheckBox,
    QMessageBox, QProgressBar, QLineEdit, QFileDialog, QGroupBox, QDialog
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QFileSystemWatcher
from PyQt6.QtGui import QPixmap, QFont, QPainter
from export_dialog import SimpleVideoExportDialog


class ImagePlayer(QThread):
    """Background thread for playing image sequences"""
    
    imageChanged = pyqtSignal(str)
    frameChanged = pyqtSignal(int, int)
    blendChanged = pyqtSignal(object)  # For blended pixmap
    
    def __init__(self):
        super().__init__()
        self.image_files = []
        self.current_index = 0
        self.is_playing = False
        self.fps = 10
        self.loop_mode = True
        self.blend_mode = False
        self.blend_frames = 3
        self.blend_alpha = 0.3
        self.interpolate_frames = False
        self.interpolation_steps = 5
        self.current_interpolation_step = 0
        
    def set_images(self, image_files: List[str]):
        """Set the image sequence, preserving current position if possible"""
        old_count = len(self.image_files) if self.image_files else 0
        self.image_files = image_files
        
        # Keep current index if still valid, otherwise reset to 0
        if self.current_index >= len(image_files):
            self.current_index = 0
        # If we had no images before, start at 0
        elif old_count == 0:
            self.current_index = 0
        # Otherwise keep current position
        
    def set_fps(self, fps: int):
        """Set playback FPS"""
        self.fps = fps
        
    def set_blend_mode(self, enabled: bool):
        """Enable/disable blend mode"""
        self.blend_mode = enabled
        
    def set_blend_frames(self, frames: int):
        """Set number of frames to blend"""
        self.blend_frames = max(1, frames)
        
    def set_blend_alpha(self, alpha: float):
        """Set blend alpha value"""
        self.blend_alpha = max(0.0, min(1.0, alpha))
        
    def set_interpolate_frames(self, enabled: bool):
        """Enable/disable frame interpolation"""
        self.interpolate_frames = enabled
        self.current_interpolation_step = 0
        
    def set_interpolation_steps(self, steps: int):
        """Set number of interpolation steps between frames"""
        self.interpolation_steps = max(2, steps)
        self.current_interpolation_step = 0
        
    def play(self):
        """Start playback"""
        self.is_playing = True
        if not self.isRunning():
            self.start()
            
    def pause(self):
        """Pause playback"""
        self.is_playing = False
        
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        self.current_index = 0
        
    def next_frame(self):
        """Go to next frame manually"""
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            if self.blend_mode:
                self.emit_blended_frame()
            else:
                self.imageChanged.emit(self.image_files[self.current_index])
            self.frameChanged.emit(self.current_index + 1, len(self.image_files))
            
    def prev_frame(self):
        """Go to previous frame manually"""
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            if self.blend_mode:
                self.emit_blended_frame()
            else:
                self.imageChanged.emit(self.image_files[self.current_index])
            self.frameChanged.emit(self.current_index + 1, len(self.image_files))
            
    def goto_frame(self, frame_number: int):
        """Go to specific frame"""
        if self.image_files and 0 <= frame_number < len(self.image_files):
            self.current_index = frame_number
            if self.blend_mode:
                self.emit_blended_frame()
            else:
                self.imageChanged.emit(self.image_files[self.current_index])
            self.frameChanged.emit(self.current_index + 1, len(self.image_files))
            
    def emit_blended_frame(self):
        """Create and emit blended frame with optional interpolation"""
        if not self.image_files:
            return
            
        try:
            from PyQt6.QtGui import QPixmap, QPainter
            
            # Handle frame interpolation for smoother transitions
            if self.interpolate_frames and self.interpolation_steps > 2:
                # Calculate current and next frame indices
                current_idx = self.current_index % len(self.image_files)
                next_idx = (current_idx + 1) % len(self.image_files)
                
                # Load both frames
                current_pixmap = QPixmap(self.image_files[current_idx])
                next_pixmap = QPixmap(self.image_files[next_idx])
                
                if current_pixmap.isNull() or next_pixmap.isNull():
                    return
                
                # Calculate smooth interpolated alpha
                step_progress = self.current_interpolation_step / (self.interpolation_steps - 1)
                
                # Use full intensity (100%) for interpolation mode instead of blend_alpha
                final_alpha = step_progress
                
                # Create interpolated frame
                result = QPixmap(current_pixmap.size())
                result.fill()
                
                painter = QPainter(result)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                
                # Draw current frame (always full opacity)
                painter.setOpacity(1.0)
                painter.drawPixmap(0, 0, current_pixmap)
                
                # Draw next frame with interpolated alpha (blend over current)
                painter.setOpacity(final_alpha)
                painter.drawPixmap(0, 0, next_pixmap.scaled(current_pixmap.size()))
                
                painter.end()
                
                # Increment interpolation step
                self.current_interpolation_step += 1
                if self.current_interpolation_step >= self.interpolation_steps:
                    self.current_interpolation_step = 0
                    # Only advance to next frame when interpolation cycle is complete
                    self.current_index = (self.current_index + 1) % len(self.image_files)
                
                self.blendChanged.emit(result)
                return
            
            # Original multi-frame blending for non-interpolation mode
            # Get frames to blend
            frames_to_blend = []
            for i in range(self.blend_frames):
                frame_index = (self.current_index - i) % len(self.image_files)
                frames_to_blend.append(self.image_files[frame_index])
            
            # Load first frame as base
            base_pixmap = QPixmap(frames_to_blend[0])
            if base_pixmap.isNull():
                return
                
            # Create result pixmap
            result = QPixmap(base_pixmap.size())
            result.fill()
            
            # Blend frames
            painter = QPainter(result)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            # Draw base frame
            painter.setOpacity(1.0)
            painter.drawPixmap(0, 0, base_pixmap)
            
            # Blend additional frames
            for i, frame_path in enumerate(frames_to_blend[1:], 1):
                frame_pixmap = QPixmap(frame_path)
                if not frame_pixmap.isNull():
                    # Calculate alpha based on frame distance
                    alpha = self.blend_alpha * (1.0 - (i / self.blend_frames))
                    painter.setOpacity(alpha)
                    painter.drawPixmap(0, 0, frame_pixmap.scaled(base_pixmap.size()))
            
            painter.end()
            self.blendChanged.emit(result)
            
        except Exception as e:
            # Fallback to normal mode
            self.imageChanged.emit(self.image_files[self.current_index])
            
    def run(self):
        """Main playback loop"""
        while True:
            if self.is_playing and self.image_files:
                # Emit current image or blended frame
                if self.blend_mode:
                    self.emit_blended_frame()
                else:
                    self.imageChanged.emit(self.image_files[self.current_index])
                    
                # Handle frame advancement with interpolation consideration
                if not (self.blend_mode and self.interpolate_frames and self.interpolation_steps > 2):
                    # Normal frame advancement
                    self.frameChanged.emit(self.current_index + 1, len(self.image_files))
                    
                    # Advance to next frame
                    self.current_index = (self.current_index + 1) % len(self.image_files)
                else:
                    # Interpolation mode: only emit frame change, don't advance yet
                    # Frame advancement is handled in emit_blended_frame()
                    self.frameChanged.emit(self.current_index + 1, len(self.image_files))
                
                # Wait based on FPS
                self.msleep(int(1000 / self.fps))
            else:
                self.msleep(100)


class VideoExporter(QThread):
    """Background thread for video generation from frames"""
    
    progressChanged = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.image_files = []
        self.output_path = ""
        self.fps = 25
        self.blend_frames = 1
        self.blend_alpha = 0.3
        self.interpolate_frames = False
        self.interpolation_steps = 5
        self.video_quality = "high"
        self.upscale_factor = 1
        self.video_codec = "libx264"
        self.video_bitrate = "auto"
        
    def set_export_settings(self, image_files, output_path, fps, 
                          blend_frames=1, blend_alpha=0.3, 
                          interpolate_frames=False, interpolation_steps=5,
                          video_quality="high", upscale_factor=1,
                          video_codec="libx264", video_bitrate="auto"):
        """Set export settings"""
        self.image_files = image_files
        self.output_path = output_path
        self.fps = fps
        self.blend_frames = blend_frames
        self.blend_alpha = blend_alpha
        self.interpolate_frames = interpolate_frames
        self.interpolation_steps = interpolation_steps
        self.video_quality = video_quality
        self.upscale_factor = upscale_factor
        self.video_codec = video_codec
        self.video_bitrate = video_bitrate
        
    def run(self):
        """Generate video from frames"""
        try:
            self.progressChanged.emit(0, "Vorbereitung...")
            
            # Generate frame sequence with current viewer settings
            temp_dir = tempfile.mkdtemp(prefix="video_export_")
            frame_files = []
            
            total_frames = len(self.image_files)
            if self.interpolate_frames:
                total_frames *= self.interpolation_steps
                
            current_frame = 0
            
            for i, image_path in enumerate(self.image_files):
                if self.interpolate_frames:
                    # Generate interpolated frames
                    for step in range(self.interpolation_steps):
                        frame_file = os.path.join(temp_dir, f"frame_{current_frame:06d}.png")
                        
                        if step == 0:
                            # First step: use original frame
                            self._copy_frame(image_path, frame_file)
                        else:
                            # Generate interpolated frame
                            self._generate_interpolated_frame(
                                image_path, 
                                self.image_files[(i + 1) % len(self.image_files)],
                                step / self.interpolation_steps,
                                frame_file
                            )
                        
                        frame_files.append(frame_file)
                        current_frame += 1
                        
                        progress = int((current_frame / total_frames) * 50)
                        self.progressChanged.emit(progress, f"Frame {current_frame}/{total_frames}")
                        
                elif self.blend_frames > 1:
                    # Generate blended frame
                    frame_file = os.path.join(temp_dir, f"frame_{current_frame:06d}.png")
                    self._generate_blended_frame(i, frame_file)
                    frame_files.append(frame_file)
                    current_frame += 1
                    
                    progress = int((current_frame / total_frames) * 50)
                    self.progressChanged.emit(progress, f"Frame {current_frame}/{total_frames}")
                else:
                    # Use original frame
                    frame_file = os.path.join(temp_dir, f"frame_{current_frame:06d}.png")
                    self._copy_frame(image_path, frame_file)
                    frame_files.append(frame_file)
                    current_frame += 1
                    
                    progress = int((current_frame / total_frames) * 50)
                    self.progressChanged.emit(progress, f"Frame {current_frame}/{total_frames}")
            
            # Generate video with FFmpeg
            self.progressChanged.emit(50, "Video-Encoding...")
            
            # Build FFmpeg command
            ffmpeg_cmd = self._build_ffmpeg_command(temp_dir, self.output_path)
            
            # Run FFmpeg
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor progress
            while process.poll() is None:
                self.msleep(100)
                
            if process.returncode == 0:
                # Cleanup temp files
                shutil.rmtree(temp_dir)
                
                self.progressChanged.emit(100, "Video erfolgreich erstellt!")
                self.finished.emit(self.output_path)
            else:
                error_output = process.stderr.read()
                self.error.emit(f"FFmpeg Fehler: {error_output}")
                
        except Exception as e:
            self.error.emit(f"Export-Fehler: {str(e)}")
    
    def _copy_frame(self, src_path, dst_path):
        """Copy frame with optional upscaling"""
        if self.upscale_factor > 1:
            # Upscale frame
            pixmap = QPixmap(src_path)
            scaled = pixmap.scaled(
                pixmap.width() * self.upscale_factor,
                pixmap.height() * self.upscale_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            scaled.save(dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    
    def _generate_interpolated_frame(self, frame1_path, frame2_path, alpha, output_path):
        """Generate interpolated frame between two frames"""
        pixmap1 = QPixmap(frame1_path)
        pixmap2 = QPixmap(frame2_path)
        
        if self.upscale_factor > 1:
            size = pixmap1.size() * self.upscale_factor
        else:
            size = pixmap1.size()
            
        result = QPixmap(size)
        result.fill()
        
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Draw base frame
        painter.setOpacity(1.0)
        painter.drawPixmap(0, 0, pixmap1.scaled(size))
        
        # Draw interpolated frame
        painter.setOpacity(alpha)
        painter.drawPixmap(0, 0, pixmap2.scaled(size))
        
        painter.end()
        result.save(output_path)
    
    def _generate_blended_frame(self, frame_index, output_path):
        """Generate blended frame using multiple source frames"""
        # Get frames to blend
        frames_to_blend = []
        for i in range(self.blend_frames):
            idx = (frame_index - i) % len(self.image_files)
            frames_to_blend.append(self.image_files[idx])
        
        # Load base frame
        base_pixmap = QPixmap(frames_to_blend[0])
        
        if self.upscale_factor > 1:
            size = base_pixmap.size() * self.upscale_factor
        else:
            size = base_pixmap.size()
            
        result = QPixmap(size)
        result.fill()
        
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Draw base frame
        painter.setOpacity(1.0)
        painter.drawPixmap(0, 0, base_pixmap.scaled(size))
        
        # Blend additional frames
        for i, frame_path in enumerate(frames_to_blend[1:], 1):
            frame_pixmap = QPixmap(frame_path)
            alpha = self.blend_alpha * (1.0 - (i / self.blend_frames))
            painter.setOpacity(alpha)
            painter.drawPixmap(0, 0, frame_pixmap.scaled(size))
        
        painter.end()
        result.save(output_path)
    
    def _build_ffmpeg_command(self, temp_dir, output_path):
        """Build FFmpeg command based on quality settings"""
        base_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", os.path.join(temp_dir, "frame_%06d.png"),
            "-c:v", self.video_codec,
            "-pix_fmt", "yuv420p"
        ]
        
        # Add bitrate or CRF based on quality
        if self.video_bitrate != "auto":
            base_cmd.extend(["-b:v", self.video_bitrate])
        else:
            if self.video_quality == "youtube":
                # YouTube optimized
                base_cmd.extend([
                    "-crf", "18",
                    "-preset", "slower",
                    "-movflags", "+faststart"
                ])
            elif self.video_quality == "high":
                # High quality
                base_cmd.extend([
                    "-crf", "15",
                    "-preset", "slow"
                ])
            elif self.video_quality == "medium":
                # Medium quality
                base_cmd.extend([
                    "-crf", "23",
                    "-preset", "medium"
                ])
            else:  # fast
                # Fast encoding
                base_cmd.extend([
                    "-crf", "28",
                    "-preset", "fast"
                ])
        
        base_cmd.append(output_path)
        return base_cmd


class VideoExportDialog(QDialog):
    """Dialog for video export settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.video_exporter = VideoExporter()
        self.video_exporter.progressChanged.connect(self.update_progress)
        self.video_exporter.finished.connect(self.export_finished)
        self.video_exporter.error.connect(self.export_error)
        
        self.setModal(True)  # Modaler Dialog
        self.init_ui()
        
    def init_ui(self):
        """Initialize export dialog UI"""
        self.setWindowTitle("🎬 Video Export")
        self.setFixedSize(420, 380)
        
        # Center on parent
        if self.parent():
            parent_geo = self.parent().geometry()
            self.move(parent_geo.x() + 50, parent_geo.y() + 50)
        
        # Simplified dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: Arial, sans-serif;
                font-size: 12px;
            }
            QLabel {
                color: #ffffff;
                margin: 2px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QSpinBox, QComboBox, QLineEdit {
                background-color: #404040;
                border: 1px solid #666;
                border-radius: 3px;
                padding: 4px;
                min-height: 20px;
            }
            QProgressBar {
                border: 1px solid #666;
                border-radius: 3px;
                background-color: #404040;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
            QCheckBox {
                color: #ffffff;
                margin: 4px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("� Video Export Einstellungen")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Basic settings
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(10)
        
        # FPS
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(25)
        self.fps_spin.setFixedWidth(80)
        fps_row.addWidget(self.fps_spin)
        fps_row.addStretch()
        settings_layout.addLayout(fps_row)
        
        # Quality
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Qualität:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["youtube", "high", "medium", "fast"])
        self.quality_combo.setCurrentText("youtube")
        self.quality_combo.setFixedWidth(100)
        quality_row.addWidget(self.quality_combo)
        quality_row.addStretch()
        settings_layout.addLayout(quality_row)
        
        # Upscaling
        upscale_row = QHBoxLayout()
        upscale_row.addWidget(QLabel("Upscaling:"))
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["1x", "2x", "4x"])
        self.upscale_combo.setCurrentIndex(1)
        self.upscale_combo.setFixedWidth(80)
        upscale_row.addWidget(self.upscale_combo)
        upscale_row.addStretch()
        settings_layout.addLayout(upscale_row)
        
        # Codec
        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel("Codec:"))
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["libx264", "libx265", "libvpx-vp9"])
        self.codec_combo.setFixedWidth(100)
        codec_row.addWidget(self.codec_combo)
        codec_row.addStretch()
        settings_layout.addLayout(codec_row)
        
        # Bitrate
        bitrate_row = QHBoxLayout()
        bitrate_row.addWidget(QLabel("Bitrate:"))
        self.bitrate_combo = QComboBox()
        self.bitrate_combo.addItems(["auto", "2M", "5M", "10M", "20M"])
        self.bitrate_combo.setFixedWidth(80)
        bitrate_row.addWidget(self.bitrate_combo)
        bitrate_row.addStretch()
        settings_layout.addLayout(bitrate_row)
        
        layout.addLayout(settings_layout)
        
        # Viewer settings
        self.copy_settings_check = QCheckBox("Viewer-Einstellungen übernehmen")
        self.copy_settings_check.setChecked(True)
        layout.addWidget(self.copy_settings_check)
        
        self.settings_preview = QLabel()
        self.settings_preview.setStyleSheet("color: #90EE90; font-size: 11px;")
        self.settings_preview.setWordWrap(True)
        layout.addWidget(self.settings_preview)
        
        # Output file
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Ausgabe:"))
        self.path_edit = QLineEdit()
        self.path_edit.setText("video_export.mp4")
        output_layout.addWidget(self.path_edit)
        
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self.browse_output_path)
        output_layout.addWidget(browse_btn)
        layout.addLayout(output_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Bereit für Export")
        self.status_label.setStyleSheet("color: #90EE90;")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("🎬 Video Exportieren")
        self.export_btn.clicked.connect(self.start_export)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: 1px solid #45a049;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(self.export_btn)
        
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.close)
        close_btn.setFixedWidth(80)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Update preview
        self.copy_settings_check.toggled.connect(self.update_settings_preview)
        self.update_settings_preview()
    
    def update_settings_preview(self):
        """Update preview of current viewer settings"""
        if not self.copy_settings_check.isChecked():
            self.settings_preview.setText("Standard-Einstellungen werden verwendet")
            return
            
        preview = "Aktuelle Viewer-Einstellungen:\n"
        
        if self.parent_viewer.blend_check.isChecked():
            frames = self.parent_viewer.blend_frames_spin.value()
            alpha = self.parent_viewer.blend_alpha_slider.value()
            preview += f"• Normale Überblendung: {frames} Frames, {alpha}%\n"
        
        if self.parent_viewer.interpolate_check.isChecked():
            steps = self.parent_viewer.interpolation_steps_spin.value()
            preview += f"• Zwischenframes: {steps} Schritte\n"
        
        if not self.parent_viewer.blend_check.isChecked() and not self.parent_viewer.interpolate_check.isChecked():
            preview += "• Keine Effekte (Original-Frames)"
        
        self.settings_preview.setText(preview.strip())
    
    def browse_output_path(self):
        """Browse for output path"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Video speichern als",
            self.path_edit.text(),
            "MP4 Videos (*.mp4);;AVI Videos (*.avi);;MOV Videos (*.mov);;MKV Videos (*.mkv)"
        )
        
        if file_path:
            self.path_edit.setText(file_path)
    
    def start_export(self):
        """Start video export"""
        if not self.parent_viewer.player.image_files:
            QMessageBox.warning(self, "Fehler", "Keine Bilder geladen!")
            return
            
        output_path = self.path_edit.text()
        if not output_path:
            QMessageBox.warning(self, "Fehler", "Bitte Ausgabepfad wählen!")
            return
        
        # Get settings
        fps = self.fps_spin.value()
        quality = self.quality_combo.currentText()
        upscale_text = self.upscale_combo.currentText()
        upscale_factor = int(upscale_text.split('x')[0])
        
        # Get codec
        codec_text = self.codec_combo.currentText()
        codec = codec_text.split()[0]  # Extract codec name
        
        # Get bitrate
        bitrate = self.bitrate_combo.currentText()
        
        # Get viewer settings if enabled
        blend_frames = 1
        blend_alpha = 0.3
        interpolate_frames = False
        interpolation_steps = 5
        
        if self.copy_settings_check.isChecked():
            if self.parent_viewer.blend_check.isChecked():
                blend_frames = self.parent_viewer.blend_frames_spin.value()
                blend_alpha = self.parent_viewer.blend_alpha_slider.value() / 100.0
            
            if self.parent_viewer.interpolate_check.isChecked():
                interpolate_frames = True
                interpolation_steps = self.parent_viewer.interpolation_steps_spin.value()
        
        # Show confirmation with estimated info
        estimated_frames = len(self.parent_viewer.player.image_files)
        if interpolate_frames:
            estimated_frames *= interpolation_steps
            
        reply = QMessageBox.question(
            self,
            "Export bestätigen",
            f"Video-Export starten?\n\n"
            f"📁 Ausgabe: {os.path.basename(output_path)}\n"
            f"🎬 Geschätzte Frames: {estimated_frames}\n"
            f"⏱️ FPS: {fps}\n"
            f"📊 Qualität: {quality}\n"
            f"🔍 Upscaling: {upscale_factor}x\n"
            f"🎨 Effekte: {'Ja' if (blend_frames > 1 or interpolate_frames) else 'Nein'}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Setup exporter
        self.video_exporter.set_export_settings(
            self.parent_viewer.player.image_files,
            output_path,
            fps,
            blend_frames,
            blend_alpha,
            interpolate_frames,
            interpolation_steps,
            quality,
            upscale_factor,
            codec,
            bitrate
        )
        
        # Start export
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Export gestartet...")
        
        self.video_exporter.start()
    
    def update_progress(self, progress, message):
        """Update export progress"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def export_finished(self, output_path):
        """Handle export completion"""
        self.export_btn.setEnabled(True)
        self.status_label.setText(f"Export abgeschlossen: {os.path.basename(output_path)}")
        
        # Show success message
        reply = QMessageBox.question(
            self,
            "Export abgeschlossen! 🎉",
            f"Video erfolgreich erstellt:\n{output_path}\n\nVideo öffnen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.startfile(output_path)  # Windows
            except:
                # Fallback for other systems
                subprocess.run(['xdg-open', output_path])
    
    def export_error(self, error_message):
        """Handle export error"""
        self.export_btn.setEnabled(True)
        self.status_label.setText("Export fehlgeschlagen!")
        QMessageBox.critical(self, "Export-Fehler", f"Fehler beim Video-Export:\n\n{error_message}")


class PreviewViewer(QMainWindow):
    """Main preview viewer application"""
    
    def __init__(self):
        super().__init__()
        self.output_dir = "outputs"
        self.current_folder = None
        self.watch_mode = False
        
        # File system watcher for live updates
        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(self.on_directory_changed)
        
        # Image player
        self.player = ImagePlayer()
        self.player.imageChanged.connect(self.display_image)
        self.player.frameChanged.connect(self.update_frame_info)
        self.player.blendChanged.connect(self.display_blended_image)
        
        # Timer for refreshing image list
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_images)
        
        # Timer for live mode
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.check_for_new_images)
        
        self.init_ui()
        self.refresh_folders()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Diffusers Preview Viewer")
        self.setGeometry(100, 100, 1000, 700)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QComboBox, QSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #5a5a5a;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #4a6fa5;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a7fb5;
            }
            QPushButton:pressed {
                background-color: #3a5f95;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
            QSlider::groove:horizontal {
                border: 1px solid #5a5a5a;
                height: 8px;
                background: #3c3c3c;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6a9bd1;
                border: 1px solid #5a5a5a;
                width: 18px;
                border-radius: 9px;
                margin: -5px 0;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)
        
        # Folder selection
        controls_layout.addWidget(QLabel("Ordner:"))
        self.folder_combo = QComboBox()
        self.folder_combo.currentTextChanged.connect(self.on_folder_changed)
        controls_layout.addWidget(self.folder_combo)
        
        # Refresh button
        self.refresh_btn = QPushButton("🔄 Aktualisieren")
        self.refresh_btn.clicked.connect(self.refresh_folders)
        controls_layout.addWidget(self.refresh_btn)
        
        controls_layout.addStretch()
        
        # Playback controls
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_btn)
        
        self.prev_btn = QPushButton("⏮️")
        self.prev_btn.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("⏭️")
        self.next_btn.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_btn)
        
        # Live mode button
        self.live_btn = QPushButton("📡 Aktuell")
        self.live_btn.setCheckable(True)
        self.live_btn.clicked.connect(self.toggle_live_mode)
        controls_layout.addWidget(self.live_btn)
        
        # FPS control
        controls_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(50)  # Default auf 50 FPS
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        controls_layout.addWidget(self.fps_spin)
        
        # Video Export button
        self.export_btn = QPushButton("🎬 Video Export")
        self.export_btn.clicked.connect(self.open_video_export)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B35;
                border: 2px solid #E55A2B;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #E55A2B;
            }
        """)
        controls_layout.addWidget(self.export_btn)
        
        # Auto-refresh checkbox
        self.auto_refresh_check = QCheckBox("Auto-Refresh")
        self.auto_refresh_check.setChecked(True)
        self.auto_refresh_check.toggled.connect(self.toggle_auto_refresh)
        controls_layout.addWidget(self.auto_refresh_check)
        
        controls_layout.addStretch()
        
        # Blend controls
        blend_layout = QHBoxLayout()
        main_layout.addLayout(blend_layout)
        
        # Blend mode checkbox
        self.blend_check = QCheckBox("🌊 Frame Überblendung")
        self.blend_check.toggled.connect(self.toggle_blend_mode)
        blend_layout.addWidget(self.blend_check)
        
        # Blend frames control
        blend_layout.addWidget(QLabel("Frames:"))
        self.blend_frames_spin = QSpinBox()
        self.blend_frames_spin.setRange(2, 10)
        self.blend_frames_spin.setValue(3)
        self.blend_frames_spin.valueChanged.connect(self.on_blend_frames_changed)
        blend_layout.addWidget(self.blend_frames_spin)
        
        # Blend alpha control
        blend_layout.addWidget(QLabel("Intensität:"))
        self.blend_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_alpha_slider.setRange(10, 80)
        self.blend_alpha_slider.setValue(30)
        self.blend_alpha_slider.valueChanged.connect(self.on_blend_alpha_changed)
        self.blend_alpha_slider.setMaximumWidth(100)
        blend_layout.addWidget(self.blend_alpha_slider)
        
        self.blend_alpha_label = QLabel("30%")
        self.blend_alpha_label.setMinimumWidth(40)
        blend_layout.addWidget(self.blend_alpha_label)
        
        blend_layout.addStretch()
        
        # Interpolation controls layout
        interpolation_layout = QHBoxLayout()
        main_layout.addLayout(interpolation_layout)
        
        # Interpolation checkbox
        self.interpolate_check = QCheckBox("🎬 Zwischenframes")
        self.interpolate_check.toggled.connect(self.toggle_interpolation)
        interpolation_layout.addWidget(self.interpolate_check)
        
        # Interpolation steps control
        interpolation_layout.addWidget(QLabel("Schritte:"))
        self.interpolation_steps_spin = QSpinBox()
        self.interpolation_steps_spin.setRange(2, 20)
        self.interpolation_steps_spin.setValue(20)  # Default auf 20 Schritte
        self.interpolation_steps_spin.valueChanged.connect(self.on_interpolation_steps_changed)
        interpolation_layout.addWidget(self.interpolation_steps_spin)
        
        interpolation_layout.addStretch()
        
        # Frame control layout
        frame_layout = QHBoxLayout()
        main_layout.addLayout(frame_layout)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        frame_layout.addWidget(self.frame_slider)
        
        # Frame info
        self.frame_label = QLabel("Frame: 0/0")
        frame_layout.addWidget(self.frame_label)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                background-color: #1e1e1e;
                min-height: 400px;
            }
        """)
        self.image_label.setText("Ordner auswählen und Play drücken")
        main_layout.addWidget(self.image_label)
        
        # Status bar
        self.status_label = QLabel("Bereit")
        self.status_label.setStyleSheet("QLabel { color: #90EE90; padding: 5px; }")
        main_layout.addWidget(self.status_label)
        
        # Initialize UI state - both options available independently
        self.blend_frames_spin.setEnabled(False)
        self.blend_alpha_slider.setEnabled(False) 
        self.interpolation_steps_spin.setEnabled(False)
        
        # Activate Zwischenframes by default
        self.interpolate_check.setChecked(True)
        self.toggle_interpolation(True)  # Activate interpolation mode
        
        # Set initial FPS to player
        self.player.set_fps(50)
        
    def refresh_folders(self):
        """Refresh the folder list"""
        try:
            if not os.path.exists(self.output_dir):
                self.status_label.setText("Output-Ordner nicht gefunden")
                return
                
            # Get all directories in output folder
            folders = []
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    folders.append(item)
                    
            # Sort by creation time (newest first)
            folders.sort(key=lambda x: os.path.getctime(os.path.join(self.output_dir, x)), reverse=True)
            
            # Update combo box
            current_text = self.folder_combo.currentText()
            self.folder_combo.clear()
            self.folder_combo.addItems(folders)
            
            # Select newest folder or restore previous selection
            if folders:
                if current_text in folders:
                    self.folder_combo.setCurrentText(current_text)
                else:
                    self.folder_combo.setCurrentIndex(0)  # Select newest
                    
            self.status_label.setText(f"Gefunden: {len(folders)} Ordner")
            
        except Exception as e:
            self.status_label.setText(f"Fehler: {e}")
            
    def on_folder_changed(self, folder_name: str):
        """Handle folder selection change"""
        if not folder_name:
            return
            
        self.current_folder = os.path.join(self.output_dir, folder_name)
        self.load_images()
        
        # Setup file watching for live mode
        self.file_watcher.removePaths(self.file_watcher.directories())
        if os.path.exists(self.current_folder):
            self.file_watcher.addPath(self.current_folder)
            
    def load_images(self):
        """Load images from current folder"""
        if not self.current_folder or not os.path.exists(self.current_folder):
            return
            
        try:
            # Find all image files
            image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
            image_files = []
            
            for pattern in image_patterns:
                image_files.extend(glob.glob(os.path.join(self.current_folder, pattern)))
                
            # Sort by filename (usually contains timestamp/frame number)
            image_files.sort()
            
            # Update player
            old_image_count = len(self.player.image_files) if self.player.image_files else 0
            self.player.set_images(image_files)
            
            # Update UI
            self.frame_slider.setRange(0, max(0, len(image_files) - 1))
            
            # Only reset slider if we had no images before or current position is invalid
            if old_image_count == 0 or self.frame_slider.value() >= len(image_files):
                self.frame_slider.setValue(0)
            # Otherwise keep current slider position
            
            self.status_label.setText(f"Geladen: {len(image_files)} Bilder")
            
            # Display current image if available
            if image_files and self.player.current_index < len(image_files):
                self.display_image(image_files[self.player.current_index])
                self.update_frame_info(self.player.current_index + 1, len(image_files))
            elif image_files:
                # Fallback if index is somehow invalid
                self.display_image(image_files[0])
                self.update_frame_info(1, len(image_files))
            else:
                self.image_label.setText("Keine Bilder gefunden")
                self.update_frame_info(0, 0)
                
        except Exception as e:
            self.status_label.setText(f"Fehler beim Laden: {e}")
            
    def display_image(self, image_path: str):
        """Display an image"""
        try:
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                self._display_pixmap(pixmap)
                
                # Update status
                filename = os.path.basename(image_path)
                blend_status = " (Überblendung)" if self.blend_check.isChecked() else ""
                self.status_label.setText(f"Anzeige: {filename}{blend_status}")
                
        except Exception as e:
            self.status_label.setText(f"Fehler bei Anzeige: {e}")
            
    def display_blended_image(self, pixmap):
        """Display a blended pixmap"""
        try:
            self._display_pixmap(pixmap)
            blend_info = f" (Überblendung: {self.blend_frames_spin.value()} Frames)"
            self.status_label.setText(f"Anzeige: Blended Frame{blend_info}")
        except Exception as e:
            self.status_label.setText(f"Fehler bei Blend-Anzeige: {e}")
            
    def _display_pixmap(self, pixmap):
        """Helper method to display any pixmap"""
        if not pixmap.isNull():
            # Scale to fit label while maintaining aspect ratio
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(
                label_size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
    def update_frame_info(self, current_frame: int, total_frames: int):
        """Update frame information"""
        self.frame_label.setText(f"Frame: {current_frame}/{total_frames}")
        
        # Update slider without triggering signal
        self.frame_slider.blockSignals(True)
        if total_frames > 0:
            self.frame_slider.setValue(current_frame - 1)
        self.frame_slider.blockSignals(False)
        
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.player.is_playing:
            self.player.pause()
            self.play_btn.setText("▶️ Play")
            self.status_label.setText("Pausiert")
        else:
            self.player.play()
            self.play_btn.setText("⏸️ Pause")
            self.status_label.setText("Spielt ab...")
            
    def stop_playback(self):
        """Stop playback"""
        self.player.stop()
        self.play_btn.setText("▶️ Play")
        self.status_label.setText("Gestoppt")
        
        # Go to first frame
        if self.player.image_files:
            self.player.goto_frame(0)
            
    def prev_frame(self):
        """Go to previous frame"""
        self.player.prev_frame()
        
    def next_frame(self):
        """Go to next frame"""
        self.player.next_frame()
        
    def on_fps_changed(self, fps: int):
        """Handle FPS change"""
        self.player.set_fps(fps)
        
    def toggle_blend_mode(self, enabled: bool):
        """Toggle normal blend mode"""
        self.player.set_blend_mode(enabled)
        
        if enabled:
            # Disable interpolation checkbox and controls to avoid conflicts
            self.interpolate_check.setChecked(False)
            self.interpolate_check.setEnabled(False)
            self.interpolation_steps_spin.setEnabled(False)
            self.player.set_interpolate_frames(False)
            
            # Enable normal blend controls
            self.blend_frames_spin.setEnabled(True)
            self.blend_alpha_slider.setEnabled(True)
            
            self.status_label.setText("Normaler Überblendmodus aktiviert")
            
            # Refresh current frame
            if self.player.image_files:
                self.player.emit_blended_frame()
        else:
            # Disable blend mode
            self.blend_frames_spin.setEnabled(False)
            self.blend_alpha_slider.setEnabled(False)
            
            # Re-enable interpolation checkbox
            self.interpolate_check.setEnabled(True)
            
            self.status_label.setText("Normaler Überblendmodus deaktiviert")
            
            # Show normal frame
            if self.player.image_files and self.player.current_index < len(self.player.image_files):
                self.display_image(self.player.image_files[self.player.current_index])
                
    def on_blend_frames_changed(self, frames: int):
        """Handle blend frames change"""
        self.player.set_blend_frames(frames)
        
        # Refresh if blend mode is active
        if self.blend_check.isChecked() and self.player.image_files:
            self.player.emit_blended_frame()
            
    def on_blend_alpha_changed(self, value: int):
        """Handle blend alpha change"""
        alpha = value / 100.0
        self.player.set_blend_alpha(alpha)
        self.blend_alpha_label.setText(f"{value}%")
        
        # Refresh if blend mode is active
        if self.blend_check.isChecked() and self.player.image_files:
            self.player.emit_blended_frame()
    
    def toggle_interpolation(self, enabled: bool):
        """Toggle frame interpolation"""
        self.player.set_interpolate_frames(enabled)
        
        if enabled:
            # Activate interpolation blend mode independently
            self.player.set_blend_mode(True)
            self.interpolation_steps_spin.setEnabled(True)
            
            # Disable normal blend checkbox and controls to avoid conflicts
            self.blend_check.setChecked(False)
            self.blend_check.setEnabled(False)
            self.blend_frames_spin.setEnabled(False)
            self.blend_alpha_slider.setEnabled(False)
            
            self.status_label.setText("Zwischenframes-Modus aktiviert (100% Intensität)")
        else:
            # Deactivate interpolation
            self.player.set_blend_mode(False)
            self.player.set_interpolate_frames(False)
            self.interpolation_steps_spin.setEnabled(False)
            
            # Re-enable normal blend checkbox
            self.blend_check.setEnabled(True)
            self.status_label.setText("Zwischenframes-Modus deaktiviert")
        
        # Refresh display
        if self.player.image_files:
            if enabled:
                self.player.emit_blended_frame()
            else:
                self.display_image(self.player.image_files[self.player.current_index])
            
    def on_interpolation_steps_changed(self, steps: int):
        """Handle interpolation steps change"""
        self.player.set_interpolation_steps(steps)
        
        # Refresh if interpolation is active
        if self.interpolate_check.isChecked() and self.blend_check.isChecked() and self.player.image_files:
            self.player.emit_blended_frame()
    
    def open_video_export(self):
        """Open video export dialog"""
        if not self.player.image_files:
            QMessageBox.warning(self, "Fehler", "Keine Bilder geladen!\n\nBitte wählen Sie zuerst einen Ordner mit generierten Frames aus.")
            return
            
        dialog = SimpleVideoExportDialog(self)
        dialog.exec()
        
    def on_frame_slider_changed(self, value: int):
        """Handle frame slider change"""
        self.player.goto_frame(value)
        
    def toggle_auto_refresh(self, enabled: bool):
        """Toggle auto-refresh"""
        if enabled:
            self.refresh_timer.start(2000)  # Refresh every 2 seconds
        else:
            self.refresh_timer.stop()
            
    def refresh_images(self):
        """Refresh image list (for auto-refresh)"""
        if not self.watch_mode:  # Don't refresh during play mode
            self.load_images()
            
    def toggle_live_mode(self, enabled: bool):
        """Toggle live mode"""
        self.watch_mode = enabled
        
        if enabled:
            self.live_btn.setText("📡 Live AN")
            self.live_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; }")
            
            # Stop normal playback
            self.player.pause()
            self.play_btn.setText("▶️ Play")
            
            # Start live monitoring
            self.live_timer.start(500)  # Check every 500ms
            self.last_image_count = len(self.player.image_files)
            self.last_displayed_image = None
            
            self.status_label.setText("Live-Modus: Warte auf neue Bilder...")
            
        else:
            self.live_btn.setText("📡 Aktuell")
            self.live_btn.setStyleSheet("")
            self.live_timer.stop()
            self.status_label.setText("Live-Modus beendet")
            
    def check_for_new_images(self):
        """Check for new images in live mode"""
        if not self.watch_mode or not self.current_folder:
            return
            
        try:
            # Find all image files
            image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
            image_files = []
            
            for pattern in image_patterns:
                image_files.extend(glob.glob(os.path.join(self.current_folder, pattern)))
                
            image_files.sort()
            
            # Check if we have new images
            if len(image_files) > self.last_image_count:
                # Update player with new images
                self.player.set_images(image_files)
                
                # Display the newest image
                newest_image = image_files[-1]
                if newest_image != self.last_displayed_image:
                    self.display_image(newest_image)
                    self.update_frame_info(len(image_files), len(image_files))
                    self.last_displayed_image = newest_image
                    
                self.last_image_count = len(image_files)
                self.status_label.setText(f"Live: Neues Bild {len(image_files)}")
                
        except Exception as e:
            self.status_label.setText(f"Live-Fehler: {e}")
            
    def on_directory_changed(self, path: str):
        """Handle directory changes from file watcher"""
        if self.watch_mode:
            # Trigger check for new images
            self.check_for_new_images()
        else:
            # Auto-refresh if enabled
            if self.auto_refresh_check.isChecked():
                self.load_images()
                
    def closeEvent(self, event):
        """Handle application close"""
        self.player.pause()
        self.refresh_timer.stop()
        self.live_timer.stop()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Diffusers Preview Viewer")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show viewer
    viewer = PreviewViewer()
    viewer.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
