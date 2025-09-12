#!/usr/bin/env python3
"""
Diffusers Preview Viewer - Standalone Image Viewer for Output Folders
"""

import sys
import os
import glob
import time
from typing import List, Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSlider, QSpinBox, QCheckBox,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QFileSystemWatcher
from PyQt6.QtGui import QPixmap, QFont, QPainter


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
        
    def set_images(self, image_files: List[str]):
        """Set the image sequence"""
        self.image_files = image_files
        self.current_index = 0
        
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
        """Create and emit blended frame"""
        if not self.image_files:
            return
            
        try:
            from PyQt6.QtGui import QPixmap, QPainter
            
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
                    
                self.frameChanged.emit(self.current_index + 1, len(self.image_files))
                
                # Move to next frame
                self.current_index = (self.current_index + 1) % len(self.image_files)
                
                # Wait based on FPS
                self.msleep(int(1000 / self.fps))
            else:
                self.msleep(100)


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
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        controls_layout.addWidget(self.fps_spin)
        
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
            self.player.set_images(image_files)
            
            # Update UI
            self.frame_slider.setRange(0, max(0, len(image_files) - 1))
            self.frame_slider.setValue(0)
            
            self.status_label.setText(f"Geladen: {len(image_files)} Bilder")
            
            # Display first image if available
            if image_files:
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
        """Toggle blend mode"""
        self.player.set_blend_mode(enabled)
        
        # Update UI state
        self.blend_frames_spin.setEnabled(enabled)
        self.blend_alpha_slider.setEnabled(enabled)
        
        if enabled:
            self.status_label.setText("Überblendmodus aktiviert")
            # Refresh current frame
            if self.player.image_files:
                self.player.emit_blended_frame()
        else:
            self.status_label.setText("Überblendmodus deaktiviert")
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
