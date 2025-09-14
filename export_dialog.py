"""
Simple Video Export Dialog with Qt-accelerated interpolation and async processing
"""

import os
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QProgressBar, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QImage


class InterpolationWorker(QThread):
    """Asynchroner Worker für Zwischenframe-Berechnung im Speicher"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(list)  # list of QPixmaps
    error = pyqtSignal(str)
    
    def __init__(self, image_files, folder_path, interp_steps, blend_intensity):
        super().__init__()
        self.image_files = image_files
        self.folder_path = folder_path
        self.interp_steps = interp_steps
        self.blend_intensity = blend_intensity
        
    def run(self):
        """Führe Interpolation in separatem Thread aus - komplett im Speicher"""
        try:
            result = self.generate_interpolated_frames_in_memory()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Interpolation failed: {str(e)}")
    
    def generate_interpolated_frames_in_memory(self):
        """Qt-basierte Interpolation komplett im Speicher - keine Dateien"""
        from PyQt6.QtGui import QPixmap, QPainter
        from PyQt6.QtCore import Qt
        
        all_pixmaps = []
        total_work = len(self.image_files) + (len(self.image_files) - 1) * self.interp_steps
        current_work = 0
        
        for i in range(len(self.image_files)):
            # Add original frame
            original_pixmap = QPixmap(self.image_files[i])
            if not original_pixmap.isNull():
                all_pixmaps.append(original_pixmap)
            
            current_work += 1
            progress = int((current_work / total_work) * 100)
            self.progress.emit(progress)
            self.status.emit(f"🚀 Frame {i+1}/{len(self.image_files)}")
            
            # Generate interpolated frames (except for last image)
            if i < len(self.image_files) - 1:
                current_pixmap = QPixmap(self.image_files[i])
                next_pixmap = QPixmap(self.image_files[i + 1])
                
                if current_pixmap.isNull() or next_pixmap.isNull():
                    continue
                
                # Ensure consistent size
                if current_pixmap.size() != next_pixmap.size():
                    next_pixmap = next_pixmap.scaled(current_pixmap.size(), 
                                                   Qt.AspectRatioMode.IgnoreAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation)
                
                # Generate interpolation steps using Qt
                for step in range(1, self.interp_steps + 1):
                    # Calculate blend ratio
                    ratio = step / (self.interp_steps + 1)
                    effective_ratio = ratio * self.blend_intensity
                    
                    # Create interpolated frame using Qt
                    result = QPixmap(current_pixmap.size())
                    result.fill(Qt.GlobalColor.transparent)
                    
                    painter = QPainter(result)
                    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                    
                    # Draw frames with blending
                    painter.setOpacity(1.0 - effective_ratio)
                    painter.drawPixmap(0, 0, current_pixmap)
                    
                    painter.setOpacity(effective_ratio)
                    painter.drawPixmap(0, 0, next_pixmap)
                    
                    painter.end()
                    
                    # Keep interpolated frame in memory
                    all_pixmaps.append(result)
                    
                    current_work += 1
                    progress = int((current_work / total_work) * 100)
                    self.progress.emit(progress)
                    
                    # Allow GUI updates
                    self.msleep(1)  # Small delay to prevent UI freezing
        
        self.status.emit(f"✅ {len(all_pixmaps)} Frames im Speicher erstellt")
        return all_pixmaps
    
    def generate_interpolated_frames_qt(self):
        """Qt-basierte Interpolation mit echten Progress-Updates"""
        import os
        
        # Create temp folder for interpolated frames
        interp_folder = os.path.join(self.folder_path, "interpolated_export")
        os.makedirs(interp_folder, exist_ok=True)
        
        all_frames = []
        frame_counter = 1
        total_work = len(self.image_files) + (len(self.image_files) - 1) * self.interp_steps
        current_work = 0
        
        for i in range(len(self.image_files)):
            # Add original frame
            original_path = self.image_files[i]
            frame_path = os.path.join(interp_folder, f"frame_{frame_counter:06d}.png")
            
            # Load and save original frame
            original_pixmap = QPixmap(original_path)
            if not original_pixmap.isNull():
                original_pixmap.save(frame_path, "PNG")
                all_frames.append(frame_path)
                frame_counter += 1
            
            current_work += 1
            progress = int((current_work / total_work) * 100)
            self.progress.emit(progress)
            self.status.emit(f"🚀 Frame {i+1}/{len(self.image_files)}")
            
            # Generate interpolated frames (except for last image)
            if i < len(self.image_files) - 1:
                current_pixmap = QPixmap(self.image_files[i])
                next_pixmap = QPixmap(self.image_files[i + 1])
                
                if current_pixmap.isNull() or next_pixmap.isNull():
                    continue
                
                # Ensure consistent size
                if current_pixmap.size() != next_pixmap.size():
                    next_pixmap = next_pixmap.scaled(current_pixmap.size(), 
                                                   Qt.AspectRatioMode.IgnoreAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation)
                
                # Generate interpolation steps using Qt
                for step in range(1, self.interp_steps + 1):
                    # Calculate blend ratio
                    ratio = step / (self.interp_steps + 1)
                    effective_ratio = ratio * self.blend_intensity
                    
                    # Create interpolated frame using Qt
                    result = QPixmap(current_pixmap.size())
                    result.fill(Qt.GlobalColor.transparent)
                    
                    painter = QPainter(result)
                    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                    
                    # Draw frames with blending
                    painter.setOpacity(1.0 - effective_ratio)
                    painter.drawPixmap(0, 0, current_pixmap)
                    
                    painter.setOpacity(effective_ratio)
                    painter.drawPixmap(0, 0, next_pixmap)
                    
                    painter.end()
                    
                    # Save interpolated frame
                    interp_path = os.path.join(interp_folder, f"frame_{frame_counter:06d}.png")
                    result.save(interp_path, "PNG")
                    all_frames.append(interp_path)
                    frame_counter += 1
                    
                    current_work += 1
                    progress = int((current_work / total_work) * 100)
                    self.progress.emit(progress)
                    
                    # Allow GUI updates
                    self.msleep(1)  # Small delay to prevent UI freezing
        
        return all_frames
    
    def generate_interpolated_frames_pil(self):
        """PIL-basierte Interpolation als Fallback"""
        from PIL import Image
        import os
        
        # Similar implementation but with PIL
        # ... (implementation similar to Qt version but with PIL)
        pass


class SimpleVideoExportDialog(QDialog):
    """Simple and clean video export dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        
        self.setWindowTitle("🎬 Video Export")
        self.setFixedSize(400, 420)  # Increased height for interpolation controls
        self.setModal(True)
        
        # Center on parent
        if parent:
            parent_geo = parent.geometry()
            x = parent_geo.x() + (parent_geo.width() - 380) // 2
            y = parent_geo.y() + (parent_geo.height() - 320) // 2
            self.move(x, y)
        
        self.init_ui()
        self.setup_default_filename()
        self.sync_viewer_settings()  # Initial sync
        
    def setup_default_filename(self):
        """Setup default filename based on current folder"""
        if not self.parent_viewer or not hasattr(self.parent_viewer, 'current_folder'):
            return
            
        current_folder = self.parent_viewer.current_folder
        if not current_folder:
            return
            
        import os
        
        # Extract run_id from folder name (e.g., "20250914-105155-miniatur-magic-mushroom")
        folder_name = os.path.basename(current_folder)
        
        # Use folder name as base for video filename
        video_filename = f"{folder_name}.mp4"
        
        # Set default output path to current folder
        default_path = os.path.join(current_folder, video_filename)
        
        # Make sure we have a unique filename from the start
        unique_path = self.get_unique_filename(default_path)
        
        self.path_edit.setText(unique_path)
        print(f"📁 Default export path: {unique_path}")
        
    def sync_viewer_settings(self):
        """Sync settings from parent viewer"""
        if not self.copy_settings_check.isChecked():
            return
            
        if not self.parent_viewer:
            return
            
        # Sync FPS
        if hasattr(self.parent_viewer, 'fps_spin'):
            fps_value = self.parent_viewer.fps_spin.value()
            self.fps_spin.setValue(fps_value)
            print(f"📊 Synced FPS: {fps_value}")
            
        # Sync interpolation settings
        if hasattr(self.parent_viewer, 'enable_interpolation_check'):
            interp_enabled = self.parent_viewer.enable_interpolation_check.isChecked()
            self.use_interpolation_check.setChecked(interp_enabled)
            
        if hasattr(self.parent_viewer, 'interpolation_steps_spin'):
            interp_steps = self.parent_viewer.interpolation_steps_spin.value()
            self.interp_steps_spin.setValue(interp_steps)
            print(f"🔄 Synced interpolation steps: {interp_steps}")
            
        if hasattr(self.parent_viewer, 'blend_intensity_slider'):
            # Convert slider value (0-100) to percentage
            intensity = self.parent_viewer.blend_intensity_slider.value()
            self.intensity_spin.setValue(intensity)
            print(f"🎨 Synced blend intensity: {intensity}%")
            
    def get_unique_filename(self, filepath):
        """Generate unique filename by adding number if file exists"""
        import os
        
        if not os.path.exists(filepath):
            return filepath
            
        # Split path into directory, base name and extension
        directory = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        name, ext = os.path.splitext(basename)
        
        # Find unique filename
        counter = 1
        while True:
            new_name = f"{name}_{counter:02d}{ext}"
            new_path = os.path.join(directory, new_name)
            
            if not os.path.exists(new_path):
                print(f"📝 File exists, using: {new_name}")
                return new_path
                
            counter += 1
            
            # Safety limit to prevent infinite loop
            if counter > 99:
                import time
                timestamp = int(time.time())
                new_name = f"{name}_{timestamp}{ext}"
                new_path = os.path.join(directory, new_name)
                print(f"📝 Using timestamp filename: {new_name}")
                return new_path
        
    def init_ui(self):
        """Initialize simple UI"""
        # Dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QSpinBox, QComboBox, QLineEdit {
                background-color: #404040;
                border: 1px solid #666;
                border-radius: 3px;
                padding: 4px;
                color: #ffffff;
                min-height: 16px;
            }
            QProgressBar {
                border: 1px solid #666;
                border-radius: 3px;
                background-color: #404040;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QCheckBox {
                color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("🎬 Video Export")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Settings in compact grid
        grid = QVBoxLayout()
        grid.setSpacing(8)
        
        # Row 1: FPS + Quality
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(25)
        self.fps_spin.setFixedWidth(60)
        row1.addWidget(self.fps_spin)
        row1.addSpacing(20)
        row1.addWidget(QLabel("Qualität:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["youtube", "high", "medium", "fast"])
        self.quality_combo.setCurrentText("high")  # Default high
        row1.addWidget(self.quality_combo)
        row1.addStretch()
        grid.addLayout(row1)
        
        # Row 2: Scale + Codec
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Scale:"))
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["1x", "2x", "4x"])
        self.upscale_combo.setCurrentIndex(2)  # Default 4x
        self.upscale_combo.setFixedWidth(60)
        row2.addWidget(self.upscale_combo)
        row2.addSpacing(20)
        row2.addWidget(QLabel("Codec:"))
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["libx264", "libx265"])
        row2.addWidget(self.codec_combo)
        row2.addStretch()
        grid.addLayout(row2)
        
        layout.addLayout(grid)
        
        # Interpolation settings
        interp_layout = QVBoxLayout()
        interp_layout.setSpacing(6)
        
        self.use_interpolation_check = QCheckBox("Zwischenframes berechnen")
        self.use_interpolation_check.setChecked(True)  # Default enabled
        interp_layout.addWidget(self.use_interpolation_check)
        
        # Interpolation parameters in compact layout
        interp_params = QHBoxLayout()
        
        interp_params.addWidget(QLabel("Frames:"))
        self.interp_steps_spin = QSpinBox()
        self.interp_steps_spin.setRange(1, 50)
        self.interp_steps_spin.setValue(20)  # Default 20
        self.interp_steps_spin.setFixedWidth(50)
        interp_params.addWidget(self.interp_steps_spin)
        
        interp_params.addSpacing(15)
        interp_params.addWidget(QLabel("Intensität:"))
        self.intensity_spin = QSpinBox()
        self.intensity_spin.setRange(10, 100)
        self.intensity_spin.setValue(30)  # Default 30%
        self.intensity_spin.setSuffix("%")
        self.intensity_spin.setFixedWidth(60)
        interp_params.addWidget(self.intensity_spin)
        
        interp_params.addStretch()
        interp_layout.addLayout(interp_params)
        
        layout.addLayout(interp_layout)
        
        # Viewer settings
        self.copy_settings_check = QCheckBox("Viewer-Einstellungen übernehmen")
        self.copy_settings_check.setChecked(True)
        self.copy_settings_check.toggled.connect(self.sync_viewer_settings)
        layout.addWidget(self.copy_settings_check)
        
        # Output file
        layout.addSpacing(10)
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Datei:"))
        self.path_edit = QLineEdit()
        self.path_edit.setText("video_export.mp4")  # Will be updated in setup_default_filename
        output_layout.addWidget(self.path_edit)
        
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self.browse_output_path)
        output_layout.addWidget(browse_btn)
        layout.addLayout(output_layout)
        
        # Progress
        layout.addSpacing(10)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Bereit für Export")
        self.status_label.setStyleSheet("color: #90EE90; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        # Buttons
        layout.addSpacing(15)
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("🎬 Exportieren")
        self.export_btn.clicked.connect(self.start_export)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: 1px solid #45a049;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(self.export_btn)
        
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def browse_output_path(self):
        """Browse for output file"""
        # Get current folder as default directory
        current_dir = ""
        if self.parent_viewer and hasattr(self.parent_viewer, 'current_folder') and self.parent_viewer.current_folder:
            current_dir = self.parent_viewer.current_folder
        else:
            current_dir = os.path.dirname(self.path_edit.text()) if self.path_edit.text() else ""
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Video speichern",
            os.path.join(current_dir, os.path.basename(self.path_edit.text())),
            "MP4 Videos (*.mp4);;Alle Dateien (*)"
        )
        if filename:
            self.path_edit.setText(filename)
            
    def start_export(self):
        """Start video export"""
        output_path = self.path_edit.text()
        if not output_path:
            self.status_label.setText("❌ Kein Ausgabepfad angegeben!")
            return
            
        # Check if file exists and create unique filename
        output_path = self.get_unique_filename(output_path)
        self.path_edit.setText(output_path)  # Update UI with new filename
            
        # Get current folder from parent viewer
        if not self.parent_viewer or not hasattr(self.parent_viewer, 'current_folder'):
            self.status_label.setText("❌ Kein Ordner ausgewählt!")
            return
            
        current_folder = self.parent_viewer.current_folder
        if not current_folder:
            self.status_label.setText("❌ Kein Ordner ausgewählt!")
            return
            
        # Update status
        self.status_label.setText("🔍 Suche Bilder...")
        self.export_btn.setEnabled(False)
        
        # Get settings
        fps = self.fps_spin.value()
        quality = self.quality_combo.currentText()
        upscale_text = self.upscale_combo.currentText()
        upscale = self.parse_upscale_factor(upscale_text)
        codec = self.codec_combo.currentText()
        
        # Get interpolation settings
        use_interpolation = self.use_interpolation_check.isChecked()
        interp_steps = self.interp_steps_spin.value()
        blend_intensity = self.intensity_spin.value() / 100.0  # Convert to 0.0-1.0
        
        print(f"🎬 Starting video export: {output_path}")
        print(f"📁 Source folder: {current_folder}")
        print(f"⚙️ Settings: {fps} FPS, {quality}, {upscale}, {codec}")
        print(f"🔄 Interpolation: {use_interpolation}, Steps: {interp_steps}, Intensity: {blend_intensity}")
        
        # Start real export in background thread
        import threading
        
        def real_export():
            try:
                print("🚀 Starting real_export...")
                self.create_video_from_folder(
                    current_folder, output_path, fps, quality, upscale, codec,
                    use_interpolation, interp_steps, blend_intensity
                )
                print("✅ Export completed successfully")
            except ImportError as e:
                error_msg = f"❌ Import-Fehler: {str(e)}"
                print(error_msg)
                if "numpy" in str(e).lower():
                    error_msg += " (NumPy-Kompatibilitätsproblem)"
                self.status_label.setText(error_msg)
                self.export_btn.setEnabled(True)
            except Exception as e:
                error_msg = f"❌ Export-Fehler: {str(e)}"
                print(error_msg)
                print(f"❌ Exception type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                self.status_label.setText(error_msg)
                self.export_btn.setEnabled(True)
                
        thread = threading.Thread(target=real_export)
        thread.daemon = True
        thread.start()
    def parse_upscale_factor(self, upscale_text):
        """Convert upscale text (e.g. '4x') to numeric factor - with error handling"""
        try:
            if isinstance(upscale_text, str):
                # Remove 'x' and convert to int
                factor = int(upscale_text.replace('x', ''))
                return factor
            elif isinstance(upscale_text, (int, float)):
                return int(upscale_text)
            else:
                print(f"⚠️ Unknown upscale format: {upscale_text}, defaulting to 1")
                return 1
        except (ValueError, TypeError) as e:
            print(f"⚠️ Error parsing upscale factor '{upscale_text}': {e}, defaulting to 1")
            return 1
    
    def create_video_in_memory(self, folder_path, output_path, fps, quality, upscale, codec,
                              use_interpolation=False, interp_steps=20, blend_intensity=0.3):
        """Ultra-fast video creation using in-memory processing with Qt"""
        import os
        import glob
        import subprocess
        import numpy as np
        from PyQt6.QtGui import QPixmap, QPainter, QImage
        from PyQt6.QtCore import Qt
        
        # Find PNG files
        png_pattern = os.path.join(folder_path, "*.png")
        image_files = sorted(glob.glob(png_pattern))
        
        if not image_files:
            self.status_label.setText(f"❌ Keine PNG-Dateien in {folder_path} gefunden!")
            self.export_btn.setEnabled(True)
            return
            
        self.status_label.setText(f"📸 {len(image_files)} Bilder gefunden")
        print(f"📸 Found {len(image_files)} images")
        
        # Try to find FFmpeg
        ffmpeg_path = self.find_ffmpeg()
        if not ffmpeg_path:
            self.status_label.setText("❌ FFmpeg nicht gefunden!")
            self.export_btn.setEnabled(True)
            return
            
        # Set up FFmpeg command for pipe input
        output_path = self.get_unique_filename(output_path)
        
        # Get video parameters
        first_pixmap = QPixmap(image_files[0])
        if first_pixmap.isNull():
            self.status_label.setText("❌ Erstes Bild konnte nicht geladen werden!")
            self.export_btn.setEnabled(True)
            return
            
        width = first_pixmap.width()
        height = first_pixmap.height()
        
        if upscale != 1:
            width = int(width * upscale)
            height = int(height * upscale)
        
        # FFmpeg command for pipe input
        cmd = [
            ffmpeg_path,
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # Read from pipe
            '-c:v', codec,
        ]
        
        # Add quality settings
        if quality == "high":
            cmd.extend(['-crf', '18'])
        elif quality == "medium":
            cmd.extend(['-crf', '23'])
        else:  # low
            cmd.extend(['-crf', '28'])
            
        cmd.append(output_path)
        
        # Start FFmpeg process
        self.status_label.setText("🚀 Starte FFmpeg-Pipeline...")
        ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        
        try:
            frame_count = 0
            total_frames = len(image_files)
            
            # Process frames with optional interpolation
            for i in range(len(image_files)):
                # Load current frame
                current_pixmap = QPixmap(image_files[i])
                if current_pixmap.isNull():
                    continue
                    
                # Scale if needed
                if upscale != 1:
                    current_pixmap = current_pixmap.scaled(width, height, 
                                                         Qt.AspectRatioMode.IgnoreAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation)
                
                # Convert to numpy array and send to FFmpeg
                frame_data = self.qpixmap_to_numpy(current_pixmap)
                ffmpeg_process.stdin.write(frame_data.tobytes())
                frame_count += 1
                
                # Generate interpolated frames if requested
                if use_interpolation and i < len(image_files) - 1:
                    next_pixmap = QPixmap(image_files[i + 1])
                    if not next_pixmap.isNull():
                        if upscale != 1:
                            next_pixmap = next_pixmap.scaled(width, height,
                                                           Qt.AspectRatioMode.IgnoreAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation)
                        
                        # Generate interpolated frames in memory
                        for step in range(1, interp_steps + 1):
                            ratio = step / (interp_steps + 1)
                            effective_ratio = ratio * blend_intensity
                            
                            # Create interpolated frame using Qt
                            result = QPixmap(current_pixmap.size())
                            result.fill(Qt.GlobalColor.transparent)
                            
                            painter = QPainter(result)
                            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                            
                            # Blend frames
                            painter.setOpacity(1.0 - effective_ratio)
                            painter.drawPixmap(0, 0, current_pixmap)
                            
                            painter.setOpacity(effective_ratio)
                            painter.drawPixmap(0, 0, next_pixmap)
                            
                            painter.end()
                            
                            # Convert to numpy and send to FFmpeg
                            interp_data = self.qpixmap_to_numpy(result)
                            ffmpeg_process.stdin.write(interp_data.tobytes())
                            frame_count += 1
                
                # Update progress
                total_frames_with_interp = len(image_files) + (len(image_files) - 1) * interp_steps if use_interpolation else len(image_files)
                progress = int((frame_count / total_frames_with_interp) * 80) + 10  # 10-90% range
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"🚀 In-Memory Export... {frame_count}/{total_frames_with_interp}")
            
            # Close FFmpeg input and wait for completion
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            
            if ffmpeg_process.returncode == 0:
                self.progress_bar.setValue(100)
                self.status_label.setText(f"✅ Video erfolgreich erstellt: {os.path.basename(output_path)}")
                print(f"✅ Video created successfully: {output_path}")
            else:
                stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                self.status_label.setText("❌ FFmpeg-Fehler beim Export!")
                print(f"❌ FFmpeg error: {stderr_output}")
                
        except Exception as e:
            ffmpeg_process.terminate()
            self.status_label.setText(f"❌ Fehler beim In-Memory Export: {str(e)}")
            print(f"❌ In-memory export error: {e}")
            
        finally:
            self.export_btn.setEnabled(True)
    
    def qpixmap_to_numpy(self, pixmap):
        """Convert QPixmap to numpy array for FFmpeg"""
        # Convert QPixmap to QImage
        image = pixmap.toImage()
        
        # Convert to RGB format
        if image.format() != QImage.Format.Format_RGB888:
            image = image.convertToFormat(QImage.Format.Format_RGB888)
        
        # Get image data
        ptr = image.constBits()
        ptr.setsize(image.sizeInBytes())
        
        # Create numpy array
        arr = np.frombuffer(ptr, np.uint8).reshape((image.height(), image.width(), 3))
        
        return arr
        
    def create_video_from_folder(self, folder_path, output_path, fps, quality, upscale, codec, 
                                use_interpolation=False, interp_steps=20, blend_intensity=0.3):
        """Create video from image folder - smart selection between methods"""
        # For now, always use legacy method to avoid NumPy issues
        # TODO: Re-enable in-memory method once NumPy is fixed
        print("🔄 Using legacy file-based export method")
        self.create_video_from_folder_legacy(folder_path, output_path, fps, quality, upscale, codec,
                                           use_interpolation, interp_steps, blend_intensity)
    
    def create_video_from_folder_legacy(self, folder_path, output_path, fps, quality, upscale, codec, 
                                use_interpolation=False, interp_steps=20, blend_intensity=0.3):
        """Legacy video creation using temporary files"""
        import os
        import glob
        import subprocess
        
        # Find PNG files
        png_pattern = os.path.join(folder_path, "*.png")
        image_files = sorted(glob.glob(png_pattern))
        
        if not image_files:
            self.status_label.setText(f"❌ Keine PNG-Dateien in {folder_path} gefunden!")
            self.export_btn.setEnabled(True)
            return
            
        self.status_label.setText(f"📸 {len(image_files)} Bilder gefunden")
        print(f"📸 Found {len(image_files)} images")
        
        # Store parameters for later use
        self.pending_video_params = {
            'folder_path': folder_path,
            'output_path': output_path,
            'fps': fps,
            'quality': quality,
            'upscale': upscale,
            'codec': codec,
            'image_files': image_files
        }
        
        # Generate interpolated frames if requested
        if use_interpolation and len(image_files) > 1:
            self.status_label.setText("🔄 Generiere Zwischenframes...")
            self.start_interpolation(image_files, folder_path, interp_steps, blend_intensity)
            return  # Processing continues in on_interpolation_finished
        else:
            # No interpolation needed, use in-memory processing with original files
            from PyQt6.QtGui import QPixmap
            pixmaps_list = []
            for img_path in image_files:
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    pixmaps_list.append(pixmap)
            self.continue_video_creation_in_memory(pixmaps_list)
    
    def continue_video_creation(self, image_files):
        """Continue with video creation after interpolation (or without)"""
        import os
        import subprocess
        
        # Retrieve stored parameters
        params = self.pending_video_params
        folder_path = params['folder_path']
        output_path = params['output_path']
        fps = params['fps']
        quality = params['quality']
        upscale = params['upscale']
        codec = params['codec']
        
        print(f"🔄 Continuing with {len(image_files)} total frames for video creation")
        
        # Try to find FFmpeg
        ffmpeg_path = self.find_ffmpeg()
        if not ffmpeg_path:
            self.status_label.setText("❌ FFmpeg nicht gefunden! Verwende OpenCV...")
            self.try_opencv_export(image_files, output_path, fps, upscale)
            return
        
        # Prepare FFmpeg command
        # Check if we have interpolated frames in subdirectory
        interp_folder = os.path.join(folder_path, "interpolated_export")
        
        if os.path.exists(interp_folder) and len(os.listdir(interp_folder)) > 0:
            # Use interpolated frames directory
            input_pattern = os.path.join(interp_folder, "frame_%06d.png")
            print(f"✅ Using interpolated frames: {input_pattern}")
        else:
            # Check if files in main folder follow frame_XXXXXX.png pattern
            frame_pattern_file = None
            for i in range(1, min(10, len(image_files) + 1)):
                test_file = os.path.join(folder_path, f"frame_{i:06d}.png")
                if os.path.exists(test_file):
                    frame_pattern_file = test_file
                    break
            
            if frame_pattern_file:
                # Files already follow the correct pattern
                input_pattern = os.path.join(folder_path, "frame_%06d.png")
                print(f"✅ Using existing frame pattern: {input_pattern}")
            else:
                # Files have different naming, create symlinks or copy
                self.status_label.setText("🔄 Bereite Dateien vor...")
                temp_folder = os.path.join(folder_path, "temp_export")
                os.makedirs(temp_folder, exist_ok=True)
                
                for i, img_file in enumerate(image_files):
                    src = img_file
                    dst = os.path.join(temp_folder, f"frame_{i+1:06d}.png")
                    if os.path.exists(dst):
                        os.remove(dst)
                    # Copy file
                    import shutil
                    shutil.copy2(src, dst)
                    
                input_pattern = os.path.join(temp_folder, "frame_%06d.png")
                print(f"📁 Created temp files: {input_pattern}")
        
        # Build FFmpeg command
        cmd = [ffmpeg_path, "-y"]  # -y to overwrite output
        
        # Input
        cmd.extend(["-framerate", str(fps)])
        cmd.extend(["-i", input_pattern])
        
        # Upscaling
        upscale_factor = self.parse_upscale_factor(upscale) if isinstance(upscale, str) else upscale
        if upscale_factor != 1:
            cmd.extend(["-vf", f"scale=iw*{upscale_factor}:ih*{upscale_factor}:flags=lanczos"])
        
        # Video codec
        cmd.extend(["-c:v", codec])
        
        # Quality settings
        if quality == "youtube":
            cmd.extend(["-crf", "18", "-preset", "slow"])
        elif quality == "high":
            cmd.extend(["-crf", "20", "-preset", "medium"])
        elif quality == "medium":
            cmd.extend(["-crf", "23", "-preset", "fast"])
        elif quality == "fast":
            cmd.extend(["-crf", "28", "-preset", "faster"])
            
        # Pixel format for compatibility
        cmd.extend(["-pix_fmt", "yuv420p"])
        
        # Output
        cmd.append(output_path)
        
        print(f"🎬 FFmpeg command: {' '.join(cmd)}")
        
        try:
            # Update status
            self.status_label.setText("🎬 Erstelle Video...")
            self.progress_bar.setValue(0)
            
            # Run FFmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=os.path.dirname(output_path) if os.path.dirname(output_path) else None
            )
            
            # Monitor progress (simple version)
            total_frames = len(image_files)
            
            for i in range(101):
                if process.poll() is not None:
                    break
                self.progress_bar.setValue(i)
                self.status_label.setText(f"🎬 Exportiere... {i}%")
                import time
                time.sleep(0.1)
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.progress_bar.setValue(100)
                self.status_label.setText(f"✅ Video erstellt: {os.path.basename(output_path)}")
                print(f"✅ Video export successful: {output_path}")
                
                # Cleanup temp folder if created
                if 'temp_folder' in locals() and os.path.exists(temp_folder):
                    import shutil
                    shutil.rmtree(temp_folder)
                    
            else:
                self.status_label.setText(f"❌ FFmpeg Fehler: {stderr[:100]}")
                print(f"❌ FFmpeg error: {stderr}")
                
        except Exception as e:
            self.status_label.setText(f"❌ Export-Fehler: {str(e)}")
            print(f"❌ Export error: {e}")
            
        finally:
            self.export_btn.setEnabled(True)
    
    def on_interpolation_finished(self, pixmaps_list):
        """Called when interpolation worker finishes successfully"""
        print(f"✅ Interpolation completed with {len(pixmaps_list)} frames in memory")
        self.status_label.setText(f"✅ Interpolation abgeschlossen: {len(pixmaps_list)} Frames")
        
        # Continue with in-memory video creation
        self.continue_video_creation_in_memory(pixmaps_list)
    
    def continue_video_creation_in_memory(self, pixmaps_list):
        """Continue with video creation using in-memory pixmaps"""
        import subprocess
        import numpy as np
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import Qt
        
        # Retrieve stored parameters
        params = self.pending_video_params
        folder_path = params['folder_path']
        output_path = params['output_path']
        fps = params['fps']
        quality = params['quality']
        upscale = params['upscale']
        codec = params['codec']
        
        print(f"🔄 Creating video from {len(pixmaps_list)} in-memory frames")
        
        # Try to find FFmpeg
        ffmpeg_path = self.find_ffmpeg()
        if not ffmpeg_path:
            self.status_label.setText("❌ FFmpeg nicht gefunden!")
            self.export_btn.setEnabled(True)
            return
        
        # Set up FFmpeg command for pipe input
        output_path = self.get_unique_filename(output_path)
        cmd = [
            ffmpeg_path, 
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{pixmaps_list[0].width()}x{pixmaps_list[0].height()}',  # Use first frame size
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # Read from pipe
            '-c:v', codec,
        ]
        
        # Add quality settings
        if quality == "high":
            cmd.extend(['-crf', '18'])
        elif quality == "medium":
            cmd.extend(['-crf', '23'])
        else:  # low
            cmd.extend(['-crf', '28'])
            
        cmd.append(output_path)
        
        print(f"🎬 FFmpeg command: {' '.join(cmd)}")
        
        try:
            # Start FFmpeg process
            self.status_label.setText("🎬 Erstelle Video...")
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Send frames to FFmpeg
            for i, pixmap in enumerate(pixmaps_list):
                # Convert QPixmap to numpy array
                frame_array = self.qpixmap_to_numpy(pixmap)
                
                # Apply upscaling if needed
                if upscale != "1x":
                    upscale_factor = self.parse_upscale_factor(upscale)
                    if upscale_factor != 1:
                        frame_array = self.upscale_frame(frame_array, upscale_factor)
                
                # Send frame to FFmpeg
                process.stdin.write(frame_array.tobytes())
                
                # Update progress
                progress = int((i + 1) / len(pixmaps_list) * 100)
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"🎬 Frame {i+1}/{len(pixmaps_list)} ({progress}%)")
            
            # Close stdin and wait
            process.stdin.close()
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.progress_bar.setValue(100)
                self.status_label.setText(f"✅ Video erstellt: {os.path.basename(output_path)}")
                print(f"✅ Video export successful: {output_path}")
            else:
                self.status_label.setText(f"❌ FFmpeg Fehler: {stderr.decode()[:100]}")
                print(f"❌ FFmpeg error: {stderr.decode()}")
                
        except Exception as e:
            self.status_label.setText(f"❌ Export-Fehler: {str(e)}")
            print(f"❌ Export error: {e}")
            
        finally:
            self.export_btn.setEnabled(True)
    
    def on_interpolation_error(self, error_message):
        """Called when interpolation worker encounters an error"""
        print(f"❌ Interpolation error: {error_message}")
        self.status_label.setText(f"❌ Interpolation-Fehler: {error_message}")
        self.export_btn.setEnabled(True)
    
    def start_interpolation(self, image_files, folder_path, interp_steps, blend_intensity):
        """Start the interpolation worker thread"""
        self.interpolation_worker = InterpolationWorker(
            image_files, folder_path, interp_steps, blend_intensity
        )
        self.interpolation_worker.progress.connect(self.update_progress)
        self.interpolation_worker.status.connect(self.update_status)
        self.interpolation_worker.finished.connect(self.on_interpolation_finished)
        self.interpolation_worker.error.connect(self.on_interpolation_error)
        self.interpolation_worker.start()
    
    def update_progress(self, value):
        """Update progress bar from worker thread"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status label from worker thread"""
        self.status_label.setText(message)
            
    def generate_interpolated_frames_qt(self, image_files, folder_path, interp_steps, blend_intensity):
        """Generate interpolated frames using Qt (GPU-accelerated) - much faster than PIL"""
        from PyQt6.QtGui import QPixmap, QPainter
        from PyQt6.QtCore import Qt
        import os
        
        # Create temp folder for interpolated frames
        interp_folder = os.path.join(folder_path, "interpolated_export")
        os.makedirs(interp_folder, exist_ok=True)
        
        all_frames = []
        frame_counter = 1
        
        for i in range(len(image_files)):
            # Add original frame
            original_path = image_files[i]
            frame_path = os.path.join(interp_folder, f"frame_{frame_counter:06d}.png")
            
            # Load and save original frame using Qt (faster than shutil.copy2)
            original_pixmap = QPixmap(original_path)
            if not original_pixmap.isNull():
                original_pixmap.save(frame_path, "PNG")
                all_frames.append(frame_path)
                frame_counter += 1
            
            # Generate interpolated frames (except for last image)
            if i < len(image_files) - 1:
                current_pixmap = QPixmap(image_files[i])
                next_pixmap = QPixmap(image_files[i + 1])
                
                if current_pixmap.isNull() or next_pixmap.isNull():
                    continue
                
                # Ensure consistent size
                if current_pixmap.size() != next_pixmap.size():
                    next_pixmap = next_pixmap.scaled(current_pixmap.size(), 
                                                   Qt.AspectRatioMode.IgnoreAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation)
                
                # Generate interpolation steps using Qt's optimized blending
                for step in range(1, interp_steps + 1):
                    # Calculate blend ratio
                    ratio = step / (interp_steps + 1)
                    
                    # Apply blend intensity
                    effective_ratio = ratio * blend_intensity
                    
                    # Create interpolated frame using Qt (GPU-accelerated)
                    result = QPixmap(current_pixmap.size())
                    result.fill(Qt.GlobalColor.transparent)
                    
                    painter = QPainter(result)
                    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                    
                    # Draw current frame (base layer)
                    painter.setOpacity(1.0 - effective_ratio)
                    painter.drawPixmap(0, 0, current_pixmap)
                    
                    # Draw next frame (overlay with interpolated alpha)
                    painter.setOpacity(effective_ratio)
                    painter.drawPixmap(0, 0, next_pixmap)
                    
                    painter.end()
                    
                    # Save interpolated frame
                    interp_path = os.path.join(interp_folder, f"frame_{frame_counter:06d}.png")
                    result.save(interp_path, "PNG")
                    all_frames.append(interp_path)
                    frame_counter += 1
                    
                # Update progress
                progress = int(((i + 1) / len(image_files)) * 30)  # 30% for interpolation
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"🚀 Qt-Interpolation... {i+1}/{len(image_files)}")
        
        print(f"🚀 Generated {len(all_frames)} total frames using Qt (GPU-accelerated)")
        return all_frames

    def generate_interpolated_frames_legacy(self, image_files, folder_path, interp_steps, blend_intensity):
        """Legacy PIL-based interpolation (slower but fallback)"""
        from PIL import Image, ImageChops
        import os
        
        # Create temp folder for interpolated frames
        interp_folder = os.path.join(folder_path, "interpolated_export")
        os.makedirs(interp_folder, exist_ok=True)
        
        all_frames = []
        frame_counter = 1
        
        for i in range(len(image_files)):
            # Add original frame
            original_path = image_files[i]
            frame_path = os.path.join(interp_folder, f"frame_{frame_counter:06d}.png")
            
            # Copy original frame
            import shutil
            shutil.copy2(original_path, frame_path)
            all_frames.append(frame_path)
            frame_counter += 1
            
            # Generate interpolated frames (except for last image)
            if i < len(image_files) - 1:
                current_img = Image.open(image_files[i]).convert('RGBA')
                next_img = Image.open(image_files[i + 1]).convert('RGBA')
                
                # Generate interpolation steps
                for step in range(1, interp_steps + 1):
                    # Calculate blend ratio
                    ratio = step / (interp_steps + 1)
                    
                    # Apply blend intensity
                    effective_ratio = ratio * blend_intensity
                    
                    # Create blended frame
                    blended = Image.blend(current_img, next_img, effective_ratio)
                    
                    # Save interpolated frame
                    interp_path = os.path.join(interp_folder, f"frame_{frame_counter:06d}.png")
                    blended.convert('RGB').save(interp_path)
                    all_frames.append(interp_path)
                    frame_counter += 1
                    
                # Update progress
                progress = int(((i + 1) / len(image_files)) * 30)  # 30% for interpolation
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"🔄 PIL-Interpolation... {i+1}/{len(image_files)}")
        
        print(f"🔄 Generated {len(all_frames)} total frames using PIL (legacy)")
        return all_frames
    
    def generate_interpolated_frames(self, image_files, folder_path, interp_steps, blend_intensity):
        """Smart interpolation with Qt primary and PIL fallback"""
        try:
            # Try Qt-based interpolation first (much faster)
            return self.generate_interpolated_frames_qt(image_files, folder_path, interp_steps, blend_intensity)
        except Exception as e:
            print(f"⚠️ Qt interpolation failed: {e}")
            print("🔄 Falling back to PIL interpolation...")
            # Fallback to PIL-based interpolation
            return self.generate_interpolated_frames_legacy(image_files, folder_path, interp_steps, blend_intensity)
        return all_frames
            
    def find_ffmpeg(self):
        """Try to find FFmpeg executable - prioritize local installation"""
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
        ]
        
        for path in possible_paths:
            # Check if file exists directly (for absolute paths)
            if os.path.isfile(path):
                print(f"✅ Found local FFmpeg: {path}")
                return path
            # Check if command exists in PATH (for relative commands)
            elif shutil.which(path):
                found_path = shutil.which(path)
                print(f"✅ Found system FFmpeg: {found_path}")
                return found_path
                
        print("❌ FFmpeg not found in any location")
        print(f"🔍 Searched in project: {project_root}")
        print("💡 Tipp: FFmpeg sollte unter ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ liegen")
        return None
        
    def try_opencv_export(self, image_files, output_path, fps, upscale):
        """Fallback: Create video using OpenCV if available"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            self.status_label.setText("🔄 Verwende OpenCV für Video-Export...")
            
            # Read first image to get dimensions
            first_img = Image.open(image_files[0])
            width, height = first_img.size
            
            # Apply upscaling
            upscale_factor = self.parse_upscale_factor(upscale) if isinstance(upscale, str) else upscale
            if upscale_factor != 1:
                width *= upscale_factor
                height *= upscale_factor
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.status_label.setText("❌ Kann Video-Datei nicht erstellen")
                self.export_btn.setEnabled(True)
                return
            
            total_frames = len(image_files)
            base_progress = 30 if hasattr(self, '_interpolated') else 0  # Account for interpolation progress
            
            for i, img_path in enumerate(image_files):
                # Update progress
                progress = base_progress + int(((i / total_frames) * (100 - base_progress)))
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"🎬 OpenCV Export... {progress}%")
                
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                
                # Apply upscaling if needed
                upscale_factor = self.parse_upscale_factor(upscale) if isinstance(upscale, str) else upscale
                if upscale_factor != 1:
                    new_size = (img.width * upscale_factor, img.height * upscale_factor)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert PIL to OpenCV format
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(img_bgr)
            
            out.release()
            
            self.progress_bar.setValue(100)
            self.status_label.setText(f"✅ Video erstellt (OpenCV): {os.path.basename(output_path)}")
            print(f"✅ OpenCV video export successful: {output_path}")
            
            # Cleanup interpolated frames if they exist
            if hasattr(self, '_interpolated_folder'):
                import shutil
                if os.path.exists(self._interpolated_folder):
                    shutil.rmtree(self._interpolated_folder)
                    print(f"🧹 Cleaned up interpolated frames: {self._interpolated_folder}")
            
        except ImportError:
            self.status_label.setText("❌ Weder FFmpeg noch OpenCV verfügbar!")
            print("❌ Neither FFmpeg nor OpenCV available")
        except Exception as e:
            self.status_label.setText(f"❌ OpenCV Export-Fehler: {str(e)}")
            print(f"❌ OpenCV export error: {e}")
        finally:
            self.export_btn.setEnabled(True)
