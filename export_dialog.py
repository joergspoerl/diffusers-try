"""
Simple Video Export Dialog
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QProgressBar
)
from PyQt6.QtCore import Qt


class SimpleVideoExportDialog(QDialog):
    """Simple and clean video export dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        
        self.setWindowTitle("🎬 Video Export")
        self.setFixedSize(380, 320)
        self.setModal(True)
        
        # Center on parent
        if parent:
            parent_geo = parent.geometry()
            x = parent_geo.x() + (parent_geo.width() - 380) // 2
            y = parent_geo.y() + (parent_geo.height() - 320) // 2
            self.move(x, y)
        
        self.init_ui()
        
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
        self.quality_combo.setCurrentText("youtube")
        row1.addWidget(self.quality_combo)
        row1.addStretch()
        grid.addLayout(row1)
        
        # Row 2: Scale + Codec
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Scale:"))
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["1x", "2x", "4x"])
        self.upscale_combo.setCurrentIndex(1)
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
        
        # Viewer settings
        self.copy_settings_check = QCheckBox("Viewer-Einstellungen übernehmen")
        self.copy_settings_check.setChecked(True)
        layout.addWidget(self.copy_settings_check)
        
        # Output file
        layout.addSpacing(10)
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Datei:"))
        self.path_edit = QLineEdit()
        self.path_edit.setText("video_export.mp4")
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
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Video speichern",
            self.path_edit.text(),
            "MP4 Videos (*.mp4);;Alle Dateien (*)"
        )
        if filename:
            self.path_edit.setText(filename)
            
    def start_export(self):
        """Start video export"""
        output_path = self.path_edit.text()
        if not output_path:
            return
            
        # Update status
        self.status_label.setText("Export wird gestartet...")
        self.export_btn.setEnabled(False)
        
        # Get settings
        fps = self.fps_spin.value()
        quality = self.quality_combo.currentText()
        upscale = self.upscale_combo.currentText()
        codec = self.codec_combo.currentText()
        
        print(f"Starting export: {output_path}")
        print(f"Settings: {fps} FPS, {quality}, {upscale}, {codec}")
        
        # For now, just simulate progress
        import threading
        import time
        
        def simulate_export():
            for i in range(101):
                self.progress_bar.setValue(i)
                self.status_label.setText(f"Exportiere... {i}%")
                time.sleep(0.02)
            
            self.status_label.setText("Export abgeschlossen!")
            self.export_btn.setEnabled(True)
            
        thread = threading.Thread(target=simulate_export)
        thread.daemon = True
        thread.start()
