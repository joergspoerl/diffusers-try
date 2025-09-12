#!/usr/bin/env python3
"""
Diffusers PyQt6 UI - Professional Parameter Control Interface
"""

import sys
import json
import os
from typing import Dict, Any, List
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QLineEdit, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QTextEdit, QGroupBox, QScrollArea,
    QSplitter, QFrame, QFileDialog, QMessageBox, QProgressBar, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor


class ImagePreviewWidget(QLabel):
    """Central image preview widget with scaling"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(512, 512)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                background-color: #2b2b2b;
                color: #cccccc;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Vorschaubild\n\nKlicken Sie 'Preview Single Frame'\nfür einen schnellen Test\n\noder 'Generate Images'\nfür die vollständige Generierung")
        self.setScaledContents(True)
        self.current_image_path = None
        
    def set_image(self, image_path: str):
        """Set preview image with proper scaling"""
        if os.path.exists(image_path):
            self.current_image_path = image_path
            pixmap = QPixmap(image_path)
            
            # Scale pixmap to fit widget while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.setPixmap(scaled_pixmap)
            
    def mousePressEvent(self, event):
        """Open image in system viewer on click"""
        if self.current_image_path and os.path.exists(self.current_image_path):
            try:
                import subprocess
                subprocess.run(['start', self.current_image_path], shell=True)
            except:
                pass
        super().mousePressEvent(event)


class ParameterGroup(QGroupBox):
    """Reusable parameter group widget"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #5a5a5a;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        self.layout = QGridLayout(self)
        self.row = 0
        
    def add_slider(self, label: str, min_val: float, max_val: float, 
                   value: float, decimals: int = 0) -> QSlider:
        """Add labeled slider with value display"""
        lbl = QLabel(label)
        slider = QSlider(Qt.Orientation.Horizontal)
        
        if decimals > 0:
            multiplier = 10 ** decimals
            slider.setMinimum(int(min_val * multiplier))
            slider.setMaximum(int(max_val * multiplier))
            slider.setValue(int(value * multiplier))
        else:
            slider.setMinimum(int(min_val))
            slider.setMaximum(int(max_val))
            slider.setValue(int(value))
            
        value_lbl = QLabel(str(value))
        value_lbl.setMinimumWidth(60)
        value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_lbl.setStyleSheet("QLabel { border: 1px solid #666; padding: 2px; }")
        
        def update_label():
            if decimals > 0:
                val = slider.value() / (10 ** decimals)
                value_lbl.setText(f"{val:.{decimals}f}")
            else:
                value_lbl.setText(str(slider.value()))
                
        slider.valueChanged.connect(update_label)
        
        self.layout.addWidget(lbl, self.row, 0)
        self.layout.addWidget(slider, self.row, 1)
        self.layout.addWidget(value_lbl, self.row, 2)
        self.row += 1
        
        return slider
        
    def add_spinbox(self, label: str, min_val: int, max_val: int, value: int) -> QSpinBox:
        """Add labeled spinbox"""
        lbl = QLabel(label)
        spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(value)
        
        self.layout.addWidget(lbl, self.row, 0)
        self.layout.addWidget(spinbox, self.row, 1, 1, 2)
        self.row += 1
        
        return spinbox
        
    def add_double_spinbox(self, label: str, min_val: float, max_val: float, 
                          value: float, decimals: int = 2) -> QDoubleSpinBox:
        """Add labeled double spinbox"""
        lbl = QLabel(label)
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(value)
        spinbox.setDecimals(decimals)
        spinbox.setSingleStep(0.1 if decimals > 0 else 1)
        
        self.layout.addWidget(lbl, self.row, 0)
        self.layout.addWidget(spinbox, self.row, 1, 1, 2)
        self.row += 1
        
        return spinbox
        
    def add_combobox(self, label: str, items: List[str], current: str) -> QComboBox:
        """Add labeled combobox"""
        lbl = QLabel(label)
        combo = QComboBox()
        combo.addItems(items)
        if current in items:
            combo.setCurrentText(current)
            
        self.layout.addWidget(lbl, self.row, 0)
        self.layout.addWidget(combo, self.row, 1, 1, 2)
        self.row += 1
        
        return combo
        
    def add_checkbox(self, label: str, checked: bool) -> QCheckBox:
        """Add checkbox"""
        checkbox = QCheckBox(label)
        checkbox.setChecked(checked)
        
        self.layout.addWidget(checkbox, self.row, 0, 1, 3)
        self.row += 1
        
        return checkbox
        
    def add_lineedit(self, label: str, text: str) -> QLineEdit:
        """Add labeled line edit"""
        lbl = QLabel(label)
        lineedit = QLineEdit(text)
        
        self.layout.addWidget(lbl, self.row, 0)
        self.layout.addWidget(lineedit, self.row, 1, 1, 2)
        self.row += 1
        
        return lineedit


class PreviewThread(QThread):
    """Background thread for single frame preview"""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    output = pyqtSignal(str)
    
    def __init__(self, config_data: Dict[str, Any]):
        super().__init__()
        self.config_data = config_data
        
    def run(self):
        """Run preview generation process"""
        try:
            import subprocess
            import tempfile
            import glob
            
            # Create preview config (single frame, first prompt only)
            preview_config = self.config_data.copy()
            
            # Modify for single frame preview
            if preview_config.get("morph_prompts"):
                # Use first prompt for preview
                preview_config["prompt"] = preview_config["morph_prompts"][0]
                preview_config["morph_prompts"] = None
                preview_config["morph_frames"] = 0
            
            # Single image settings
            preview_config["images"] = 1
            preview_config["video"] = False
            preview_config["morph_latent"] = False
            preview_config["seed_cycle"] = 0
            preview_config["interp_frames"] = 0
            
            # Save preview config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(preview_config, f, indent=2)
                config_path = f.name
                
            self.output.emit(f"Starting preview generation...")
            self.progress.emit(10)
            
            # Build command
            python_path = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
            if not os.path.exists(python_path):
                python_path = "python"
                
            cmd = [python_path, "generate.py", "--config", config_path]
            
            self.output.emit(f"Executing preview: {' '.join(cmd)}")
            self.progress.emit(20)
            
            # Run generation
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            
            output_dir = None
            
            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                self.output.emit(line.strip())
                
                # Extract output directory
                if "Run ID:" in line:
                    try:
                        run_id = line.split("Run ID:")[-1].strip()
                        output_dir = os.path.join(preview_config.get("outdir", "outputs"), run_id)
                    except:
                        pass
                
                # Update progress
                if "Generating" in line or "Saving" in line:
                    self.progress.emit(70)
                elif "Config saved" in line or "Dateien:" in line:
                    self.progress.emit(90)
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                self.progress.emit(100)
                
                # Find generated image
                if output_dir and os.path.exists(output_dir):
                    image_files = glob.glob(os.path.join(output_dir, "*.png"))
                    if image_files:
                        self.finished.emit(image_files[0])  # Return path to first image
                    else:
                        self.error.emit("No image files found in output directory")
                else:
                    self.error.emit("Output directory not found")
            else:
                self.error.emit(f"Preview generation failed with return code: {return_code}")
                
            # Clean up temp file
            try:
                os.unlink(config_path)
            except:
                pass
                
        except Exception as e:
            self.error.emit(str(e))


class GenerationThread(QThread):
    """Background thread for image generation"""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    output = pyqtSignal(str)
    
    def __init__(self, config_data: Dict[str, Any]):
        super().__init__()
        self.config_data = config_data
        
    def run(self):
        """Run generation process"""
        try:
            import subprocess
            import tempfile
            
            # Save config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(self.config_data, f, indent=2)
                config_path = f.name
                
            self.output.emit(f"Starting generation with config: {config_path}")
            self.progress.emit(10)
            
            # Build command
            python_path = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
            if not os.path.exists(python_path):
                python_path = "python"  # Fallback to system python
                
            cmd = [python_path, "generate.py", "--config", config_path]
            
            self.output.emit(f"Executing: {' '.join(cmd)}")
            self.progress.emit(20)
            
            # Run generation
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            
            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                self.output.emit(line.strip())
                
                # Update progress based on output
                if "Frame" in line and "/" in line:
                    try:
                        # Extract frame progress
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "Frame" in part and i+1 < len(parts):
                                frame_info = parts[i+1]
                                if "/" in frame_info:
                                    current, total = frame_info.split("/")
                                    progress = 20 + int((int(current) / int(total)) * 70)
                                    self.progress.emit(min(progress, 90))
                                    break
                    except:
                        pass
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                self.progress.emit(100)
                self.finished.emit("Generation completed successfully!")
            else:
                self.error.emit(f"Generation failed with return code: {return_code}")
                
            # Clean up temp file
            try:
                os.unlink(config_path)
            except:
                pass
                
        except Exception as e:
            self.error.emit(str(e))


class DiffusersUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.config_data = {}
        self.init_ui()
        self.load_default_config()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Diffusers Control Center - Professional AI Video Generation")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left side: Parameter controls
        self.create_parameter_panel(splitter)
        
        # Right side: Preview and actions
        self.create_preview_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([800, 600])
        
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #5a5a5a;
                padding: 5px;
                border-radius: 3px;
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
            QTabWidget::pane {
                border: 1px solid #5a5a5a;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4a6fa5;
            }
        """)
        
    def create_parameter_panel(self, parent):
        """Create left parameter control panel"""
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(800)
        
        # Parameter widget
        param_widget = QWidget()
        scroll.setWidget(param_widget)
        
        # Tab widget for organized parameters
        tab_widget = QTabWidget()
        param_layout = QVBoxLayout(param_widget)
        param_layout.addWidget(tab_widget)
        
        # Basic Parameters Tab
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        tab_widget.addTab(basic_tab, "Basic")
        
        # Basic Parameters Group
        basic_group = ParameterGroup("Basic Generation Parameters")
        basic_layout.addWidget(basic_group)
        
        self.prompt_edit = QLineEdit()
        basic_group.layout.addWidget(QLabel("Prompt"), basic_group.row, 0)
        basic_group.layout.addWidget(self.prompt_edit, basic_group.row, 1, 1, 2)
        basic_group.row += 1
        
        self.negative_edit = QLineEdit()
        basic_group.layout.addWidget(QLabel("Negative Prompt"), basic_group.row, 0)
        basic_group.layout.addWidget(self.negative_edit, basic_group.row, 1, 1, 2)
        basic_group.row += 1
        
        self.width_spin = basic_group.add_spinbox("Width", 256, 2048, 768)
        self.height_spin = basic_group.add_spinbox("Height", 256, 2048, 768)
        self.steps_spin = basic_group.add_spinbox("Steps", 1, 150, 25)
        self.guidance_spin = basic_group.add_double_spinbox("Guidance", 1.0, 20.0, 8.5, 1)
        self.images_spin = basic_group.add_spinbox("Images per Prompt", 1, 20, 1)
        self.seed_spin = basic_group.add_spinbox("Seed (0=Random)", 0, 999999, 0)
        self.outdir_edit = basic_group.add_lineedit("Output Directory", "outputs")
        
        # Model Selection Group
        model_group = ParameterGroup("Model Configuration")
        basic_layout.addWidget(model_group)
        
        models = [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "CompVis/stable-diffusion-v1-4"
        ]
        self.model_combo = model_group.add_combobox("Model", models, models[0])
        self.half_checkbox = model_group.add_checkbox("Half Precision", True)
        self.cpu_offload_checkbox = model_group.add_checkbox("CPU Offload", True)
        self.seq_offload_checkbox = model_group.add_checkbox("Sequential CPU Offload", False)
        self.no_slicing_checkbox = model_group.add_checkbox("Disable Attention Slicing", False)
        self.info_only_checkbox = model_group.add_checkbox("Info Only (No Generation)", False)
        
        # Morphing Parameters Tab
        morph_tab = QWidget()
        morph_layout = QVBoxLayout(morph_tab)
        tab_widget.addTab(morph_tab, "Morphing")
        
        # Legacy Morphing Group
        legacy_morph_group = ParameterGroup("Legacy Two-Point Morphing")
        morph_layout.addWidget(legacy_morph_group)
        
        self.morph_from_edit = legacy_morph_group.add_lineedit("Morph From", "")
        self.morph_to_edit = legacy_morph_group.add_lineedit("Morph To", "")
        self.morph_continuous_checkbox = legacy_morph_group.add_checkbox("Continuous Multi-Prompt Morph", False)
        
        # Morphing Settings Group
        morph_group = ParameterGroup("Multi-Prompt Morphing Configuration")
        morph_layout.addWidget(morph_group)
        
        self.morph_frames_spin = morph_group.add_spinbox("Frames", 10, 2000, 600)
        self.morph_seed_start_spin = morph_group.add_spinbox("Seed Start", 1, 999999, 1111)
        self.morph_seed_end_spin = morph_group.add_spinbox("Seed End", 1, 999999, 99999)
        
        ease_options = ["linear", "sine", "cubic", "expo", "ease", "ease-in", "ease-out", "quad"]
        self.morph_ease_combo = morph_group.add_combobox("Ease Function", ease_options, "sine")
        
        self.morph_latent_checkbox = morph_group.add_checkbox("Latent Morphing", True)
        self.morph_slerp_checkbox = morph_group.add_checkbox("SLERP Interpolation", True)
        self.morph_smooth_checkbox = morph_group.add_checkbox("Smooth Morphing", True)
        
        # Advanced Morphing Group
        adv_morph_group = ParameterGroup("Advanced Morphing Effects")
        morph_layout.addWidget(adv_morph_group)
        
        self.color_shift_checkbox = adv_morph_group.add_checkbox("Color Shift", True)
        self.color_intensity_slider = adv_morph_group.add_slider("Color Intensity", 0.0, 1.0, 0.25, 2)
        self.noise_pulse_slider = adv_morph_group.add_slider("Noise Pulse", 0.0, 0.1, 0.01, 3)
        self.frame_perturb_slider = adv_morph_group.add_slider("Frame Perturbation", 0.0, 0.1, 0.01, 3)
        self.temporal_blend_slider = adv_morph_group.add_slider("Temporal Blend", 0.0, 0.5, 0.02, 3)
        
        curve_options = ["linear", "center", "flat", "edges"]
        self.effect_curve_combo = adv_morph_group.add_combobox("Effect Curve", curve_options, "center")
        
        # Seed Cycling & Interpolation Tab
        seed_tab = QWidget()
        seed_layout = QVBoxLayout(seed_tab)
        tab_widget.addTab(seed_tab, "Seeds & Interpolation")
        
        # Seed Cycling Group
        seed_group = ParameterGroup("Seed Cycling")
        seed_layout.addWidget(seed_group)
        
        self.seed_cycle_spin = seed_group.add_spinbox("Seed Cycle Count", 0, 1000, 0)
        self.seed_step_spin = seed_group.add_spinbox("Seed Step Size", 1, 9999, 997)
        self.latent_jitter_slider = seed_group.add_slider("Latent Jitter", 0.0, 1.0, 0.0, 3)
        
        # Interpolation Group
        interp_group = ParameterGroup("Latent Interpolation")
        seed_layout.addWidget(interp_group)
        
        self.interp_seed_start_spin = interp_group.add_spinbox("Interpolation Seed Start", 0, 999999, 0)
        self.interp_seed_end_spin = interp_group.add_spinbox("Interpolation Seed End", 0, 999999, 0)
        self.interp_frames_spin = interp_group.add_spinbox("Interpolation Frames", 0, 1000, 0)
        self.interp_slerp_checkbox = interp_group.add_checkbox("Use SLERP for Interpolation", False)
        
        # Video Parameters Tab
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        tab_widget.addTab(video_tab, "Video")
        
        # Video Settings Group
        video_group = ParameterGroup("Video Generation")
        video_layout.addWidget(video_group)
        
        self.video_checkbox = video_group.add_checkbox("Generate Video", True)
        self.video_name_edit = video_group.add_lineedit("Custom Video Name", "")
        self.video_duration_spin = video_group.add_spinbox("Duration (seconds)", 10, 1800, 240)
        self.video_fps_spin = video_group.add_spinbox("Custom FPS (0=Auto)", 0, 120, 0)
        self.video_frames_spin = video_group.add_spinbox("Custom Frame Count (0=Auto)", 0, 10000, 0)
        self.video_blend_steps_spin = video_group.add_spinbox("Blend Steps", 1, 10, 4)
        
        blend_modes = ["none", "linear", "cubic", "smooth", "sharp", "flow"]
        self.video_blend_combo = video_group.add_combobox("Blend Mode", blend_modes, "linear")
        
        # Prompts Tab
        prompts_tab = QWidget()
        prompts_layout = QVBoxLayout(prompts_tab)
        tab_widget.addTab(prompts_tab, "Prompts")
        
        # Morph Prompts Group
        prompts_group = ParameterGroup("Morphing Prompts")
        prompts_layout.addWidget(prompts_group)
        
        self.prompts_text = QTextEdit()
        self.prompts_text.setMinimumHeight(400)
        prompts_group.layout.addWidget(QLabel("Prompts (one per line)"), 0, 0)
        prompts_group.layout.addWidget(self.prompts_text, 1, 0, 1, 3)
        
        parent.addWidget(scroll)
        
    def create_preview_panel(self, parent):
        """Create right preview and action panel"""
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        
        # Preview image
        self.preview_image = ImagePreviewWidget()
        preview_layout.addWidget(self.preview_image)
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        self.generate_btn = QPushButton("🎨 Generate Images")
        self.generate_btn.clicked.connect(self.start_generation)
        self.generate_btn.setToolTip("Start the complete image/video generation process")
        action_layout.addWidget(self.generate_btn)
        
        self.load_config_btn = QPushButton("📁 Load Config")
        self.load_config_btn.clicked.connect(self.load_config)
        self.load_config_btn.setToolTip("Load configuration from JSON file")
        action_layout.addWidget(self.load_config_btn)
        
        self.save_config_btn = QPushButton("💾 Save Config")
        self.save_config_btn.clicked.connect(self.save_config)
        self.save_config_btn.setToolTip("Save current settings to JSON file")
        action_layout.addWidget(self.save_config_btn)
        
        self.preview_btn = QPushButton("👁️ Preview Single Frame")
        self.preview_btn.clicked.connect(self.preview_frame)
        self.preview_btn.setToolTip("Generate a single frame for preview with current settings")
        action_layout.addWidget(self.preview_btn)
        
        self.quick_preview_btn = QPushButton("⚡ Quick Preview")
        self.quick_preview_btn.clicked.connect(self.quick_preview)
        self.quick_preview_btn.setToolTip("Fast preview with reduced quality settings (512x512, 10 steps)")
        action_layout.addWidget(self.quick_preview_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #90EE90; }")
        action_layout.addWidget(self.status_label)
        
        # Output log
        log_group = QGroupBox("Generation Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)
        
        action_layout.addWidget(log_group)
        
        # Preview history
        history_group = QGroupBox("Preview History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_list = QWidget()
        self.history_layout = QVBoxLayout(self.history_list)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        
        history_scroll = QScrollArea()
        history_scroll.setWidgetResizable(True)
        history_scroll.setMaximumHeight(150)
        history_scroll.setWidget(self.history_list)
        history_layout.addWidget(history_scroll)
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_preview_history)
        history_layout.addWidget(clear_history_btn)
        
        action_layout.addWidget(history_group)
        
        preview_layout.addWidget(action_group)
        
        parent.addWidget(preview_widget)
        
    def load_default_config(self):
        """Load default configuration"""
        try:
            with open("config-psy-004.json", 'r') as f:
                self.config_data = json.load(f)
            self.update_ui_from_config()
        except FileNotFoundError:
            self.status_label.setText("Default config not found")
            
    def update_ui_from_config(self):
        """Update UI elements from loaded config"""
        if not self.config_data:
            return
            
        # Basic parameters
        self.prompt_edit.setText(self.config_data.get("prompt", ""))
        self.negative_edit.setText(self.config_data.get("negative", ""))
        self.width_spin.setValue(self.config_data.get("width", 768))
        self.height_spin.setValue(self.config_data.get("height", 768))
        self.steps_spin.setValue(self.config_data.get("steps", 25))
        self.guidance_spin.setValue(self.config_data.get("guidance", 8.5))
        self.images_spin.setValue(self.config_data.get("images", 1))
        self.seed_spin.setValue(self.config_data.get("seed", 0))
        self.outdir_edit.setText(self.config_data.get("outdir", "outputs"))
        
        # Model settings
        self.model_combo.setCurrentText(self.config_data.get("model", ""))
        self.half_checkbox.setChecked(self.config_data.get("half", True))
        self.cpu_offload_checkbox.setChecked(self.config_data.get("cpu_offload", True))
        self.seq_offload_checkbox.setChecked(self.config_data.get("seq_offload", False))
        self.no_slicing_checkbox.setChecked(self.config_data.get("no_slicing", False))
        self.info_only_checkbox.setChecked(self.config_data.get("info_only", False))
        
        # Morphing parameters
        self.morph_from_edit.setText(self.config_data.get("morph_from", ""))
        self.morph_to_edit.setText(self.config_data.get("morph_to", ""))
        self.morph_continuous_checkbox.setChecked(self.config_data.get("morph_continuous", False))
        self.morph_frames_spin.setValue(self.config_data.get("morph_frames", 600))
        self.morph_seed_start_spin.setValue(self.config_data.get("morph_seed_start", 1111))
        self.morph_seed_end_spin.setValue(self.config_data.get("morph_seed_end", 99999))
        self.morph_ease_combo.setCurrentText(self.config_data.get("morph_ease", "sine"))
        
        # Checkboxes
        self.morph_latent_checkbox.setChecked(self.config_data.get("morph_latent", True))
        self.morph_slerp_checkbox.setChecked(self.config_data.get("morph_slerp", True))
        self.morph_smooth_checkbox.setChecked(self.config_data.get("morph_smooth", True))
        self.color_shift_checkbox.setChecked(self.config_data.get("morph_color_shift", True))
        
        # Sliders (multiply by appropriate factor for display)
        self.color_intensity_slider.setValue(int(self.config_data.get("morph_color_intensity", 0.25) * 100))
        self.noise_pulse_slider.setValue(int(self.config_data.get("morph_noise_pulse", 0.01) * 1000))
        self.frame_perturb_slider.setValue(int(self.config_data.get("morph_frame_perturb", 0.01) * 1000))
        self.temporal_blend_slider.setValue(int(self.config_data.get("morph_temporal_blend", 0.02) * 1000))
        
        # Seed cycling and interpolation
        self.seed_cycle_spin.setValue(self.config_data.get("seed_cycle", 0))
        self.seed_step_spin.setValue(self.config_data.get("seed_step", 997))
        self.latent_jitter_slider.setValue(int(self.config_data.get("latent_jitter", 0.0) * 1000))
        self.interp_seed_start_spin.setValue(self.config_data.get("interp_seed_start", 0))
        self.interp_seed_end_spin.setValue(self.config_data.get("interp_seed_end", 0))
        self.interp_frames_spin.setValue(self.config_data.get("interp_frames", 0))
        self.interp_slerp_checkbox.setChecked(self.config_data.get("interp_slerp", False))
        
        # Video settings
        self.video_checkbox.setChecked(self.config_data.get("video", True))
        self.video_name_edit.setText(self.config_data.get("video_name", ""))
        self.video_duration_spin.setValue(self.config_data.get("video_target_duration", 240))
        self.video_fps_spin.setValue(self.config_data.get("video_fps", 0))
        self.video_frames_spin.setValue(self.config_data.get("video_frames", 0))
        self.video_blend_steps_spin.setValue(self.config_data.get("video_blend_steps", 4))
        self.video_blend_combo.setCurrentText(self.config_data.get("video_blend_mode", "linear"))
        
        # Prompts
        prompts = self.config_data.get("morph_prompts", [])
        self.prompts_text.setPlainText("\n".join(prompts))
        
        self.status_label.setText("Config loaded successfully")
        
    def get_config_from_ui(self) -> Dict[str, Any]:
        """Extract configuration from UI elements"""
        config = {
            "prompt": self.prompt_edit.text(),
            "negative": self.negative_edit.text(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
            "images": self.images_spin.value(),
            "seed": self.seed_spin.value() if self.seed_spin.value() > 0 else None,
            "outdir": self.outdir_edit.text(),
            "model": self.model_combo.currentText(),
            "half": self.half_checkbox.isChecked(),
            "cpu_offload": self.cpu_offload_checkbox.isChecked(),
            "seq_offload": self.seq_offload_checkbox.isChecked(),
            "no_slicing": self.no_slicing_checkbox.isChecked(),
            "info_only": self.info_only_checkbox.isChecked(),
            # Legacy morphing
            "morph_from": self.morph_from_edit.text() if self.morph_from_edit.text() else None,
            "morph_to": self.morph_to_edit.text() if self.morph_to_edit.text() else None,
            "morph_continuous": self.morph_continuous_checkbox.isChecked(),
            # Multi-prompt morphing
            "morph_frames": self.morph_frames_spin.value(),
            "morph_seed_start": self.morph_seed_start_spin.value(),
            "morph_seed_end": self.morph_seed_end_spin.value(),
            "morph_ease": self.morph_ease_combo.currentText(),
            "morph_latent": self.morph_latent_checkbox.isChecked(),
            "morph_slerp": self.morph_slerp_checkbox.isChecked(),
            "morph_smooth": self.morph_smooth_checkbox.isChecked(),
            "morph_color_shift": self.color_shift_checkbox.isChecked(),
            "morph_color_intensity": self.color_intensity_slider.value() / 100.0,
            "morph_noise_pulse": self.noise_pulse_slider.value() / 1000.0,
            "morph_frame_perturb": self.frame_perturb_slider.value() / 1000.0,
            "morph_temporal_blend": self.temporal_blend_slider.value() / 1000.0,
            "morph_effect_curve": self.effect_curve_combo.currentText(),
            # Seed cycling and interpolation
            "seed_cycle": self.seed_cycle_spin.value(),
            "seed_step": self.seed_step_spin.value(),
            "latent_jitter": self.latent_jitter_slider.value() / 1000.0,
            "interp_seed_start": self.interp_seed_start_spin.value() if self.interp_seed_start_spin.value() > 0 else None,
            "interp_seed_end": self.interp_seed_end_spin.value() if self.interp_seed_end_spin.value() > 0 else None,
            "interp_frames": self.interp_frames_spin.value(),
            "interp_slerp": self.interp_slerp_checkbox.isChecked(),
            # Video
            "video": self.video_checkbox.isChecked(),
            "video_name": self.video_name_edit.text() if self.video_name_edit.text() else None,
            "video_target_duration": self.video_duration_spin.value(),
            "video_fps": self.video_fps_spin.value() if self.video_fps_spin.value() > 0 else 0,
            "video_frames": self.video_frames_spin.value() if self.video_frames_spin.value() > 0 else 0,
            "video_blend_steps": self.video_blend_steps_spin.value(),
            "video_blend_mode": self.video_blend_combo.currentText(),
            "morph_prompts": [line.strip() for line in self.prompts_text.toPlainText().split('\n') if line.strip()]
        }
        return config
        
    def start_generation(self):
        """Start image generation process"""
        config = self.get_config_from_ui()
        
        self.generate_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.quick_preview_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating...")
        
        # Start generation thread
        self.generation_thread = GenerationThread(config)
        self.generation_thread.progress.connect(self.progress_bar.setValue)
        self.generation_thread.finished.connect(self.generation_finished)
        self.generation_thread.error.connect(self.generation_error)
        self.generation_thread.output.connect(self.log_output)
        self.generation_thread.start()
        
    def log_output(self, message: str):
        """Log output message"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def generation_finished(self, message: str):
        """Handle generation completion"""
        self.generate_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.quick_preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        QMessageBox.information(self, "Success", message)
        
    def generation_error(self, error: str):
        """Handle generation error"""
        self.generate_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.quick_preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error}")
        QMessageBox.critical(self, "Error", f"Generation failed: {error}")
        
    def quick_preview(self):
        """Quick preview with reduced quality settings"""
        config = self.get_config_from_ui()
        
        # Validate basic settings
        if not config.get("prompt") and not config.get("morph_prompts"):
            QMessageBox.warning(self, "Warning", "Please enter a prompt or morph prompts for preview")
            return
            
        # Override with quick settings
        config["width"] = 512
        config["height"] = 512
        config["steps"] = 10
        config["guidance"] = 7.0
        
        self.quick_preview_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating quick preview...")
        self.log_text.append("\n=== QUICK PREVIEW (512x512, 10 steps) ===")
        
        # Start preview thread
        self.preview_thread = PreviewThread(config)
        self.preview_thread.progress.connect(self.progress_bar.setValue)
        self.preview_thread.finished.connect(self.quick_preview_finished)
        self.preview_thread.error.connect(self.preview_error)
        self.preview_thread.output.connect(self.log_output)
        self.preview_thread.start()
        
    def quick_preview_finished(self, image_path: str):
        """Handle quick preview completion"""
        self.quick_preview_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Quick preview completed!")
        
        # Load preview image
        if image_path and os.path.exists(image_path):
            self.preview_image.set_image(image_path)
            self.log_text.append(f"Quick preview saved: {image_path}")
            self.add_to_preview_history(image_path, "Quick Preview")
        else:
            QMessageBox.warning(self, "Preview Error", "Quick preview image not found")
            
    def preview_frame(self):
        """Preview a single frame with current settings"""
        config = self.get_config_from_ui()
        
        # Validate basic settings
        if not config.get("prompt") and not config.get("morph_prompts"):
            QMessageBox.warning(self, "Warning", "Please enter a prompt or morph prompts for preview")
            return
            
        self.preview_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating preview...")
        self.log_text.append("\n=== PREVIEW GENERATION ===")
        
        # Start preview thread
        self.preview_thread = PreviewThread(config)
        self.preview_thread.progress.connect(self.progress_bar.setValue)
        self.preview_thread.finished.connect(self.preview_finished)
        self.preview_thread.error.connect(self.preview_error)
        self.preview_thread.output.connect(self.log_output)
        self.preview_thread.start()
        
    def preview_finished(self, image_path: str):
        """Handle preview completion"""
        self.preview_btn.setEnabled(True)
        self.quick_preview_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Preview completed!")
        
        # Load preview image
        if image_path and os.path.exists(image_path):
            self.preview_image.set_image(image_path)
            self.log_text.append(f"Preview saved: {image_path}")
            self.add_to_preview_history(image_path, "Preview")
            QMessageBox.information(self, "Preview Complete", f"Preview generated successfully!\n\nSaved to: {image_path}")
        else:
            QMessageBox.warning(self, "Preview Error", "Preview image not found")
            
    def preview_error(self, error: str):
        """Handle preview error"""
        self.preview_btn.setEnabled(True)
        self.quick_preview_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Preview error: {error}")
        QMessageBox.critical(self, "Preview Error", f"Preview generation failed: {error}")
        
    def load_config(self):
        """Load configuration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.config_data = json.load(f)
                self.update_ui_from_config()
                self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {e}")
                
    def save_config(self):
        """Save current configuration to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "config.json", "JSON Files (*.json)"
        )
        if file_path:
            try:
                config = self.get_config_from_ui()
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.status_label.setText(f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
                
    def add_to_preview_history(self, image_path: str, preview_type: str):
        """Add image to preview history"""
        import datetime
        
        # Create history item
        history_item = QWidget()
        item_layout = QHBoxLayout(history_item)
        item_layout.setContentsMargins(5, 2, 5, 2)
        
        # Thumbnail
        thumbnail = QLabel()
        thumbnail.setFixedSize(40, 40)
        thumbnail.setScaledContents(True)
        thumbnail.setStyleSheet("border: 1px solid #666;")
        
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            thumbnail.setPixmap(pixmap.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio))
        
        # Info
        info_label = QLabel(f"{preview_type} - {datetime.datetime.now().strftime('%H:%M:%S')}")
        info_label.setStyleSheet("color: #cccccc;")
        
        # Load button
        load_btn = QPushButton("Load")
        load_btn.setMaximumWidth(50)
        load_btn.clicked.connect(lambda: self.preview_image.set_image(image_path))
        
        item_layout.addWidget(thumbnail)
        item_layout.addWidget(info_label)
        item_layout.addStretch()
        item_layout.addWidget(load_btn)
        
        # Add to history (newest first)
        self.history_layout.insertWidget(0, history_item)
        
        # Limit history to 10 items
        while self.history_layout.count() > 10:
            item = self.history_layout.takeAt(self.history_layout.count() - 1)
            if item.widget():
                item.widget().deleteLater()
                
    def clear_preview_history(self):
        """Clear preview history"""
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Diffusers Control Center")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = DiffusersUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
