# ===============================================================
# fam_srm_gui.py — FAM-SRM Virtual Spectral Engine GUI v3.2
# Professional GIS / Scientific UI (Tiled Engine + Graphon + Multiband)
#
# Features:
#   • Emoji sidebar navigation (🧪 📊 📁 ⚙️ 🛈 🖥️)
#   • Pages: Engine, Reports, Projects, Settings, About, Diagnostics
#   • Real-time diagnostic logging
#   • Dark / Light theme with instant switching
#   • Settings persisted in ~/.famsrm_settings.json
#   • Auto-refresh of Reports + Projects after engine run and on tab switch
#   • Integrated with fam_srm_orthomosaic_engine.process_orthomosaic
#
# Produced for: Gabriel Allit (Salt19 LLC)
# ===============================================================

import os
import sys
import json
import platform
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QToolButton, QStackedWidget, QMenuBar, QStatusBar, QFileDialog,
    QMessageBox, QProgressBar, QTextEdit, QPushButton, QLineEdit,
    QGroupBox, QCheckBox, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QAction, QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread

# Optional system info enhancement
try:
    import psutil
except ImportError:
    psutil = None

# Engine
from fam_srm_orthomosaic_engine import process_orthomosaic


# ===============================================================
# Settings Manager (JSON persistence)
# ===============================================================
class SettingsManager:
    CONFIG_PATH = os.path.expanduser("~/.famsrm_settings.json")

    DEFAULTS = {
        "theme": "dark",          # "dark" or "light"
        "default_input_dir": "",
        "default_output_dir": "",
        "gpu_enabled": False,     # reserved for future GPU integration
        "downscale_mode": "off",  # retained for compatibility; currently not used
        "tile_size": 2048,
        "tile_threshold": 4096,
    }

    @classmethod
    def load(cls) -> dict:
        if os.path.exists(cls.CONFIG_PATH):
            try:
                with open(cls.CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged = cls.DEFAULTS.copy()
                merged.update(data)
                return merged
            except Exception:
                return cls.DEFAULTS.copy()
        else:
            return cls.DEFAULTS.copy()

    @classmethod
    def save(cls, data: dict) -> None:
        try:
            with open(cls.CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")


# ===============================================================
# Theme Manager (instant dark/light switching)
# ===============================================================
class ThemeManager:
    DARK = {
        "bg": "#0B0B0B",
        "panel": "#111111",
        "text": "#E3E3E3",
        "accent": "#00E5FF",
        "accent_soft": "#42C0FB",
        "border": "#333333",
        "console_bg": "#000000",
        "console_text": "#0FFFCA",
    }

    LIGHT = {
        "bg": "#E5E5E5",
        "panel": "#FFFFFF",
        "text": "#222222",
        "accent": "#0078A0",
        "accent_soft": "#2894C5",
        "border": "#CCCCCC",
        "console_bg": "#F5F5F5",
        "console_text": "#003344",
    }

    @classmethod
    def get_theme(cls, name: str) -> dict:
        if name == "light":
            return cls.LIGHT
        return cls.DARK

    @classmethod
    def build_qss(cls, theme: dict) -> str:
        return f"""
        QWidget {{
            background-color: {theme['bg']};
            color: {theme['text']};
            font-family: 'Segoe UI';
            font-size: 12pt;
        }}
        QMainWindow {{
            background-color: {theme['bg']};
        }}
        QLineEdit {{
            background-color: {theme['panel']};
            border: 1px solid {theme['border']};
            border-radius: 4px;
            padding: 6px;
        }}
        QPushButton {{
            background-color: {theme['panel']};
            color: {theme['accent']};
            border: 1px solid {theme['accent']};
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {theme['border']};
        }}
        QToolButton {{
            background-color: transparent;
            border-radius: 8px;
            padding: 8px;
            color: {theme['text']};
            font-size: 20pt;
        }}
        QToolButton:hover {{
            background-color: {theme['panel']};
        }}
        QToolButton:checked {{
            background-color: {theme['panel']};
            border: 2px solid {theme['accent']};
            color: {theme['accent']};
        }}
        QGroupBox {{
            border: 1px solid {theme['border']};
            border-radius: 6px;
            margin-top: 12px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            color: {theme['accent_soft']};
        }}
        QProgressBar {{
            background-color: {theme['panel']};
            border: 1px solid {theme['border']};
            color: {theme['text']};
            height: 22px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {theme['accent']};
        }}
        QTextEdit {{
            background-color: {theme['console_bg']};
            color: {theme['console_text']};
            border-radius: 6px;
            border: 1px solid {theme['border']};
            font-family: Consolas, monospace;
            font-size: 11pt;
        }}
        QCheckBox {{
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid {theme['border']};
            background-color: {theme['panel']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {theme['accent']};
            border: 1px solid {theme['accent']};
        }}
        QListWidget {{
            background-color: {theme['panel']};
            border: 1px solid {theme['border']};
            border-radius: 4px;
        }}
        QListWidget::item:selected {{
            background-color: {theme['accent_soft']};
            color: {theme['panel']};
        }}
        """

    @classmethod
    def apply_theme(cls, app: QApplication, name: str) -> None:
        app.setStyleSheet(cls.build_qss(cls.get_theme(name)))


# ===============================================================
# Diagnostics Logger (global)
# ===============================================================
class DiagnosticsLogger(QObject):
    message = pyqtSignal(str)

    def log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        msg = f"[{ts}] {text}"
        self.message.emit(msg)
        print(msg)


LOGGER = DiagnosticsLogger()


# ===============================================================
# Utility: Thumbnail loader
# ===============================================================
def load_thumb(path: str, size: int = 200) -> QPixmap | None:
    if not path or not os.path.exists(path):
        return None
    img = QImage(path)
    if img.isNull():
        return None
    return QPixmap.fromImage(
        img.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                   Qt.TransformationMode.SmoothTransformation)
    )


# ===============================================================
# EngineWorker (QThread) — runs FAM-SRM pipeline
# ===============================================================
class EngineWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str, dict)  # out_prefix, out_dir, options
    error = pyqtSignal(str)

    def __init__(self, main_window: "MainWindow", input_path: str, output_dir: str, options: dict, parent=None):
        super().__init__(parent)
        self.main = main_window
        self.input_path = input_path
        self.output_dir = output_dir
        self.options = options

    def run(self):
        try:
            self.progress.emit(0)
            LOGGER.log("EngineWorker started.")

            if not os.path.isfile(self.input_path):
                raise RuntimeError(f"Input file does not exist: {self.input_path}")

            ext = os.path.splitext(self.input_path)[1].lower()
            if ext not in (".tif", ".tiff", ".geotiff", ".gtif", ".jpg", ".jpeg", ".png"):
                raise RuntimeError(f"Unsupported input extension: {ext}")

            if not os.path.isdir(self.output_dir):
                raise RuntimeError(f"Output directory does not exist: {self.output_dir}")

            base = os.path.splitext(os.path.basename(self.input_path))[0]
            out_prefix = os.path.join(self.output_dir, base)

            LOGGER.log(f"Processing input: {self.input_path}")
            LOGGER.log(f"Output prefix: {out_prefix}")

            # Downscale mode is currently logged but not used by engine (reserved)
            downscale_mode = self.main.settings.get("downscale_mode", "off")
            LOGGER.log(f"Using downscale_mode (reserved): {downscale_mode}")

            tile_size = int(self.main.settings.get("tile_size", 2048))
            tile_threshold = int(self.main.settings.get("tile_threshold", 4096))
            LOGGER.log(f"Tile size: {tile_size}, tile threshold: {tile_threshold}")

            self.progress.emit(10)

            outputs = process_orthomosaic(
                input_path=self.input_path,
                output_folder=self.output_dir,
                tile_size=tile_size,
                tile_threshold=tile_threshold,
            )

            self.progress.emit(90)
            LOGGER.log("EngineWorker finished successfully.")
            LOGGER.log(f"Engine outputs: {outputs}")
            self.finished.emit(out_prefix, self.output_dir, self.options)
            self.progress.emit(100)

        except Exception as e:
            msg = f"Engine error: {e}"
            LOGGER.log(msg)
            self.error.emit(msg)


# ===============================================================
# Pages
# ===============================================================

# ------------------- Engine Page -------------------------------
class EnginePage(QWidget):
    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Input
        row_in = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select RGB orthomosaic or photo (GeoTIFF/JPG/PNG)…")
        if self.main.settings.get("default_input_dir"):
            self.input_edit.setText(self.main.settings["default_input_dir"])
        btn_in = QPushButton("Browse Input…")
        btn_in.clicked.connect(self.pick_input)
        row_in.addWidget(self.input_edit)
        row_in.addWidget(btn_in)

        # Output
        row_out = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output folder for FAM-SRM deliverables…")
        if self.main.settings.get("default_output_dir"):
            self.output_edit.setText(self.main.settings["default_output_dir"])
        btn_out = QPushButton("Browse Output…")
        btn_out.clicked.connect(self.pick_output)
        row_out.addWidget(self.output_edit)
        row_out.addWidget(btn_out)

        # Options
        opt_group = QGroupBox("Processing Options")
        opt_layout = QHBoxLayout()
        self.chk_graphon = QCheckBox("Graphon Map")
        self.chk_graphon.setChecked(True)
        self.chk_fsi = QCheckBox("FSI Map")
        self.chk_fsi.setChecked(True)
        self.chk_multiband = QCheckBox("Multiband GeoTIFF")
        self.chk_multiband.setChecked(True)
        opt_layout.addWidget(self.chk_graphon)
        opt_layout.addWidget(self.chk_fsi)
        opt_layout.addWidget(self.chk_multiband)
        opt_layout.addStretch()
        opt_group.setLayout(opt_layout)

        # Run + progress
        self.btn_run = QPushButton("Run FAM-SRM Engine")
        self.btn_run.clicked.connect(self.start_engine)
        self.progress = QProgressBar()
        self.progress.setValue(0)

        layout.addLayout(row_in)
        layout.addLayout(row_out)
        layout.addWidget(opt_group)
        layout.addSpacing(10)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.progress)
        layout.addStretch()

        self.setLayout(layout)

    def pick_input(self):
        start_dir = self.main.settings.get("default_input_dir") or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            start_dir,
            "Images (*.tif *.tiff *.geotiff *.jpg *.jpeg *.png)"
        )
        if path:
            self.input_edit.setText(path)
            LOGGER.log(f"Selected input: {path}")

    def pick_output(self):
        start_dir = self.main.settings.get("default_output_dir") or ""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", start_dir)
        if folder:
            self.output_edit.setText(folder)
            LOGGER.log(f"Selected output folder: {folder}")

    def start_engine(self):
        inp = self.input_edit.text().strip()
        out = self.output_edit.text().strip()
        if not os.path.isfile(inp):
            QMessageBox.warning(self, "Invalid Input", "Input path does not exist or is not a file.")
            return
        if not os.path.isdir(out):
            QMessageBox.warning(self, "Invalid Output", "Output folder does not exist.")
            return

        ext = os.path.splitext(inp)[1].lower()
        if ext not in (".tif", ".tiff", ".geotiff", ".gtif", ".jpg", ".jpeg", ".png"):
            QMessageBox.warning(self, "Invalid Input", f"Unsupported input extension: {ext}")
            return

        opts = {
            "graphon": self.chk_graphon.isChecked(),
            "fsi": self.chk_fsi.isChecked(),
            "multiband": self.chk_multiband.isChecked(),
        }

        # Persist defaults
        self.main.settings["default_input_dir"] = os.path.dirname(inp)
        self.main.settings["default_output_dir"] = out
        SettingsManager.save(self.main.settings)

        self.main.start_engine(inp, out, opts)


# ------------------- Reports Page ------------------------------
class ReportsPage(QWidget):
    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main = main_window
        self.current_prefix = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        self.lbl_title = QLabel("No report loaded.")
        layout.addWidget(self.lbl_title)

        # Four previews: vNIR, NDVI_FAM, FSI, Graphon
        row = QHBoxLayout()
        self.lbl_vnir = QLabel("vNIR")
        self.img_vnir = QLabel()
        self.lbl_ndvi = QLabel("NDVI_FAM")
        self.img_ndvi = QLabel()
        self.lbl_fsi = QLabel("FSI")
        self.img_fsi = QLabel()
        self.lbl_graphon = QLabel("Graphon")
        self.img_graphon = QLabel()

        for lab in (self.lbl_vnir, self.lbl_ndvi, self.lbl_fsi, self.lbl_graphon):
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for img in (self.img_vnir, self.img_ndvi, self.img_fsi, self.img_graphon):
            img.setFixedSize(220, 220)
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        col1 = QVBoxLayout()
        col1.addWidget(self.lbl_vnir)
        col1.addWidget(self.img_vnir)
        col2 = QVBoxLayout()
        col2.addWidget(self.lbl_ndvi)
        col2.addWidget(self.img_ndvi)
        col3 = QVBoxLayout()
        col3.addWidget(self.lbl_fsi)
        col3.addWidget(self.img_fsi)
        col4 = QVBoxLayout()
        col4.addWidget(self.lbl_graphon)
        col4.addWidget(self.img_graphon)

        row.addLayout(col1)
        row.addLayout(col2)
        row.addLayout(col3)
        row.addLayout(col4)
        layout.addLayout(row)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_report_pdf = QPushButton("Open Analytics PDF")
        self.btn_open_report_pdf.clicked.connect(self.open_report_pdf)
        self.btn_open_tech_pdf = QPushButton("Open Tech Spec PDF")
        self.btn_open_tech_pdf.clicked.connect(self.open_tech_pdf)
        self.btn_open_multiband = QPushButton("Open Multiband GeoTIFF")
        self.btn_open_multiband.clicked.connect(self.open_multiband)

        btn_row.addWidget(self.btn_open_folder)
        btn_row.addWidget(self.btn_open_report_pdf)
        btn_row.addWidget(self.btn_open_tech_pdf)
        btn_row.addWidget(self.btn_open_multiband)
        layout.addLayout(btn_row)

        layout.addStretch()
        self.setLayout(layout)

    def refresh(self):
        """Auto-refresh using main.last_out_prefix."""
        if self.main.last_out_prefix:
            self.load_from_prefix(self.main.last_out_prefix)

    def load_from_prefix(self, prefix: str):
        self.current_prefix = prefix
        self.lbl_title.setText(f"Report: {prefix}")

        mapping = {
            self.img_vnir: f"{prefix}_vnir.png",
            self.img_ndvi: f"{prefix}_ndvi_fam.png",
            self.img_fsi: f"{prefix}_fsi.png",
            self.img_graphon: f"{prefix}_graphon.png",
        }
        for widget, path in mapping.items():
            pix = load_thumb(path)
            widget.setPixmap(pix if pix else QPixmap())

    def _get_out_dir(self) -> str | None:
        if not self.current_prefix:
            return None
        return os.path.dirname(self.current_prefix)

    def open_output_folder(self):
        out_dir = self._get_out_dir()
        if out_dir and os.path.isdir(out_dir):
            try:
                os.startfile(out_dir)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Cannot open folder:\n{e}")

    def open_report_pdf(self):
        if not self.current_prefix:
            return
        path = f"{self.current_prefix}_FAM_SRM_Report.pdf"
        if os.path.exists(path):
            try:
                os.startfile(path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Cannot open PDF:\n{e}")
        else:
            QMessageBox.information(self, "Missing", "Analytics PDF not found.")

    def open_tech_pdf(self):
        if not self.current_prefix:
            return
        path = f"{self.current_prefix}_FAM_SRM_TechSpec.pdf"
        if os.path.exists(path):
            try:
                os.startfile(path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Cannot open PDF:\n{e}")
        else:
            QMessageBox.information(self, "Missing", "Technical Spec PDF not found.")

    def open_multiband(self):
        if not self.current_prefix:
            return
        path = f"{self.current_prefix}_fam_srm_products.tif"
        if os.path.exists(path):
            try:
                os.startfile(path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Cannot open GeoTIFF:\n{e}")
        else:
            QMessageBox.information(self, "Missing", "Multiband GeoTIFF not found.")


# ------------------- Projects Page -----------------------------
class ProjectsPage(QWidget):
    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main = main_window
        self.root_dir = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        self.lbl_info = QLabel("Projects will be detected from output folder.")
        layout.addWidget(self.lbl_info)

        self.list_projects = QListWidget()
        self.list_projects.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_projects)

        layout.addStretch()
        self.setLayout(layout)

    def refresh(self):
        """Auto-refresh from main.last_out_dir or default output dir."""
        self.root_dir = self.main.last_out_dir or self.main.settings.get("default_output_dir", "")
        self.load_projects(self.root_dir)

    def load_projects(self, folder: str):
        self.list_projects.clear()
        if not folder or not os.path.isdir(folder):
            self.lbl_info.setText("No valid output folder to scan for projects.")
            return

        self.lbl_info.setText(f"Scanning: {folder}")
        projects = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith("_FAM_SRM_Report.pdf"):
                    prefix = f.replace("_FAM_SRM_Report.pdf", "")
                    projects.append((prefix, os.path.join(root, prefix)))

        if not projects:
            self.lbl_info.setText("No FAM-SRM reports found in this folder.")
            return

        for prefix, full_prefix in projects:
            item = QListWidgetItem(prefix)
            item.setData(Qt.ItemDataRole.UserRole, full_prefix)
            self.list_projects.addItem(item)

        self.lbl_info.setText(f"Found {len(projects)} FAM-SRM runs.")

    def _on_selection_changed(self):
        items = self.list_projects.selectedItems()
        if not items:
            return
        item = items[0]
        full_prefix = item.data(Qt.ItemDataRole.UserRole)
        self.main.last_out_prefix = full_prefix
        self.main.reports_page.load_from_prefix(full_prefix)
        self.main.set_page(1)  # switch to Reports


# ------------------- Settings Page -----------------------------
class SettingsPage(QWidget):
    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Theme toggle
        theme_group = QGroupBox("Theme")
        tlay = QHBoxLayout()
        self.btn_dark = QPushButton("Dark")
        self.btn_light = QPushButton("Light")
        self.btn_dark.setCheckable(True)
        self.btn_light.setCheckable(True)
        theme = self.main.settings.get("theme", "dark")
        (self.btn_dark if theme == "dark" else self.btn_light).setChecked(True)
        self.btn_dark.clicked.connect(lambda: self.change_theme("dark"))
        self.btn_light.clicked.connect(lambda: self.change_theme("light"))
        tlay.addWidget(self.btn_dark)
        tlay.addWidget(self.btn_light)
        tlay.addStretch()
        theme_group.setLayout(tlay)

        # Default paths
        path_group = QGroupBox("Default Paths")
        play = QVBoxLayout()
        row1 = QHBoxLayout()
        self.edit_in = QLineEdit(self.main.settings.get("default_input_dir", ""))
        btn_in = QPushButton("...")
        btn_in.clicked.connect(self.pick_default_input)
        row1.addWidget(QLabel("Input:"))
        row1.addWidget(self.edit_in)
        row1.addWidget(btn_in)

        row2 = QHBoxLayout()
        self.edit_out = QLineEdit(self.main.settings.get("default_output_dir", ""))
        btn_out = QPushButton("...")
        btn_out.clicked.connect(self.pick_default_output)
        row2.addWidget(QLabel("Output:"))
        row2.addWidget(self.edit_out)
        row2.addWidget(btn_out)

        play.addLayout(row1)
        play.addLayout(row2)
        path_group.setLayout(play)

        # Advanced
        adv_group = QGroupBox("Advanced")
        alay = QHBoxLayout()
        self.chk_gpu = QCheckBox("GPU Enabled (reserved)")
        self.chk_gpu.setChecked(self.main.settings.get("gpu_enabled", False))
        self.chk_gpu.stateChanged.connect(self.update_gpu_setting)

        self.chk_downscale_half = QCheckBox("Prefer 0.5× downscale for huge non-GeoTIFF images (reserved)")
        self.chk_downscale_half.setChecked(self.main.settings.get("downscale_mode", "off") == "half")
        self.chk_downscale_half.stateChanged.connect(self.update_downscale_setting)

        alay.addWidget(self.chk_gpu)
        alay.addWidget(self.chk_downscale_half)
        alay.addStretch()
        adv_group.setLayout(alay)

        layout.addWidget(theme_group)
        layout.addWidget(path_group)
        layout.addWidget(adv_group)
        layout.addStretch()
        self.setLayout(layout)

    def change_theme(self, name: str):
        self.main.settings["theme"] = name
        SettingsManager.save(self.main.settings)
        ThemeManager.apply_theme(QApplication.instance(), name)
        LOGGER.log(f"Theme changed to {name}.")

        self.btn_dark.setChecked(name == "dark")
        self.btn_light.setChecked(name == "light")

    def pick_default_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Default Input Folder")
        if folder:
            self.edit_in.setText(folder)
            self.main.settings["default_input_dir"] = folder
            SettingsManager.save(self.main.settings)
            LOGGER.log(f"Default input folder set to: {folder}")

    def pick_default_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Default Output Folder")
        if folder:
            self.edit_out.setText(folder)
            self.main.settings["default_output_dir"] = folder
            SettingsManager.save(self.main.settings)
            LOGGER.log(f"Default output folder set to: {folder}")

    def update_gpu_setting(self, state):
        self.main.settings["gpu_enabled"] = bool(state)
        SettingsManager.save(self.main.settings)
        LOGGER.log(f"GPU enabled flag set to: {bool(state)}")

    def update_downscale_setting(self, state):
        self.main.settings["downscale_mode"] = "half" if state else "off"
        SettingsManager.save(self.main.settings)
        LOGGER.log(f"Downscale mode set to: {self.main.settings['downscale_mode']}")


# ------------------- About Page -----------------------------
class AboutPage(QWidget):
    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Optional logo
        logo_path = None
        for candidate in ("LOGO.png", "LOGO.jpg", "logo.png", "logo.jpg"):
            if os.path.exists(candidate):
                logo_path = candidate
                break

        if logo_path:
            img = QImage(logo_path)
            if not img.isNull():
                pix = QPixmap.fromImage(
                    img.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
                )
                lbl_logo = QLabel()
                lbl_logo.setPixmap(pix)
                lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(lbl_logo)

        uname = platform.uname()
        sys_lines = [
            f"System: {uname.system} {uname.release}",
            f"Node: {uname.node}",
            f"Machine: {uname.machine}",
            f"Python: {platform.python_version()}",
        ]
        if psutil:
            try:
                mem = psutil.virtual_memory()
                sys_lines.append(f"RAM: {round(mem.total / (1024**3), 1)} GB")
            except Exception:
                pass

        info = QLabel()
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = (
            "<b>FAM-SRM Virtual Spectral Engine</b><br><br>"
            "Virtual NIR, FAM-enhanced NDVI, fractal stress, and structural graphon "
            "for RGB-only orthomosaics and imagery.<br><br>"
            "Developed by <b>Salt19 LLC</b> (Gabriel Allit).<br><br>"
            "<b>Runtime Environment:</b><br>"
            + "<br>".join(sys_lines)
        )
        info.setText(text)
        layout.addWidget(info)
        layout.addStretch()
        self.setLayout(layout)


# ------------------- Diagnostics Page -----------------------------
class DiagnosticsPage(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        btn_row = QHBoxLayout()
        self.btn_clear = QPushButton("Clear Log")
        self.btn_save = QPushButton("Save Log to File")
        self.btn_clear.clicked.connect(self.log_box.clear)
        self.btn_save.clicked.connect(self.save_log)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch()

        layout.addWidget(self.log_box)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def append(self, text: str):
        self.log_box.append(text)

    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Diagnostic Log", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.log_box.toPlainText())
                QMessageBox.information(self, "Saved", f"Log saved to:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save log:\n{e}")


# ===============================================================
# Main Window
# ===============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = SettingsManager.load()
        ThemeManager.apply_theme(QApplication.instance(), self.settings.get("theme", "dark"))

        self.last_out_prefix = None
        self.last_out_dir = None
        self.engine_thread: EngineWorker | None = None

        self.setWindowTitle("FAM-SRM Virtual Spectral Engine")
        self.setMinimumSize(1400, 900)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._build_menubar()
        self._build_central()

        LOGGER.message.connect(self.diagnostics_page.append)

    # -------------- Menubar -----------------
    def _build_menubar(self):
        menu_bar: QMenuBar = self.menuBar()

        m_file = menu_bar.addMenu("&File")
        act_new = QAction("New Session", self)
        act_new.triggered.connect(self.new_session)
        m_file.addAction(act_new)

        m_file.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        m_view = menu_bar.addMenu("&View")
        act_dark = QAction("Dark Theme", self)
        act_dark.triggered.connect(lambda: self.change_theme("dark"))
        act_light = QAction("Light Theme", self)
        act_light.triggered.connect(lambda: self.change_theme("light"))
        m_view.addAction(act_dark)
        m_view.addAction(act_light)

        m_help = menu_bar.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(lambda: self.set_page(4))
        m_help.addAction(act_about)

    # -------------- Central layout: sidebar + pages ---------
    def _build_central(self):
        central = QWidget()
        root = QHBoxLayout()
        central.setLayout(root)
        self.setCentralWidget(central)

        # Sidebar
        self.sidebar = QWidget()
        s_layout = QVBoxLayout()
        s_layout.setContentsMargins(6, 6, 6, 6)
        s_layout.setSpacing(8)
        self.sidebar.setLayout(s_layout)

        self.btn_engine = self._make_nav_button("🧪", "Engine")
        self.btn_reports = self._make_nav_button("📊", "Reports")
        self.btn_projects = self._make_nav_button("📁", "Projects")
        self.btn_settings = self._make_nav_button("⚙️", "Settings")
        self.btn_about = self._make_nav_button("🛈", "About")
        self.btn_diag = self._make_nav_button("🖥️", "Diagnostics")

        self.nav_buttons = [
            self.btn_engine, self.btn_reports, self.btn_projects,
            self.btn_settings, self.btn_about, self.btn_diag
        ]
        for b in self.nav_buttons:
            b.setCheckable(True)

        self.btn_engine.setChecked(True)

        s_layout.addWidget(self.btn_engine)
        s_layout.addWidget(self.btn_reports)
        s_layout.addWidget(self.btn_projects)
        s_layout.addWidget(self.btn_settings)
        s_layout.addWidget(self.btn_about)
        s_layout.addWidget(self.btn_diag)
        s_layout.addStretch()

        root.addWidget(self.sidebar)

        # Pages (stack)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)

        self.engine_page = EnginePage(self)
        self.reports_page = ReportsPage(self)
        self.projects_page = ProjectsPage(self)
        self.settings_page = SettingsPage(self)
        self.about_page = AboutPage(self)
        self.diagnostics_page = DiagnosticsPage()

        self.pages.addWidget(self.engine_page)      # 0
        self.pages.addWidget(self.reports_page)     # 1
        self.pages.addWidget(self.projects_page)    # 2
        self.pages.addWidget(self.settings_page)    # 3
        self.pages.addWidget(self.about_page)       # 4
        self.pages.addWidget(self.diagnostics_page) # 5

        self.btn_engine.clicked.connect(lambda: self.set_page(0))
        self.btn_reports.clicked.connect(lambda: self.set_page(1))
        self.btn_projects.clicked.connect(lambda: self.set_page(2))
        self.btn_settings.clicked.connect(lambda: self.set_page(3))
        self.btn_about.clicked.connect(lambda: self.set_page(4))
        self.btn_diag.clicked.connect(lambda: self.set_page(5))

    def _make_nav_button(self, emoji: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(emoji)
        btn.setToolTip(tooltip)
        btn.setFixedSize(70, 70)
        return btn

    # -------------- Navigation & Page logic -----------------
    def set_page(self, index: int):
        self.pages.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

        if index == 1:
            self.reports_page.refresh()
        elif index == 2:
            self.projects_page.refresh()

    # -------------- Session / Theme / Engine control --------
    def new_session(self):
        self.last_out_prefix = None
        self.last_out_dir = None
        self.engine_page.input_edit.clear()
        self.engine_page.output_edit.clear()
        self.reports_page.current_prefix = None
        self.reports_page.lbl_title.setText("No report loaded.")
        self.projects_page.list_projects.clear()
        self.projects_page.lbl_info.setText("Projects will be detected from output folder.")
        self.diagnostics_page.log_box.clear()
        self.status_bar.showMessage("New session started.", 3000)

    def change_theme(self, name: str):
        self.settings["theme"] = name
        SettingsManager.save(self.settings)
        ThemeManager.apply_theme(QApplication.instance(), name)
        LOGGER.log(f"Theme changed to {name} (via menu).")

    def start_engine(self, input_path: str, output_dir: str, options: dict):
        if self.engine_thread is not None and self.engine_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Engine is already running.")
            return

        self.status_bar.showMessage("Running FAM-SRM engine…")
        LOGGER.log("Launching engine thread.")

        self.engine_thread = EngineWorker(self, input_path, output_dir, options, parent=self)
        self.engine_thread.progress.connect(self.engine_page.progress.setValue)
        self.engine_thread.finished.connect(self._engine_finished)
        self.engine_thread.error.connect(self._engine_error)
        self.engine_thread.start()

    def _engine_finished(self, out_prefix: str, out_dir: str, options: dict):
        self.last_out_prefix = out_prefix
        self.last_out_dir = out_dir
        self.status_bar.showMessage("FAM-SRM processing complete.", 5000)
        LOGGER.log(f"Engine finished. Last out_prefix = {out_prefix}")
        self.reports_page.load_from_prefix(out_prefix)
        self.projects_page.refresh()
        QMessageBox.information(self, "Complete", "FAM-SRM processing complete.")

    def _engine_error(self, msg: str):
        self.status_bar.showMessage("Engine error.", 5000)
        QMessageBox.critical(self, "Engine Error", msg)


# ===============================================================
# Main entry point
# ===============================================================
def main():
    app = QApplication(sys.argv)

    settings = SettingsManager.load()
    ThemeManager.apply_theme(app, settings.get("theme", "dark"))

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
