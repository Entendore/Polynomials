import sys
import json
import numpy as np
from scipy.interpolate import CubicSpline
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSlider, QLabel, QCheckBox, QPushButton, QScrollArea, QFrame,
    QGroupBox, QComboBox, QDoubleSpinBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

# Max polynomial degree
MAX_DEGREE = 9

# Distinct colors for powers 9 down to 0
COLORS = {
    9: '#e6194b', 8: '#3cb44b', 7: '#4363d8', 6: '#f58231', 
    5: '#911eb4', 4: '#42d4f4', 3: '#f032e6', 2: '#bcf60c', 
    1: '#fabebe', 0: '#008080'
}

def calc_poly(x, coeffs, mode='poly'):
    """Helper to calculate polynomial, derivative, or integral."""
    y = np.zeros_like(x)
    for p, val in coeffs.items():
        if mode == 'poly':
            y += val * (x ** p)
        elif mode == 'deriv' and p > 0:
            y += val * p * (x ** (p - 1))
        elif mode == 'integral':
            y += val * (x ** (p + 1)) / (p + 1)
    return y

def get_real_roots(coeffs, max_deg):
    """Finds real roots of the polynomial."""
    # Format coefficients for numpy (highest degree first)
    coeff_list = [coeffs.get(p, 0.0) for p in range(max_deg, -1, -1)]
    
    # Remove leading zeros to avoid numpy errors
    while len(coeff_list) > 2 and coeff_list[0] == 0:
        coeff_list.pop(0)
        
    if len(set(coeff_list)) == 1 and coeff_list[0] == 0:
        return [] # Polynomial is just 0
        
    roots = np.roots(coeff_list)
    # Filter for real roots (imaginary part close to 0)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-5]
    return real_roots


class CoeffControl(QFrame):
    def __init__(self, power, default_val=0.0):
        super().__init__()
        self.power = power
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(4, 2, 4, 2)

        text = "Constant" if power == 0 else f"x<sup>{power}</sup>"
        self.name_label = QLabel(text)
        self.name_label.setFixedWidth(50)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(-100, 100)
        self.slider.setValue(int(default_val * 10))
        
        self.val_label = QLabel(f"{default_val:.1f}")
        self.val_label.setFixedWidth(40)
        
        self.vis_checkbox = QCheckBox("Show")
        self.vis_checkbox.setChecked(True)

        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.slider, stretch=1)
        self.layout.addWidget(self.val_label)
        self.layout.addWidget(self.vis_checkbox)
        self.slider.valueChanged.connect(self.update_label)

    def update_label(self, val): self.val_label.setText(f"{val / 10.0:.1f}")
    def value(self): return self.slider.value() / 10.0
    def is_visible(self): return self.vis_checkbox.isChecked()


class MplCanvas(FigureCanvas):
    spline_changed = Signal()  # Added signal for safe communication

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.apply_dark_theme()
        
        self.spline_points = []
        self.spline_active = False
        
        self.mpl_connect('button_press_event', self.on_click)

    def apply_dark_theme(self):
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#252526')
        self.ax.tick_params(colors='#cccccc')
        for spine in self.ax.spines.values(): spine.set_color('#cccccc')

    def on_click(self, event):
        if not self.spline_active or event.inaxes != self.ax:
            return
            
        if event.button == 1: # Left click
            self.spline_points.append((event.xdata, event.ydata))
        elif event.button == 3 and self.spline_points: # Right click
            pts = np.array(self.spline_points)
            dist = np.sqrt((pts[:, 0] - event.xdata)**2 + (pts[:, 1] - event.ydata)**2)
            idx = np.argmin(dist)
            x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            if dist[idx] < x_range * 0.1:
                self.spline_points.pop(idx)
                
        self.spline_changed.emit()  # Emit signal instead of parent traversal

    def plot_polynomial(self, coeffs, auto_scale_y, x_range, show_deriv, show_integral, show_roots, show_extrema):
        self.ax.clear()
        self.apply_dark_theme()
        x = np.linspace(-x_range, x_range, 400)

        # --- Convert CoeffControl objects into float values ---
        float_coeffs = {}
        visible_powers = set()
        for p, ctrl in coeffs.items():
            float_coeffs[p] = ctrl.value()
            if ctrl.is_visible():
                visible_powers.add(p)

        y_total = calc_poly(x, float_coeffs, 'poly')

        # Plot individual terms
        for power, val in float_coeffs.items():
            if val != 0 and power in visible_powers:
                y_term = val * (x ** power)
                self.ax.plot(x, y_term, '--', color=COLORS[power], alpha=0.7, label=f'{val:.1f}x^{power}')

        # Calculus overlays
        if show_integral:
            y_int = calc_poly(x, float_coeffs, 'integral')
            self.ax.plot(x, y_int, color='#ff9966', linewidth=2, alpha=0.8, label='∫P(x)dx')
        if show_deriv:
            y_der = calc_poly(x, float_coeffs, 'deriv')
            self.ax.plot(x, y_der, color='#ffcc00', linewidth=2, alpha=0.8, label="P'(x)")

        # Main Polynomial
        self.ax.plot(x, y_total, color='white', linewidth=3, label='P(x) = Sum')

        # Spline
        if self.spline_active and len(self.spline_points) >= 2:
            pts = sorted(self.spline_points, key=lambda p: p[0])
            x_pts = [p[0] for p in pts]
            y_pts = [p[1] for p in pts]
            try:
                cs = CubicSpline(x_pts, y_pts)
                y_spline = cs(x)
                self.ax.plot(x, y_spline, color='#00ff00', linewidth=3, alpha=0.9, label='Cubic Spline')
            except Exception:
                pass 
            self.ax.scatter(x_pts, y_pts, color='#00ff00', s=80, zorder=5, edgecolors='black')

        # Markers
        if show_roots:
            roots = get_real_roots(float_coeffs, MAX_DEGREE)
            valid_roots = [r for r in roots if -x_range <= r <= x_range]
            if valid_roots:
                self.ax.scatter(valid_roots, [0]*len(valid_roots), color='red', s=100, zorder=10, edgecolors='black', label='Roots')
        if show_extrema:
            deriv_coeffs = {p-1: val*p for p, val in float_coeffs.items() if p > 0}
            extrema = get_real_roots(deriv_coeffs, MAX_DEGREE-1)
            valid_extrema = [e for e in extrema if -x_range <= e <= x_range]
            if valid_extrema:
                y_extrema = calc_poly(np.array(valid_extrema), float_coeffs, 'poly')
                self.ax.scatter(valid_extrema, y_extrema, color='cyan', s=100, zorder=10, marker='^', edgecolors='black', label='Extrema')

        self.ax.set_xlim(-x_range, x_range)
        
        # Y-Axis Scaling
        y_min, y_max = -50, 50
        if auto_scale_y:
            y_vals = [y_total]
            if show_deriv: y_vals.append(y_der)
            if show_integral: y_vals.append(y_int)
            if self.spline_active and len(self.spline_points) >= 2: y_vals.append(y_spline)
            
            y_min = min([np.min(y) for y in y_vals]) - 2
            y_max = max([np.max(y) for y in y_vals]) + 2
            y_min = max(-1000, y_min)
            y_max = min(1000, y_max)
            
        self.ax.set_ylim(y_min, y_max)
            
        self.ax.axhline(0, color='gray', linewidth=0.5)
        self.ax.axvline(0, color='gray', linewidth=0.5)
        self.ax.grid(True, linestyle=':', alpha=0.5, color='gray')
        self.ax.set_xlabel("X", color='#cccccc')
        self.ax.set_ylabel("Y", color='#cccccc')
        self.ax.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#444444', labelcolor='white')
        self.draw_idle()


class RenderWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, coeffs, anim_power, start_val, end_val, fmt, fps, duration, filepath, x_range, spline_points, show_deriv, show_integral):
        super().__init__()
        self.coeffs = coeffs
        self.anim_power = anim_power
        self.start_val = start_val
        self.end_val = end_val
        self.fmt = fmt
        self.fps = fps
        self.duration = duration
        self.filepath = filepath
        self.x_range = x_range
        self.spline_points = spline_points
        self.show_deriv = show_deriv
        self.show_integral = show_integral

    def get_eq_string(self, c):
        terms = []
        for p in range(MAX_DEGREE, -1, -1):
            val = c.get(p, 0.0)
            if val == 0: continue
            if p == 0: term = f"{abs(val):.1f}"
            elif p == 1: term = f"{abs(val):.1f}x"
            else: term = f"{abs(val):.1f}x^{p}"
            if not terms: terms.append(f"-{term}" if val < 0 else term)
            else: terms.append(f" - {term}" if val < 0 else f" + {term}")
        return "P(x) = " + " ".join(terms) if terms else "P(x) = 0"

    def run(self):
        try:
            if "Shorts" in self.fmt:
                fig = Figure(figsize=(7.2, 12.8), dpi=150)
                ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
            else:
                fig = Figure(figsize=(12.8, 7.2), dpi=150)
                ax = fig.add_subplot(111)

            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#252526')
            ax.tick_params(colors='#cccccc')
            for spine in ax.spines.values(): spine.set_color('#cccccc')

            x = np.linspace(-self.x_range, self.x_range, 400)
            
            # Pre-calculate Y bounds for video stability
            c_start = self.coeffs.copy()
            c_end = self.coeffs.copy()
            c_start[self.anim_power] = self.start_val
            c_end[self.anim_power] = self.end_val
            
            y1 = calc_poly(x, c_start, 'poly')
            y2 = calc_poly(x, c_end, 'poly')
            y_min = min(np.min(y1), np.min(y2)) - 2
            y_max = max(np.max(y1), np.max(y2)) + 2
            
            ax.set_xlim(-self.x_range, self.x_range)
            ax.set_ylim(y_min, y_max)

            # Setup lines
            line_sum, = ax.plot([], [], color='white', linewidth=4, label='P(x) = Sum')
            line_anim, = ax.plot([], [], '--', color=COLORS[self.anim_power], linewidth=2, alpha=0.8, label=f'Anim Term')
            
            line_deriv = ax.plot([], [], color='#ffcc00', linewidth=2, alpha=0.8, label="P'(x)")[0] if self.show_deriv else None
            line_integ = ax.plot([], [], color='#ff9966', linewidth=2, alpha=0.8, label='∫P(x)dx')[0] if self.show_integral else None
            
            # Static Spline
            if self.spline_points and len(self.spline_points) >= 2:
                pts = sorted(self.spline_points, key=lambda p: p[0])
                x_pts = [p[0] for p in pts]
                y_pts = [p[1] for p in pts]
                cs = CubicSpline(x_pts, y_pts)
                y_spline = cs(x)
                ax.plot(x, y_spline, color='#00ff00', linewidth=3, alpha=0.9, label='Cubic Spline')
                ax.scatter(x_pts, y_pts, color='#00ff00', s=150, zorder=5, edgecolors='black')

            ax.axhline(0, color='gray', linewidth=1)
            ax.axvline(0, color='gray', linewidth=1)
            ax.grid(True, linestyle=':', alpha=0.5, color='gray')
            ax.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#444444', labelcolor='white', fontsize=12)
            
            title_text = fig.text(0.5, 0.94, "Additive Decomposition", ha='center', color='white', fontsize=24, weight='bold')
            eq_text = fig.text(0.5, 0.05, "", ha='center', color='#42d4f4', fontsize=18, family='monospace')

            frames = int(self.fps * self.duration)

            def init():
                line_sum.set_data([], [])
                line_anim.set_data([], [])
                return line_sum, line_anim

            def update(frame):
                t = frame / frames
                ease_t = t * t * (3 - 2 * t) 
                current_val = self.start_val + (self.end_val - self.start_val) * ease_t
                
                c_current = self.coeffs.copy()
                c_current[self.anim_power] = current_val

                y_total = calc_poly(x, c_current, 'poly')
                line_sum.set_data(x, y_total)
                
                y_anim = current_val * (x ** self.anim_power)
                line_anim.set_data(x, y_anim)
                
                if line_deriv: line_deriv.set_data(x, calc_poly(x, c_current, 'deriv'))
                if line_integ: line_integ.set_data(x, calc_poly(x, c_current, 'integral'))
                
                eq_text.set_text(self.get_eq_string(c_current))
                self.progress.emit(int((frame / frames) * 100))
                return [ln for ln in [line_sum, line_anim, line_deriv, line_integ, eq_text] if ln is not None]

            ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=1000/self.fps)
            writer = animation.FFMpegWriter(fps=self.fps, bitrate=8000)
            ani.save(self.filepath, writer=writer)
            self.finished.emit(self.filepath)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Calculus & Polynomial Studio")
        self.resize(1150, 850)

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #e0e0e0; font-size: 13px; }
            QFrame { background-color: #2b2b2b; border-radius: 4px; }
            QLabel { background-color: transparent; }
            QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #007acc; subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QCheckBox { background-color: transparent; spacing: 5px; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #0098ff; }
            QPushButton:disabled { background-color: #3a3a3a; color: #777; }
            QSlider::groove:horizontal { border: 1px solid #3a3a3a; height: 4px; background: #1e1e1e; border-radius: 2px; }
            QSlider::handle:horizontal { background: #007acc; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }
            QComboBox, QDoubleSpinBox { background-color: #2b2b2b; border: 1px solid #3a3a3a; padding: 4px; border-radius: 4px; }
            QStatusBar { background-color: #252526; color: #888; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.eq_label = QLabel("")
        self.eq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eq_label.setStyleSheet("font-size: 18px; font-family: 'Consolas', monospace; padding: 10px; background-color: #252526;")
        main_layout.addWidget(self.eq_label)

        middle_layout = QHBoxLayout()
        main_layout.addLayout(middle_layout, stretch=1)

        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        self.canvas.spline_changed.connect(self.update_plot)  # Connect signal
        middle_layout.addWidget(self.canvas, stretch=3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(controls_widget)
        middle_layout.addWidget(scroll, stretch=1)

        # --- Session Management ---
        session_group = QGroupBox("Session")
        session_layout = QHBoxLayout(session_group)
        save_btn = QPushButton("Save State")
        load_btn = QPushButton("Load State")
        save_btn.clicked.connect(self.save_state)
        load_btn.clicked.connect(self.load_state)
        session_layout.addWidget(save_btn)
        session_layout.addWidget(load_btn)
        controls_layout.addWidget(session_group)

        # --- Coefficients ---
        self.coeffs = {}
        defaults = {3: 1.0, 2: -2.0, 1: 0.5, 0: 3.0}
        for p in range(MAX_DEGREE, -1, -1):
            default_val = defaults.get(p, 0.0)
            ctrl = CoeffControl(p, default_val)
            self.coeffs[p] = ctrl
            controls_layout.addWidget(ctrl)
            ctrl.slider.valueChanged.connect(self.update_plot)
            ctrl.vis_checkbox.stateChanged.connect(self.update_plot)

        controls_layout.addStretch()

        # --- Calculus & Markers ---
        math_group = QGroupBox("Calculus & Markers")
        math_layout = QVBoxLayout(math_group)
        self.deriv_cb = QCheckBox("Show First Derivative P'(x)")
        self.deriv_cb.stateChanged.connect(self.update_plot)
        math_layout.addWidget(self.deriv_cb)
        self.integ_cb = QCheckBox("Show Integral ∫P(x)dx")
        self.integ_cb.stateChanged.connect(self.update_plot)
        math_layout.addWidget(self.integ_cb)
        self.roots_cb = QCheckBox("Mark Real Roots (Red)")
        self.roots_cb.setChecked(True)
        self.roots_cb.stateChanged.connect(self.update_plot)
        math_layout.addWidget(self.roots_cb)
        self.extrema_cb = QCheckBox("Mark Extrema (Cyan)")
        self.extrema_cb.setChecked(True)
        self.extrema_cb.stateChanged.connect(self.update_plot)
        math_layout.addWidget(self.extrema_cb)
        controls_layout.addWidget(math_group)

        # --- Splines ---
        spline_group = QGroupBox("Spline Mode")
        spline_layout = QVBoxLayout(spline_group)
        self.spline_cb = QCheckBox("Enable Click-to-Add Points")
        self.spline_cb.stateChanged.connect(self.toggle_spline_mode)
        spline_layout.addWidget(self.spline_cb)
        info_lbl = QLabel("Left-click: Add\nRight-click: Remove")
        info_lbl.setStyleSheet("color: #888; font-size: 11px;")
        spline_layout.addWidget(info_lbl)
        clear_btn = QPushButton("Clear Points")
        clear_btn.clicked.connect(self.clear_spline)
        spline_layout.addWidget(clear_btn)
        controls_layout.addWidget(spline_group)

        # --- UI Settings ---
        ui_settings_group = QGroupBox("View Settings")
        ui_settings_layout = QHBoxLayout(ui_settings_group)
        self.auto_scale_cb = QCheckBox("Auto-Y")
        self.auto_scale_cb.setChecked(True)
        self.auto_scale_cb.stateChanged.connect(self.update_plot)
        ui_settings_layout.addWidget(self.auto_scale_cb)
        ui_settings_layout.addWidget(QLabel("X-Range:"))
        self.x_range_spin = QDoubleSpinBox()
        self.x_range_spin.setRange(0.5, 20.0)
        self.x_range_spin.setSingleStep(0.5)
        self.x_range_spin.setValue(3.0)
        self.x_range_spin.valueChanged.connect(self.update_plot)
        ui_settings_layout.addWidget(self.x_range_spin)
        ui_settings_layout.addStretch()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_values)
        ui_settings_layout.addWidget(reset_btn)
        controls_layout.addWidget(ui_settings_group)

        # --- Animation Studio ---
        anim_group = QGroupBox("Animation Studio (Export MP4)")
        anim_layout = QVBoxLayout(anim_group)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Format:"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["YouTube (16:9)", "Shorts (9:16)"])
        row1.addWidget(self.fmt_combo)
        row1.addWidget(QLabel("Animate:"))
        self.param_combo = QComboBox()
        for p in range(MAX_DEGREE, -1, -1):
            self.param_combo.addItem(f"x^{p}" if p > 0 else "Constant", p)
        self.param_combo.setCurrentIndex(MAX_DEGREE - 3)
        row1.addWidget(self.param_combo)
        anim_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Start:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-10.0, 10.0)
        self.start_spin.setSingleStep(0.1)
        self.start_spin.setValue(-5.0)
        row2.addWidget(self.start_spin)
        row2.addWidget(QLabel("End:"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-10.0, 10.0)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setValue(5.0)
        row2.addWidget(self.end_spin)
        row2.addWidget(QLabel("Dur(s):"))
        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(1.0, 300.0)
        self.dur_spin.setValue(5.0)
        row2.addWidget(self.dur_spin)
        anim_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.render_btn = QPushButton("🎬 Generate MP4")
        self.render_btn.clicked.connect(self.start_render)
        row3.addWidget(self.render_btn)
        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row3.addWidget(self.progress_label)
        anim_layout.addLayout(row3)
        controls_layout.addWidget(anim_group)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        self.update_plot()

    def toggle_spline_mode(self, state):
        self.canvas.spline_active = bool(state)
        if self.canvas.spline_active:
            self.canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.status_bar.showMessage("Spline Mode ON: Left-click to add points, Right-click to remove.")
        else:
            self.canvas.unsetCursor()
            self.status_bar.showMessage("Spline Mode OFF.")
        self.update_plot()

    def clear_spline(self):
        self.canvas.spline_points.clear()
        self.update_plot()
        self.status_bar.showMessage("Spline points cleared.")

    def save_state(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save State", "poly_state.json", "JSON Files (*.json)")
        if not filepath: return
        
        state = {
            "coeffs": {str(p): ctrl.value() for p, ctrl in self.coeffs.items()},
            "visible": {str(p): ctrl.is_visible() for p, ctrl in self.coeffs.items()},
            "spline_points": self.canvas.spline_points,
            "x_range": self.x_range_spin.value(),
            "auto_scale": self.auto_scale_cb.isChecked(),
            "show_deriv": self.deriv_cb.isChecked(),
            "show_integral": self.integ_cb.isChecked(),
            "show_roots": self.roots_cb.isChecked(),
            "show_extrema": self.extrema_cb.isChecked()
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=4)
            self.status_bar.showMessage(f"Saved state to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def load_state(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load State", "", "JSON Files (*.json)")
        if not filepath: return
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            for p_str, val in state.get("coeffs", {}).items():
                self.coeffs[int(p_str)].slider.setValue(int(val * 10))
            for p_str, vis in state.get("visible", {}).items():
                self.coeffs[int(p_str)].vis_checkbox.setChecked(vis)
                
            self.canvas.spline_points = [tuple(p) for p in state.get("spline_points", [])]
            self.x_range_spin.setValue(state.get("x_range", 3.0))
            self.auto_scale_cb.setChecked(state.get("auto_scale", True))
            self.deriv_cb.setChecked(state.get("show_deriv", False))
            self.integ_cb.setChecked(state.get("show_integral", False))
            self.roots_cb.setChecked(state.get("show_roots", True))
            self.extrema_cb.setChecked(state.get("show_extrema", True))
            
            self.status_bar.showMessage(f"Loaded state from {filepath}")
            self.update_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def get_equation_string(self):
        terms = []
        for p in range(MAX_DEGREE, -1, -1):
            val = self.coeffs[p].value()
            if val == 0: continue
            if p == 0: term = f"{abs(val):.1f}"
            elif p == 1: term = f"{abs(val):.1f}x"
            else: term = f"{abs(val):.1f}x<sup>{p}</sup>"
            if not terms: terms.append(f"-{term}" if val < 0 else term)
            else: terms.append(f" - {term}" if val < 0 else f" + {term}")
        return "P(x) = " + "".join(terms) if terms else "P(x) = 0"

    def update_plot(self):
        self.canvas.plot_polynomial(
            self.coeffs, self.auto_scale_cb.isChecked(), self.x_range_spin.value(),
            self.deriv_cb.isChecked(), self.integ_cb.isChecked(), 
            self.roots_cb.isChecked(), self.extrema_cb.isChecked()
        )
        self.eq_label.setText(self.get_equation_string())

    def reset_values(self):
        defaults = {3: 1.0, 2: -2.0, 1: 0.5, 0: 3.0}
        for p in range(MAX_DEGREE, -1, -1):
            val = defaults.get(p, 0.0)
            self.coeffs[p].slider.setValue(int(val * 10))
            self.coeffs[p].vis_checkbox.setChecked(True)
        self.x_range_spin.setValue(3.0)
        self.auto_scale_cb.setChecked(True)
        self.deriv_cb.setChecked(False)
        self.integ_cb.setChecked(False)
        self.roots_cb.setChecked(True)
        self.extrema_cb.setChecked(True)
        self.status_bar.showMessage("Reset to defaults.")

    def start_render(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save MP4 Video", "animation.mp4", "MP4 Video (*.mp4)")
        if not filepath: return

        self.render_btn.setEnabled(False)
        self.progress_label.setText("Starting render...")

        current_coeffs = {p: ctrl.value() for p, ctrl in self.coeffs.items()}
        anim_power = self.param_combo.currentData()
        fmt = self.fmt_combo.currentText()
        
        spline_pts = self.canvas.spline_points if self.canvas.spline_active else []
        
        self.worker = RenderWorker(
            coeffs=current_coeffs, anim_power=anim_power,
            start_val=self.start_spin.value(), end_val=self.end_spin.value(),
            fmt=fmt, fps=30, duration=self.dur_spin.value(), filepath=filepath,
            x_range=self.x_range_spin.value(), spline_points=spline_pts,
            show_deriv=self.deriv_cb.isChecked(), show_integral=self.integ_cb.isChecked()
        )
        self.worker.progress.connect(self.on_render_progress)
        self.worker.finished.connect(self.on_render_finished)
        self.worker.error.connect(self.on_render_error)
        self.worker.start()
        self.status_bar.showMessage("Rendering video in background...")

    def on_render_progress(self, val): 
        self.progress_label.setText(f"Rendering... {val}%")
        self.status_bar.showMessage(f"Rendering video... {val}%")

    def on_render_finished(self, filepath):
        self.progress_label.setText("Ready")
        self.render_btn.setEnabled(True)
        self.status_bar.showMessage(f"Video saved to {filepath}")
        QMessageBox.information(self, "Success", f"Video successfully saved to:\n{filepath}")

    def on_render_error(self, err_msg):
        self.progress_label.setText("Error!")
        self.render_btn.setEnabled(True)
        self.status_bar.showMessage("Render failed.")
        QMessageBox.critical(self, "Rendering Error", f"Failed to render video.\nEnsure FFmpeg is installed and in your PATH.\n\nError:\n{err_msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())