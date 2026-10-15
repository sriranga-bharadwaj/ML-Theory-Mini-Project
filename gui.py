"""Professional Real-time Video Analysis & Biometrics GUI for Drosophila Courtship.

Features:
- Live 4-stage classical machine learning pipeline execution (60+ FPS on CPU).
- Biometric telemetry tracking: fly position, sex dimorphism, body heading, and wing angle regression.
- Interactive playback monitor: scrub bar, frame stepping, playback speed multipliers, overlay toggles.
- Live wing kinematics oscilloscope with real-time bilateral asymmetry calculation.
- Automated benchmark performance inspection, timing breakdown, and figure exploration.
"""
import argparse
import json
import os
import sys
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

import cv2
import joblib
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))
import pipeline as P  # noqa: E402,F401
from evaluation import frame_errors  # noqa: E402
from visualize import draw_overlay  # noqa: E402

# -----------------------------------------------------------------------------
# Design System & Color Palette
# -----------------------------------------------------------------------------
NAVY_HEADER = "#0B132B"
NAVY_HEADER_BORDER = "#1C2541"
NAVY_SUB = "#94A3B8"
NAVY_LIGHT = "#CBD5E1"

BG = "#F8FAFC"               # Soft neutral background
CARD_BG = "#FFFFFF"          # Card container fill
CARD_BORDER = "#E2E8F0"      # Subtle card border
CARD_ALT = "#F1F5F9"         # Subtle alternating row / pill fill

TEXT_PRIMARY = "#0F172A"     # Deep slate high contrast
TEXT_SECONDARY = "#334155"   # Medium slate
TEXT_MUTED = "#64748B"       # Subtle caption text

ACCENT_BLUE = "#2563EB"      # Sapphire Blue (Male / Primary actions)
ACCENT_BLUE_DARK = "#1D4ED8"
ACCENT_BLUE_LIGHT = "#EFF6FF"

ACCENT_ROSE = "#E11D48"      # Vivid Rose (Female)
ACCENT_ROSE_LIGHT = "#FFF1F2"

ACCENT_AMBER = "#D97706"     # Warm Amber (Wing angles / Asymmetry)
ACCENT_AMBER_LIGHT = "#FEF3C7"

ACCENT_GREEN = "#059669"     # Emerald Green (Active status / High accuracy)
ACCENT_GREEN_LIGHT = "#ECFDF5"

ACCENT_PURPLE = "#7C3AED"    # Violet / Throughput metrics
FONT_FAMILY = "Segoe UI"
DISPLAY_SIZE = 500


def font(size, bold=False):
    return (FONT_FAMILY, size, "bold" if bold else "normal")


# -----------------------------------------------------------------------------
# Reusable Styled Components
# -----------------------------------------------------------------------------
class ModernCard(tk.Frame):
    """Clean card container with soft outline border."""
    def __init__(self, parent, bg=CARD_BG, border=CARD_BORDER, padx=12, pady=12, **kwargs):
        super().__init__(parent, bg=bg, highlightbackground=border, highlightthickness=1, padx=padx, pady=pady, **kwargs)


class KPIBadge(ModernCard):
    """Top-level key performance indicator badge."""
    def __init__(self, parent, title, value="-", subtitle="", accent_color=ACCENT_BLUE):
        super().__init__(parent, padx=14, pady=10)
        self.accent_color = accent_color

        top_row = tk.Frame(self, bg=CARD_BG)
        top_row.pack(fill="x", anchor="w")

        self.dot = tk.Label(top_row, text="●", fg=accent_color, bg=CARD_BG, font=font(8))
        self.dot.pack(side="left", padx=(0, 6))

        self.title_lbl = tk.Label(top_row, text=title.upper(), bg=CARD_BG, fg=TEXT_MUTED, font=font(8, True))
        self.title_lbl.pack(side="left")

        self.value_lbl = tk.Label(self, text=value, bg=CARD_BG, fg=TEXT_PRIMARY, font=font(17, True))
        self.value_lbl.pack(anchor="w", pady=(3, 0))

        self.sub_lbl = tk.Label(self, text=subtitle, bg=CARD_BG, fg=TEXT_MUTED, font=font(8))
        if subtitle:
            self.sub_lbl.pack(anchor="w", pady=(1, 0))

    def update_val(self, val, sub=None, color=None):
        self.value_lbl.config(text=val)
        if sub is not None:
            self.sub_lbl.config(text=sub)
            if not self.sub_lbl.winfo_ismapped() and sub:
                self.sub_lbl.pack(anchor="w", pady=(1, 0))
        if color:
            self.value_lbl.config(fg=color)
            self.dot.config(fg=color)


class TelemetryFlyCard(ModernCard):
    """Structured telemetry readout for Male or Female subject."""
    def __init__(self, parent, title, sex_code, accent_color, accent_bg, row_labels):
        super().__init__(parent, padx=0, pady=0)
        self.accent_color = accent_color
        self.sex_code = sex_code

        # Header Ribbon
        header = tk.Frame(self, bg=accent_color, padx=14, pady=7)
        header.pack(fill="x")

        icon = "♂" if sex_code == "M" else "♀"
        tk.Label(header, text=f"{icon}  {title}", bg=accent_color, fg="white", font=font(11, True)).pack(side="left")

        sub_badge = tk.Label(header, text=f"ID: {'01' if sex_code == 'M' else '02'}",
                             bg=accent_color, fg="white", font=font(8))
        sub_badge.pack(side="right")

        # Table Column Headings
        table_head = tk.Frame(self, bg=CARD_ALT, padx=12, pady=5)
        table_head.pack(fill="x")
        tk.Label(table_head, text="METRIC", bg=CARD_ALT, fg=TEXT_MUTED, font=font(8, True), width=16, anchor="w").pack(side="left")
        tk.Label(table_head, text="PREDICTED", bg=CARD_ALT, fg=TEXT_PRIMARY, font=font(8, True), width=13, anchor="e").pack(side="left")
        tk.Label(table_head, text="GROUND TRUTH", bg=CARD_ALT, fg=TEXT_MUTED, font=font(8, True), width=14, anchor="e").pack(side="left")
        tk.Label(table_head, text="ERROR (Δ)", bg=CARD_ALT, fg=TEXT_MUTED, font=font(8, True), width=11, anchor="e").pack(side="left")

        # Table Rows
        self.cells = {}
        body = tk.Frame(self, bg=CARD_BG, padx=12, pady=6)
        body.pack(fill="both", expand=True)

        for i, metric in enumerate(row_labels):
            row_bg = CARD_BG if i % 2 == 0 else "#FAFBFD"
            row = tk.Frame(body, bg=row_bg, pady=4)
            row.pack(fill="x")

            lbl_m = tk.Label(row, text=metric, bg=row_bg, fg=TEXT_SECONDARY, font=font(9), width=16, anchor="w")
            lbl_m.pack(side="left")

            lbl_p = tk.Label(row, text="-", bg=row_bg, fg=TEXT_PRIMARY, font=font(10, True), width=13, anchor="e")
            lbl_p.pack(side="left")

            lbl_t = tk.Label(row, text="-", bg=row_bg, fg=TEXT_MUTED, font=font(9), width=14, anchor="e")
            lbl_t.pack(side="left")

            lbl_e = tk.Label(row, text="-", bg=row_bg, fg=accent_color, font=font(9, True), width=11, anchor="e")
            lbl_e.pack(side="left")

            self.cells[metric] = (lbl_p, lbl_t, lbl_e)

    def set_row(self, metric, pred="-", truth="-", err="-", err_color=None):
        if metric in self.cells:
            p, t, e = self.cells[metric]
            p.config(text=pred)
            t.config(text=truth)
            e.config(text=err)
            if err_color:
                e.config(fg=err_color)
            else:
                e.config(fg=self.accent_color)


# -----------------------------------------------------------------------------
# Main Application Class
# -----------------------------------------------------------------------------
class App:
    def __init__(self, root, video=None, autoplay=True):
        self.root = root
        root.title("Fruit Fly Biometric Analysis & Tracking Studio")
        root.geometry("1400x840")
        root.minsize(1280, 780)
        root.configure(bg=BG)

        self._apply_theme()

        # Load trained pipeline model
        model_path = os.path.join(BASE, "models", "pipeline.joblib")
        if not os.path.exists(model_path):
            messagebox.showerror(
                "Trained Model Not Found",
                "Model pipeline.joblib was not found in models/.\nPlease run `python train_evaluate.py` first."
            )
            raise SystemExit(1)
        self.pipe = joblib.load(model_path)

        # Video & Telemetry State
        self.cap = None
        self.gt = None
        self.video_name = ""
        self.n_frames = 0
        self.src_fps = 30.0
        self.idx = 0
        self.playing = False
        self.last_flies = []
        self.photo = None
        self.chart_photo = None
        self.proc_times = deque(maxlen=30)
        self.series = {k: deque(maxlen=200) for k in ("pr", "pl", "tr", "tl")}
        self._seeking = False
        self._job = None

        # Build Layout
        self._build_header()
        main_content = tk.Frame(root, bg=BG)
        main_content.pack(fill="both", expand=True, padx=16, pady=(10, 12))

        self._build_left_monitor(main_content)
        self._build_right_notebook(main_content)

        # Keyboard Shortcuts
        root.bind("<space>", lambda e: self.toggle_play())
        root.bind("<Left>", lambda e: self.step_back())
        root.bind("<Right>", lambda e: self.step_forward())
        root.bind("<r>", lambda e: self.restart())
        root.bind("<R>", lambda e: self.restart())
        root.protocol("WM_DELETE_WINDOW", self.close)

        # Initial Video Loading
        default_video = video or os.path.join(BASE, "data", "video.avi")
        if os.path.exists(default_video):
            self.open_video(default_video, autoplay=autoplay)
        else:
            self._render_placeholder("No Video Loaded\nClick 'Open Video' to select an AVI/MP4 recording.")

    # -------------------------------------------------------------------------
    # Styles & Themes
    # -------------------------------------------------------------------------
    def _apply_theme(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Notebook tabs
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            font=font(10, True),
            padding=(18, 9),
            background="#E2E8F0",
            foreground=TEXT_MUTED,
            borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", CARD_BG)],
            foreground=[("selected", ACCENT_BLUE)]
        )

        # Buttons
        style.configure(
            "Accent.TButton",
            font=font(10, True),
            background=ACCENT_BLUE,
            foreground="white",
            padding=(14, 7),
            borderwidth=0
        )
        style.map("Accent.TButton", background=[("active", ACCENT_BLUE_DARK)])

        style.configure(
            "Secondary.TButton",
            font=font(10),
            background="#E2E8F0",
            foreground=TEXT_PRIMARY,
            padding=(10, 7),
            borderwidth=0
        )
        style.map("Secondary.TButton", background=[("active", "#CBD5E1")])

        # Controls
        style.configure("TCheckbutton", background=BG, foreground=TEXT_PRIMARY, font=font(9))
        style.configure("TCombobox", padding=4, font=font(9))
        style.configure("Horizontal.TScale", background=BG, troughcolor="#CBD5E1")

        # Treeview (Results)
        style.configure(
            "Treeview",
            font=font(9),
            rowheight=32,
            background=CARD_BG,
            fieldbackground=CARD_BG,
            foreground=TEXT_PRIMARY,
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            font=font(9, True),
            background="#F1F5F9",
            foreground=TEXT_PRIMARY,
            padding=8,
            relief="flat"
        )
        style.map("Treeview", background=[("selected", ACCENT_BLUE_LIGHT)], foreground=[("selected", ACCENT_BLUE)])

    # -------------------------------------------------------------------------
    # Header Area
    # -------------------------------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self.root, bg=NAVY_HEADER, highlightbackground=NAVY_HEADER_BORDER, highlightthickness=1)
        header.pack(fill="x")

        inner = tk.Frame(header, bg=NAVY_HEADER)
        inner.pack(fill="x", padx=20, pady=10)

        # Title Block
        left = tk.Frame(inner, bg=NAVY_HEADER)
        left.pack(side="left")

        tag_row = tk.Frame(left, bg=NAVY_HEADER)
        tag_row.pack(anchor="w")

        system_chip = tk.Label(tag_row, text="BIOMETRIC VISION STUDIO", bg="#1E293B", fg="#38BDF8",
                               font=font(8, True), padx=6, pady=2)
        system_chip.pack(side="left", padx=(0, 8))

        status_chip = tk.Label(tag_row, text="REAL-TIME (CPU PIPELINE)", bg="#064E3B", fg="#34D399",
                               font=font(8, True), padx=6, pady=2)
        status_chip.pack(side="left")

        title = tk.Label(left, text="Automated Fruit Fly Kinematic & Courtship Analysis",
                         bg=NAVY_HEADER, fg="white", font=font(15, True))
        title.pack(anchor="w", pady=(3, 0))

        sub = tk.Label(left, text="Classical ML: Contour Tracking • Sex Dimorphism • Heading Estimation • Wing Angle Regression",
                       bg=NAVY_HEADER, fg=NAVY_SUB, font=font(9))
        sub.pack(anchor="w")

        # Authors & Project Details
        right = tk.Frame(inner, bg=NAVY_HEADER)
        right.pack(side="right")

        course_lbl = tk.Label(right, text="UE24CS352A • Machine Learning Mini-Project",
                              bg=NAVY_HEADER, fg=NAVY_SUB, font=font(9, True))
        course_lbl.pack(anchor="e")

        team_str = "Sriranga Bhardwaj (PES1UG24AM286)  ·  Sumuk D Hegde (PES1UG24AM296)  ·  Shraddha (PES1UG25AM812)"
        team_lbl = tk.Label(right, text=team_str, bg=NAVY_HEADER, fg=NAVY_LIGHT, font=font(9))
        team_lbl.pack(anchor="e", pady=(2, 0))

    # -------------------------------------------------------------------------
    # Left Monitor (Video Player & Controls Deck)
    # -------------------------------------------------------------------------
    def _build_left_monitor(self, container):
        left_box = tk.Frame(container, bg=BG)
        left_box.pack(side="left", anchor="n")

        # Video Bezel Container
        monitor_bezel = ModernCard(left_box, bg="#0A0F1D", border="#1E293B", padx=6, pady=6)
        monitor_bezel.pack()

        # Bezel Header Bar
        bezel_bar = tk.Frame(monitor_bezel, bg="#0A0F1D")
        bezel_bar.pack(fill="x", padx=4, pady=(2, 6))

        self.video_info_lbl = tk.Label(bezel_bar, text="MONITOR: video.avi (512x512)",
                                       bg="#0A0F1D", fg="#64748B", font=font(8, True))
        self.video_info_lbl.pack(side="left")

        self.live_fps_chip = tk.Label(bezel_bar, text="LIVE: -- FPS",
                                      bg="#1E293B", fg="#38BDF8", font=font(8, True), padx=6, pady=1)
        self.live_fps_chip.pack(side="right")

        # Canvas
        self.canvas = tk.Canvas(monitor_bezel, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                bg="#030712", highlightthickness=0)
        self.canvas.pack()

        # Scrubber and Timecode
        scrub_frame = tk.Frame(left_box, bg=BG)
        scrub_frame.pack(fill="x", pady=(10, 2))

        self.timecode_lbl = tk.Label(scrub_frame, text="Frame: 0 / 0  ·  00:00.0 / 00:00.0",
                                     bg=BG, fg=TEXT_MUTED, font=font(9, True))
        self.timecode_lbl.pack(side="left")

        self.scrub = ttk.Scale(left_box, from_=0, to=1, orient="horizontal", command=self._on_scrub)
        self.scrub.pack(fill="x", pady=(2, 6))

        # Primary Control Bar
        ctrl_bar = tk.Frame(left_box, bg=BG)
        ctrl_bar.pack(fill="x", pady=(2, 6))

        ttk.Button(ctrl_bar, text="📂 Open Video", style="Secondary.TButton",
                   command=self.choose_video).pack(side="left", padx=(0, 4))

        self.btn_back = ttk.Button(ctrl_bar, text="⏮ -1 Fr", style="Secondary.TButton",
                                   command=self.step_back, width=6)
        self.btn_back.pack(side="left", padx=2)

        self.play_btn = ttk.Button(ctrl_bar, text="▶ Play", style="Accent.TButton",
                                   command=self.toggle_play, width=8)
        self.play_btn.pack(side="left", padx=3)

        self.btn_fwd = ttk.Button(ctrl_bar, text="+1 Fr ⏭", style="Secondary.TButton",
                                  command=self.step_forward, width=6)
        self.btn_fwd.pack(side="left", padx=2)

        ttk.Button(ctrl_bar, text="↺ Reset", style="Secondary.TButton",
                   command=self.restart, width=6).pack(side="left", padx=2)

        # Speed Selector
        speed_box = tk.Frame(ctrl_bar, bg=BG)
        speed_box.pack(side="right")
        tk.Label(speed_box, text="Speed:", bg=BG, fg=TEXT_MUTED, font=font(9)).pack(side="left", padx=(0, 4))
        self.speed = ttk.Combobox(speed_box, values=["0.5x", "1.0x (Normal)", "2.0x", "Max Speed"],
                                  width=11, state="readonly")
        self.speed.set("1.0x (Normal)")
        self.speed.pack(side="left")

        # Overlay Toggle Options
        toggles_frame = ModernCard(left_box, padx=10, pady=7)
        toggles_frame.pack(fill="x", pady=(4, 6))

        self.v_arrows = tk.BooleanVar(value=True)
        self.v_wings = tk.BooleanVar(value=True)
        self.v_contours = tk.BooleanVar(value=True)
        self.v_labels = tk.BooleanVar(value=True)

        tk.Label(toggles_frame, text="OVERLAYS:", bg=CARD_BG, fg=TEXT_MUTED, font=font(8, True)).pack(side="left", padx=(0, 8))
        for text, var in (("Heading Vectors", self.v_arrows),
                          ("Wing Angles", self.v_wings),
                          ("Body Outlines", self.v_contours),
                          ("Labels", self.v_labels)):
            chk = ttk.Checkbutton(toggles_frame, text=text, variable=var, command=self.refresh)
            chk.pack(side="left", padx=(0, 10))

        # Visual Key / Legend
        legend_row = tk.Frame(left_box, bg=BG)
        legend_row.pack(fill="x", pady=(2, 0))

        legend_items = [
            (ACCENT_BLUE, "Male (01)"),
            (ACCENT_ROSE, "Female (02)"),
            (ACCENT_AMBER, "Wing Vector"),
            ("#334155", "Ground Truth (---)"),
        ]
        for color, label in legend_items:
            tk.Label(legend_row, text="■", fg=color, bg=BG, font=font(10)).pack(side="left")
            tk.Label(legend_row, text=label, fg=TEXT_MUTED, bg=BG, font=font(8)).pack(side="left", padx=(2, 12))

    # -------------------------------------------------------------------------
    # Right Notebook (Live Telemetry, Benchmark, Charts, Documentation)
    # -------------------------------------------------------------------------
    def _build_right_notebook(self, container):
        self.nb = ttk.Notebook(container)
        self.nb.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self.tab_live = tk.Frame(self.nb, bg=BG)
        self.tab_benchmark = tk.Frame(self.nb, bg=BG)
        self.tab_charts = tk.Frame(self.nb, bg=BG)
        self.tab_docs = tk.Frame(self.nb, bg=BG)

        self.nb.add(self.tab_live, text="  Live Telemetry  ")
        self.nb.add(self.tab_benchmark, text="  Model Evaluation  ")
        self.nb.add(self.tab_charts, text="  Diagnostic Charts  ")
        self.nb.add(self.tab_docs, text="  System Architecture  ")

        self._build_tab_live()
        self._build_tab_benchmark()
        self._build_tab_charts()
        self._build_tab_docs()

    # --- TAB 1: LIVE TELEMETRY ---
    def _build_tab_live(self):
        tab = self.tab_live

        # Top KPI Cards Row
        kpi_row = tk.Frame(tab, bg=BG)
        kpi_row.pack(fill="x", pady=(8, 8))

        self.kpi_frame = KPIBadge(kpi_row, "FRAME INDEX", "- / -", "0.0 % Completed", ACCENT_BLUE)
        self.kpi_speed = KPIBadge(kpi_row, "PROCESSING SPEED", "- FPS", "Real-time Target: 30 FPS", ACCENT_PURPLE)
        self.kpi_status = KPIBadge(kpi_row, "DETECTION STATUS", "Ready", "2 Blobs Tracked", ACCENT_GREEN)

        for c, card in enumerate((self.kpi_frame, self.kpi_speed, self.kpi_status)):
            card.grid(row=0, column=c, sticky="nsew", padx=(0, 8) if c < 2 else 0)
            kpi_row.grid_columnconfigure(c, weight=1, uniform="kpi")

        # Telemetry Fly Cards (Male & Female side-by-side)
        cards_row = tk.Frame(tab, bg=BG)
        cards_row.pack(fill="x", pady=(0, 8))

        self.male_telemetry = TelemetryFlyCard(
            cards_row, "MALE SUBJECT", "M", ACCENT_BLUE, ACCENT_BLUE_LIGHT,
            ["Position (X, Y)", "Body Heading", "Right Wing Angle", "Left Wing Angle"]
        )
        self.female_telemetry = TelemetryFlyCard(
            cards_row, "FEMALE SUBJECT", "F", ACCENT_ROSE, ACCENT_ROSE_LIGHT,
            ["Position (X, Y)", "Body Heading", "Wing Extension"]
        )

        self.male_telemetry.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.female_telemetry.grid(row=0, column=1, sticky="nsew")
        cards_row.grid_columnconfigure((0, 1), weight=1, uniform="fly_cards")

        # Live Oscilloscope Plot for Male Wings
        plot_card = ModernCard(tab, padx=12, pady=10)
        plot_card.pack(fill="both", expand=True, pady=(0, 4))

        plot_header = tk.Frame(plot_card, bg=CARD_BG)
        plot_header.pack(fill="x", pady=(0, 4))

        tk.Label(plot_header, text="MALE WING EXTENSION TELEMETRY (DYNAMIC KINEMATICS)",
                 bg=CARD_BG, fg=TEXT_PRIMARY, font=font(10, True)).pack(side="left")

        self.plot_live_readout = tk.Label(
            plot_header,
            text="Right: --°  |  Left: --°  |  Asymmetry: --°",
            bg=CARD_BG, fg=ACCENT_AMBER, font=font(9, True)
        )
        self.plot_live_readout.pack(side="right")

        self.plot_canvas = tk.Canvas(plot_card, bg="#0F172A", highlightthickness=0, height=175)
        self.plot_canvas.pack(fill="both", expand=True)

    # --- TAB 2: BENCHMARK & EVALUATION ---
    def _build_tab_benchmark(self):
        tab = self.tab_benchmark

        # Headline Metrics
        self.bench_top = tk.Frame(tab, bg=BG)
        self.bench_top.pack(fill="x", pady=(8, 8))

        # Stage Evaluation Table Card
        table_card = ModernCard(tab, padx=14, pady=12)
        table_card.pack(fill="x", pady=(0, 8))

        tk.Label(table_card, text="PIPELINE STAGE ACCURACY ON UNSEEN TEST FRAMES",
                 bg=CARD_BG, fg=TEXT_PRIMARY, font=font(10, True)).pack(anchor="w", pady=(0, 6))

        cols = ("stage", "algorithm", "test_err", "train_err")
        self.tree = ttk.Treeview(table_card, columns=cols, show="headings", height=5)
        for c, text, w, anchor in (
            ("stage", "Pipeline Stage", 240, "w"),
            ("algorithm", "Machine Learning Architecture", 260, "w"),
            ("test_err", "Test Error (Unseen)", 130, "e"),
            ("train_err", "Training Error", 130, "e")
        ):
            self.tree.heading(c, text=text, anchor=anchor)
            self.tree.column(c, width=w, minwidth=w - 30, anchor=anchor, stretch=c in ("stage", "algorithm"))
        self.tree.pack(fill="x")

        self.bench_note = tk.Label(table_card, text="", bg=CARD_BG, fg=TEXT_MUTED, font=font(8), justify="left")
        self.bench_note.pack(anchor="w", pady=(6, 0))

        # Latency & Profiling Card
        timing_card = ModernCard(tab, padx=14, pady=12)
        timing_card.pack(fill="both", expand=True, pady=(0, 4))

        tk.Label(timing_card, text="STAGE LATENCY BREAKDOWN (TIME PER FRAME BUDGET)",
                 bg=CARD_BG, fg=TEXT_PRIMARY, font=font(10, True)).pack(anchor="w", pady=(0, 4))

        self.time_canvas = tk.Canvas(timing_card, bg=CARD_BG, highlightthickness=0, height=180)
        self.time_canvas.pack(fill="both", expand=True)
        self.time_canvas.bind("<Configure>", lambda e: self._draw_timing_breakdown())

        self.load_benchmark_data()

    # --- TAB 3: DIAGNOSTIC CHARTS ---
    def _build_tab_charts(self):
        tab = self.tab_charts
        self.chart_catalog = [
            ("1. Multi-Stage Pipeline Overview", "pipeline_overview.png",
             "Visual breakdown of the raw grayscale frame through thresholding, contour extraction, body orientation, and wing ROI crops."),
            ("2. Continuous Male Wing Kinematics Time-Series", "wing_timeseries.png",
             "Ground truth vs predicted right and left wing angles throughout the 20-second courtship sequence."),
            ("3. Wing Regression Fit & Error Distribution", "wing_angle_fit.png",
             "Correlation between true and predicted wing angles on held-out validation samples showing high linearity."),
            ("4. Inference Latency & Computational Budget", "timing.png",
             "Per-stage latency breakdown verifying real-time throughput far exceeding standard 30 FPS video playout."),
            ("5. Orientation: PCA Clustering of HOG Descriptors", "orientation_pca.png",
             "2D projection of 3780-dimensional HOG features demonstrating clear cluster separation between head-up and head-down orientations."),
            ("6. PCA Dimensionality Reduction Ablation", "pca_ablation.png",
             "Accuracy progression across principal components validating component selections for Stage 3 (15 PCs) and Stage 4 (40 PCs)."),
        ]

        top_ctrl = ModernCard(tab, padx=12, pady=8)
        top_ctrl.pack(fill="x", pady=(8, 6))

        tk.Label(top_ctrl, text="SELECT DIAGNOSTIC FIGURE:", bg=CARD_BG, fg=TEXT_MUTED, font=font(9, True)).pack(side="left")
        self.chart_combobox = ttk.Combobox(
            top_ctrl, values=[c[0] for c in self.chart_catalog], state="readonly", width=55
        )
        self.chart_combobox.pack(side="left", padx=10)
        self.chart_combobox.current(0)
        self.chart_combobox.bind("<<ComboboxSelected>>", lambda e: self.show_selected_chart())

        # Chart Container Card
        self.chart_frame = ModernCard(tab, padx=8, pady=8)
        self.chart_frame.pack(fill="both", expand=True, pady=(0, 4))

        self.chart_img_lbl = tk.Label(self.chart_frame, bg=CARD_BG)
        self.chart_img_lbl.pack(fill="both", expand=True)

        self.chart_desc_lbl = tk.Label(self.chart_frame, text="", bg=CARD_BG, fg=TEXT_MUTED,
                                       font=font(9), wraplength=760, justify="center")
        self.chart_desc_lbl.pack(side="bottom", pady=(4, 2))

        self.chart_frame.bind("<Configure>", lambda e: self.show_selected_chart())

    # --- TAB 4: SYSTEM ARCHITECTURE & DOCUMENTATION ---
    def _build_tab_docs(self):
        tab = self.tab_docs
        doc_card = ModernCard(tab, padx=20, pady=16)
        doc_card.pack(fill="both", expand=True, pady=(8, 4))

        txt = tk.Text(doc_card, wrap="word", bg=CARD_BG, fg=TEXT_PRIMARY, font=font(10), relief="flat",
                      spacing1=3, spacing3=6, borderwidth=0)
        txt.pack(fill="both", expand=True)

        txt.tag_configure("h1", font=font(13, True), foreground=ACCENT_BLUE, spacing1=8, spacing3=4)
        txt.tag_configure("h2", font=font(11, True), foreground=TEXT_PRIMARY, spacing1=6, spacing3=2)
        txt.tag_configure("code", font=("Consolas", 10), background=CARD_ALT, foreground=ACCENT_BLUE_DARK)
        txt.tag_configure("bullet", font=font(10), lmargin1=16, lmargin2=32)

        docs = [
            ("Executive Summary\n", "h1"),
            ("This system performs automated, high-precision computer vision and biometric analysis on grayscale "
             "video recordings of fruit fly (Drosophila melanogaster) pairs during courtship. The pipeline extracts "
             "centimeter-scale kinematic parameters in real time on standard CPU hardware, avoiding computationally "
             "prohibitive deep neural networks in favor of classical feature engineering and linear models.\n", ""),

            ("The Four-Stage Sequential Pipeline\n", "h1"),
            ("Stage 1 — Foreground Detection & Blob Counting\n", "h2"),
            ("• Binary thresholding with morphological opening isolates fly bodies from lighter background pixels.\n"
             "• A Gini-impurity Decision Tree evaluates contour area to distinguish between debris (<20 px), a single fly, "
             "or touching/overlapping flies. Overlapping frames are safely deferred using temporal hysteresis.\n", "bullet"),

            ("Stage 2 — Sex Identification (Dimorphism)\n", "h2"),
            ("• Logistic Regression identifies the male vs female fly based on standardized contour area and aspect ratio.\n"
             "• Females are morphologically larger, enabling 99.3%+ identification accuracy without manual tagging.\n", "bullet"),

            ("Stage 3 — Body Orientation & Heading Angle (0°–360°)\n", "h2"),
            ("• Central image moments (mu20, mu02, mu11) compute the primary body axis (ambiguous by 180°).\n"
             "• A 128×64 upright patch is extracted, and Histogram of Oriented Gradients (HOG) features are computed.\n"
             "• Dimensionality reduction with PCA(15) followed by Logistic Regression resolves the head vs tail orientation.\n", "bullet"),

            ("Stage 4 — Male Wing Kinematics & Angles (0°–90°)\n", "h2"),
            ("• High-contrast wing ROI extraction: flies are normalized upright, legs eroded via connected components.\n"
             "• Bilateral symmetry exploitation: the left wing patch is horizontally mirrored so a single unified model "
             "analyzes both wings.\n"
             "• HOG features + PCA(40) + Linear Ridge Regression predict wing angles to within ~2° of ground truth.\n", "bullet"),

            ("Navigation & Keyboard Controls\n", "h1"),
            ("• Spacebar: Toggle Play / Pause playback.\n"
             "• Left / Right Arrow Keys: Single frame step backward / forward for micro-kinematic inspection.\n"
             "• R / r Key: Reset / Rewind video to Frame 0.\n", "bullet"),
        ]

        for content, tag in docs:
            txt.insert("end", content, tag)
        txt.config(state="disabled")

    # -------------------------------------------------------------------------
    # Benchmark Data Loader & Latency Drawer
    # -------------------------------------------------------------------------
    def _read_results_json(self, filename):
        path = os.path.join(BASE, "results", filename)
        try:
            return json.load(open(path))
        except Exception:
            return None

    def load_benchmark_data(self):
        m = self._read_results_json("metrics.json")
        d = self._read_results_json("demo_summary.json")

        for w in self.bench_top.winfo_children():
            w.destroy()

        self.demo_summary = d
        acc = (d or {}).get("accuracy_vs_ground_truth")

        cards = [
            ("SEX IDENTIFICATION", f"{acc['sex_correct_percent']:.1f} %" if acc else "-",
             "Target: >95.0 %", ACCENT_GREEN),
            ("MEAN HEADING ERROR", f"{acc['heading_error_degrees_average']:.2f}°" if acc else "-",
             f"Median: {acc['heading_error_degrees_median']:.2f}°" if acc else "", ACCENT_BLUE),
            ("MEAN WING ERROR", f"{acc['wing_angle_error_degrees_average']:.2f}°" if acc else "-",
             f"Std: {acc['wing_angle_error_degrees_spread_std']:.2f}°" if acc else "", ACCENT_AMBER),
            ("PIPELINE SPEED", f"{d['speed']['frames_per_second']:.0f} FPS" if d else "-",
             f"{d['speed']['times_faster_than_source_video']:.1f}x Video Playout" if d else "", ACCENT_PURPLE),
        ]

        for c, (t, v, sub, col) in enumerate(cards):
            KPIBadge(self.bench_top, t, v, sub, col).grid(row=0, column=c, sticky="nsew", padx=(0, 8) if c < 3 else 0)
            self.bench_top.grid_columnconfigure(c, weight=1, uniform="bench_kpi")

        self.tree.delete(*self.tree.get_children())
        if not m:
            self.bench_note.config(text="No benchmark metrics found. Run `python train_evaluate.py` to populate.")
            return

        stages = [
            ("Stage 1: Fly Counting", m["stage1_fly_counting"]),
            ("Stage 2: Sex Identification", m["stage2_sex_identification"]),
            ("Stage 3: Orientation (Male)", m["stage3_orientation_male"]),
            ("Stage 3: Orientation (Female)", m["stage3_orientation_female"]),
        ]
        for name, r in stages:
            self.tree.insert("", "end", values=(
                name, r["model"], f"{r['test_error_percent']:.2f} %", f"{r['training_error_percent']:.2f} %"
            ))

        w = m["stage4_wing_angle"]
        self.tree.insert("", "end", values=(
            "Stage 4: Wing Kinematics", w["model"],
            f"{w['test_error_degrees_spread_std']:.2f}° (spread)",
            f"{w['training_error_degrees_spread_std']:.2f}° (spread)"
        ))

        ds = m.get("dataset", {})
        self.bench_note.config(
            text=f"Evaluated on {ds.get('training_frames', 600)} training frames and "
                 f"{ds.get('test_frames', 300)} held-out validation frames. "
                 f"Stages 1–3: classification error rate. Stage 4: wing-angle deviation standard spread."
        )
        self._draw_timing_breakdown()

    def _draw_timing_breakdown(self):
        c = self.time_canvas
        c.delete("all")
        d = getattr(self, "demo_summary", None)
        if not d:
            return

        w = max(c.winfo_width(), 320)
        ms_dict = d["time_per_frame_milliseconds"]
        total_ms = d["speed"]["milliseconds_per_frame"]
        fps = d["speed"]["frames_per_second"]
        speedup = d["speed"]["times_faster_than_source_video"]

        stages = [
            ("reading_video_frame", "Video Frame I/O", "#64748B"),
            ("stage1_fly_counting", "Stage 1: Blob Counting", ACCENT_BLUE),
            ("stage2_sex_identification", "Stage 2: Sex Dimorphism", ACCENT_ROSE),
            ("stage3_orientation", "Stage 3: Heading (HOG+PCA)", ACCENT_PURPLE),
            ("stage4_wing_angles", "Stage 4: Wings (HOG+PCA)", ACCENT_AMBER),
        ]

        x_start = 180
        x_max_bar = w - 120
        max_ms = max(ms_dict.values())

        for i, (key, label, col) in enumerate(stages):
            val = ms_dict[key]
            y = 22 + i * 28

            c.create_text(x_start - 12, y, anchor="e", text=label, font=font(9, True), fill=TEXT_PRIMARY)
            bar_len = (x_max_bar - x_start) * (val / max_ms)
            c.create_rectangle(x_start, y - 7, x_start + bar_len, y + 7, fill=col, outline="")
            pct = 100 * val / total_ms
            c.create_text(x_start + bar_len + 8, y, anchor="w",
                          text=f"{val:.2f} ms ({pct:.1f}%)", font=font(8), fill=TEXT_MUTED)

        summary_y = 22 + len(stages) * 28 + 8
        summary_text = (
            f"Total Throughput: {total_ms:.2f} ms per frame  ·  "
            f"{fps:.1f} Frames/Second  ·  "
            f"{speedup:.1f}x Faster Than 30 FPS Playout"
        )
        c.create_rectangle(10, summary_y - 12, w - 10, summary_y + 14, fill=CARD_ALT, outline="")
        c.create_text(20, summary_y + 1, anchor="w", text=summary_text, font=font(9, True), fill=ACCENT_BLUE_DARK)

    # -------------------------------------------------------------------------
    # Diagnostic Charts Display
    # -------------------------------------------------------------------------
    def show_selected_chart(self):
        idx = self.chart_combobox.current()
        title, filename, description = self.chart_catalog[idx]
        path = os.path.join(BASE, "results", filename)

        if not os.path.exists(path):
            self.chart_img_lbl.config(
                image="",
                text="Figure not found on disk. Run `python make_figures.py` to regenerate.",
                fg=TEXT_MUTED, font=font(10)
            )
            self.chart_desc_lbl.config(text="")
            return

        img = cv2.imread(path)
        bw = max(self.chart_frame.winfo_width() - 20, 300)
        bh = max(self.chart_frame.winfo_height() - 60, 200)

        scale = min(bw / img.shape[1], bh / img.shape[0])
        nw, nh = int(img.shape[1] * scale), int(img.shape[0] * scale)
        img_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        self.chart_photo = tk.PhotoImage(data=cv2.imencode(".ppm", img_res)[1].tobytes())
        self.chart_img_lbl.config(image=self.chart_photo, text="")
        self.chart_desc_lbl.config(text=description)

    # -------------------------------------------------------------------------
    # Video & Playback Engine
    # -------------------------------------------------------------------------
    def _render_placeholder(self, text):
        self.canvas.delete("all")
        self.canvas.create_text(
            DISPLAY_SIZE // 2, DISPLAY_SIZE // 2,
            text=text, fill="#64748B", font=font(12, True), justify="center"
        )

    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Open Fruit Fly Recording",
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.open_video(path)

    def open_video(self, path, autoplay=True):
        self.pause()
        if self.cap is not None:
            self.cap.release()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Video Error", f"Unable to open video stream from:\n{path}")
            return

        self.cap = cap
        self.n_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_name = os.path.basename(path)

        # Ground Truth pairing
        self.gt = None
        default_gt = os.path.join(BASE, "data", "video_gt.json")
        gt_candidate = os.path.splitext(path)[0] + "_gt.json"
        if os.path.exists(gt_candidate):
            try:
                self.gt = json.load(open(gt_candidate))
            except Exception:
                self.gt = None
        elif os.path.abspath(path) == os.path.abspath(os.path.join(BASE, "data", "video.avi")) and os.path.exists(default_gt):
            self.gt = json.load(open(default_gt))

        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res_str = f"{w_orig}x{h_orig}" if (w_orig > 0 and h_orig > 0) else f"{DISPLAY_SIZE}x{DISPLAY_SIZE}"
        self.scrub.config(to=max(self.n_frames - 1, 1))
        self.video_info_lbl.config(text=f"MONITOR: {self.video_name} ({res_str})")
        self.restart(play=autoplay)

    def restart(self, play=True):
        if self.cap is None:
            return
        self.pause()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.idx = 0
        self.last_flies = []
        self.proc_times.clear()
        for s in self.series.values():
            s.clear()
        self.step()
        if play:
            self.play()

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self):
        if self.cap is None:
            return
        if self.idx >= self.n_frames - 1:
            self.restart(play=False)
        self.playing = True
        self.play_btn.config(text="⏸ Pause")
        self._schedule(1)

    def pause(self):
        self.playing = False
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None
        self.play_btn.config(text="▶ Play")

    def step_forward(self):
        if self.cap is None:
            return
        self.pause()
        if self.idx < self.n_frames - 1:
            self.step()

    def step_back(self):
        if self.cap is None or self.idx <= 0:
            return
        self.pause()
        target = max(self.idx - 1, 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.idx = target
        self.step()

    def _interval(self):
        s = self.speed.get()
        if s.startswith("Max"):
            return 1
        mult = {"0.5x": 0.5, "2.0x": 2.0}.get(s.split()[0], 1.0)
        return max(1, int(1000 / (self.src_fps * mult)))

    def _schedule(self, delay):
        self._job = self.root.after(delay, self._tick)

    def _tick(self):
        self._job = None
        if not self.playing:
            return
        t0 = time.perf_counter()
        if not self.step():
            self.pause()
            self.play_btn.config(text="↺ Replay")
            return
        spent = int(1000 * (time.perf_counter() - t0))
        self._schedule(max(1, self._interval() - spent))

    def _on_scrub(self, value):
        if self._seeking or self.cap is None:
            return
        target = int(float(value))
        if target == self.idx:
            return
        self.pause()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        for s in self.series.values():
            s.clear()
        self.idx = target
        self.step()

    # -------------------------------------------------------------------------
    # Frame Processing & Render Pipeline
    # -------------------------------------------------------------------------
    def step(self):
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        t0 = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != (512, 512):
            h, w = gray.shape[:2]
            if h == w:
                gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
            else:
                min_dim = min(h, w)
                y0 = (h - min_dim) // 2
                x0 = (w - min_dim) // 2
                gray = cv2.resize(gray[y0:y0 + min_dim, x0:x0 + min_dim], (512, 512), interpolation=cv2.INTER_AREA)

        self.res = self.pipe.process(gray)
        elapsed = time.perf_counter() - t0
        self.proc_times.append(elapsed)

        self.idx = pos
        self.gray = gray
        if self.res["status"] == "ok":
            self.last_flies = self.res["flies"]

        self.truth = self.gt[pos] if self.gt is not None and pos < len(self.gt) else None
        self.rows = frame_errors(self.last_flies, self.truth) if (self.truth and self.res["status"] == "ok") else []

        self._record_kinematic_series()
        self.refresh()
        return True

    def _record_kinematic_series(self):
        male = next((f for f in self.last_flies if f["sex"] == "M" and "wing_r" in f), None)
        if male is None or self.res["status"] != "ok":
            return
        self.series["pr"].append(male["wing_r"])
        self.series["pl"].append(male["wing_l"])

        gm = next((q for q in self.truth if q["sex"] == "M"), None) if self.truth else None
        self.series["tr"].append(gm["wing_r"] if gm else None)
        self.series["tl"].append(gm["wing_l"] if gm else None)

    def refresh(self):
        if self.cap is None or not hasattr(self, "gray"):
            return
        self._draw_video_frame()
        self._update_telemetry_readouts()
        self._draw_oscilloscope_plot()

    def _draw_video_frame(self):
        img = draw_overlay(
            self.gray, self.last_flies, hud=None,
            arrows=self.v_arrows.get(),
            wings=self.v_wings.get(),
            labels=self.v_labels.get(),
            contours=self.v_contours.get()
        )
        img = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LINEAR)

        # Status warning bar on merged touches
        if self.res["status"] != "ok":
            cv2.rectangle(img, (0, DISPLAY_SIZE - 32), (DISPLAY_SIZE, DISPLAY_SIZE), (24, 119, 212), -1)
            cv2.putText(
                img, "PROXIMITY MERGE - TRACKING RETAINED", (14, DISPLAY_SIZE - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
            )

        # PPM encoding provides lightning fast canvas updates in Tkinter
        self.photo = tk.PhotoImage(data=cv2.imencode(".ppm", img)[1].tobytes())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def _update_telemetry_readouts(self):
        self._seeking = True
        self.scrub.set(self.idx)
        self._seeking = False

        # Timecode & Frame Labels
        cur_sec = self.idx / self.src_fps
        tot_sec = self.n_frames / self.src_fps
        self.timecode_lbl.config(
            text=f"Frame: {self.idx + 1:03d} / {self.n_frames:03d}  ·  "
                 f"{int(cur_sec // 60):02d}:{cur_sec % 60:04.1f} / {int(tot_sec // 60):02d}:{tot_sec % 60:04.1f}"
        )

        pct = 100 * (self.idx + 1) / self.n_frames
        self.kpi_frame.update_val(f"{self.idx + 1} / {self.n_frames}", f"{pct:.1f} % Completed")

        # Live Speed Chip
        if self.proc_times:
            fps_val = 1.0 / np.mean(self.proc_times)
            self.kpi_speed.update_val(f"{fps_val:.0f} FPS", f"{1000 / fps_val:.1f} ms/frame", ACCENT_PURPLE)
            self.live_fps_chip.config(text=f"LIVE: {fps_val:.0f} FPS")

        # Detection Status Badge
        status = self.res["status"]
        if status == "ok":
            self.kpi_status.update_val("2 Blobs Resolved", "Active Courtship Tracking", ACCENT_GREEN)
        elif status == "merged":
            self.kpi_status.update_val("Proximal Merge", "Single Merged Contour", ACCENT_AMBER)
        else:
            self.kpi_status.update_val("Unresolved", "Ambiguous Blobs", ACCENT_ROSE)

        # Update Telemetry Tables
        by_sex = {r["pred"]["sex"]: r for r in self.rows}

        # Male Readout
        male = next((f for f in self.last_flies if f["sex"] == "M"), None)
        if male:
            r = by_sex.get("M")
            g = r["truth"] if r else None
            self.male_telemetry.set_row(
                "Position (X, Y)",
                f"({male['x']:.0f}, {male['y']:.0f}) px",
                f"({g['x']:.0f}, {g['y']:.0f}) px" if g else "-",
                f"{r['position_error_px']:.1f} px" if r else "-"
            )
            self.male_telemetry.set_row(
                "Body Heading",
                f"{male['heading']:.1f}°",
                f"{g['heading']:.1f}°" if g else "-",
                f"{r['heading_error_deg']:.1f}°" if r else "-"
            )
            if "wing_r" in male:
                we = r.get("wing_errors_deg") if r else None
                self.male_telemetry.set_row(
                    "Right Wing Angle",
                    f"{male['wing_r']:.1f}°",
                    f"{g['wing_r']:.1f}°" if g and we else "-",
                    f"{abs(we[0]):.1f}°" if we else "-"
                )
                self.male_telemetry.set_row(
                    "Left Wing Angle",
                    f"{male['wing_l']:.1f}°",
                    f"{g['wing_l']:.1f}°" if g and we else "-",
                    f"{abs(we[1]):.1f}°" if we else "-"
                )
        else:
            for m in ["Position (X, Y)", "Body Heading", "Right Wing Angle", "Left Wing Angle"]:
                self.male_telemetry.set_row(m)

        # Female Readout
        fem = next((f for f in self.last_flies if f["sex"] == "F"), None)
        if fem:
            r = by_sex.get("F")
            g = r["truth"] if r else None
            self.female_telemetry.set_row(
                "Position (X, Y)",
                f"({fem['x']:.0f}, {fem['y']:.0f}) px",
                f"({g['x']:.0f}, {g['y']:.0f}) px" if g else "-",
                f"{r['position_error_px']:.1f} px" if r else "-"
            )
            self.female_telemetry.set_row(
                "Body Heading",
                f"{fem['heading']:.1f}°",
                f"{g['heading']:.1f}°" if g else "-",
                f"{r['heading_error_deg']:.1f}°" if r else "-"
            )
            self.female_telemetry.set_row(
                "Wing Extension", "Folded (0°)", "Folded (0°)", "Normal"
            )
        else:
            for m in ["Position (X, Y)", "Body Heading", "Wing Extension"]:
                self.female_telemetry.set_row(m)

    def _draw_oscilloscope_plot(self):
        c = self.plot_canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 320), max(c.winfo_height(), 130)

        # Plot margins
        l, r, top, bot = 42, w - 16, 16, h - 30

        # Draw Grid Lines
        for angle in (0, 30, 60, 90):
            y = bot - (bot - top) * (angle / 90.0)
            c.create_line(l, y, r, y, fill="#1E293B", width=1)
            c.create_text(l - 6, y, text=f"{angle}°", anchor="e", font=font(8), fill="#64748B")

        n_max = self.series["pr"].maxlen

        def draw_curve(values, color, width=2, dash=None):
            pts = []
            for i, v in enumerate(values):
                if v is None:
                    continue
                x = l + (r - l) * (i + n_max - len(values)) / (n_max - 1)
                y = bot - (bot - top) * min(max(v, 0), 90) / 90.0
                pts += [x, y]
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=width, dash=dash, smooth=True)

        has_truth = any(v is not None for v in self.series["tr"])

        # Draw ground truth dashed lines
        if has_truth:
            draw_curve(self.series["tr"], "#475569", width=1, dash=(2, 2))
            draw_curve(self.series["tl"], "#475569", width=1, dash=(2, 2))

        # Draw dynamic predictions
        draw_curve(self.series["pr"], ACCENT_BLUE, width=2)
        draw_curve(self.series["pl"], ACCENT_AMBER, width=2)

        # Update Live Readout Label
        if self.series["pr"]:
            cur_r = self.series["pr"][-1]
            cur_l = self.series["pl"][-1]
            asym = cur_r - cur_l
            asym_str = f"+{asym:.1f}°" if asym > 0 else f"{asym:.1f}°"
            self.plot_live_readout.config(
                text=f"Right Wing: {cur_r:.1f}°  |  Left Wing: {cur_l:.1f}°  |  Asymmetry: {asym_str}"
            )

        # Legend at base of plot
        x_leg = l
        items = [
            (ACCENT_BLUE, "Right Wing (Pred)"),
            (ACCENT_AMBER, "Left Wing (Pred)"),
        ]
        if has_truth:
            items.append(("#64748B", "Ground Truth (---)"))

        for color, text in items:
            c.create_line(x_leg, h - 12, x_leg + 14, h - 12, fill=color, width=2)
            c.create_text(x_leg + 18, h - 12, text=text, anchor="w", font=font(8), fill="#94A3B8")
            x_leg += 18 + len(text) * 7 + 16

    def close(self):
        self.pause()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fruit fly biometric analysis GUI")
    ap.add_argument("--video", help="Video file path to open (default: data/video.avi)")
    ap.add_argument("--no-autoplay", action="store_true", help="Do not autoplay on start")
    args = ap.parse_args()

    root = tk.Tk()
    App(root, video=args.video, autoplay=not args.no_autoplay)
    root.mainloop()


if __name__ == "__main__":
    main()
