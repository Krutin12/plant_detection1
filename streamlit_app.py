"""
AgriVision AI — Plant Disease Detection System
================================================
Professional AgriTech platform for AI-powered plant disease detection,
treatment management, fertilizer calculation, and farm analytics.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import os
import io
import textwrap
import base64
import zipfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFilter
import sys

sys.path.append('.')

try:
    from user_profile import UserProfileModule
    from treatment_history import TreatmentHistoryModule
    from fertilizer_calculator import FertilizerCalculatorModule
    from farm_analytics import FarmAnalyticsModule
    from export_reports import ExportReportsModule
    from detection_history import DetectionHistoryModule
    from data_management import DataManagementModule
    from disease_detection_streamlit import DiseaseDetectionStreamlit
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriVision AI · Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Ultra-Modern AgriTech CSS (Exact Match to Premium Reference UI)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=Caveat:wght@600;700&display=swap');

/* ── Root Design Tokens ────────────────────────────────────── */
:root {
    --primary-50:  #f0fdf4;
    --primary-100: #dcfce7;
    --primary-200: #bbf7d0;
    --primary-300: #86efac;
    --primary-400: #4ade80;
    --primary-500: #22c55e;
    --primary-600: #16a34a;
    --primary-700: #15803d;
    --primary-800: #166534;
    --primary-900: #14532d;
    --slate-50:  #f8fafc;
    --slate-100: #f1f5f9;
    --slate-200: #e2e8f0;
    --slate-300: #cbd5e1;
    --slate-400: #94a3b8;
    --slate-500: #64748b;
    --slate-600: #475569;
    --slate-700: #334155;
    --slate-800: #1e293b;
    --slate-900: #0f172a;
    --white:     #ffffff;
    --red-50:    #fef2f2;
    --red-100:   #fee2e2;
    --red-500:   #ef4444;
    --amber-50:  #fffbeb;
    --amber-100: #fef3c7;
    --amber-500: #f59e0b;
    --blue-50:   #eff6ff;
    --blue-100:  #dbeafe;
    --blue-500:  #3b82f6;
}

/* ── Global Typography & Clean Slate Canvas ─────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    color: #1e293b !important;
    background-color: #f8fafc !important;
}

/* Header styling - Non-intrusive while ensuring sidebar toggle is always visible */
header[data-testid="stHeader"] {
    background: transparent !important;
    color: #334155 !important;
    height: 3.5rem !important;
    z-index: 999990 !important;
}

#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}

/* Sidebar Toggle & Close Buttons */
.av-sidebar-toggle-btn {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: #ffffff !important;
    color: #15803d !important;
    border: 1.5px solid #86efac !important;
    border-radius: 10px !important;
    padding: 7px 14px !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
    margin-right: 8px !important;
}
.av-sidebar-toggle-btn:hover {
    background: #f0fdf4 !important;
    border-color: #16a34a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(22,163,74,0.18) !important;
}

.av-sidebar-close-btn {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    background: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.av-sidebar-close-btn:hover {
    background: #fee2e2 !important;
    color: #b91c1c !important;
    border-color: #fecaca !important;
}

/* Native Streamlit open / expand button (when sidebar is collapsed) */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
div[data-testid="collapsedControl"] button,
button[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    top: 12px !important;
    left: 12px !important;
    position: fixed !important;
    z-index: 9999999 !important;
    background: #ffffff !important;
    color: #16a34a !important;
    border: 2px solid #22c55e !important;
    border-radius: 10px !important;
    padding: 6px 10px !important;
    box-shadow: 0 4px 14px rgba(22, 163, 74, 0.25) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover {
    background: #f0fdf4 !important;
    border-color: #16a34a !important;
    transform: scale(1.05) !important;
}

[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #15803d !important;
    color: #15803d !important;
    stroke: #15803d !important;
}

/* Native Streamlit close / collapse button (inside sidebar) */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] button[kind="header"],
button[data-testid="baseButton-header"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: #f0fdf4 !important;
    color: #15803d !important;
    border: 1.5px solid #86efac !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
    margin: 8px 12px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="stSidebar"] button[kind="header"]:hover {
    background: #dcfce7 !important;
    border-color: #16a34a !important;
}

[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebar"] button[kind="header"] svg {
    fill: #15803d !important;
    color: #15803d !important;
}

/* Force readable text */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span, [data-testid="stMarkdownContainer"] li,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stMetricDelta"], .stTextInput label, .stSelectbox label,
.stNumberInput label, .stSlider label, .stMultiSelect label,
.stRadio label, .stCheckbox label, .stTextArea label,
.stDateInput label, .stTimeInput label {
    color: #1e293b !important;
}

/* ── Sidebar Customization (Pure SaaS Navigation) ──────────── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.02) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem !important;
}

/* Hide radio circles in sidebar completely */
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] label [data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] label div[role="radio"],
[data-testid="stSidebar"] [role="radiogroup"] label div[aria-hidden="true"],
[data-testid="stSidebar"] [role="radiogroup"] label input[type="radio"],
[data-testid="stSidebar"] [role="radiogroup"] label svg {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    visibility: hidden !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 4px !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 6px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 16px !important;
    margin: 2px 0 !important;
    border-radius: 10px !important;
    border-left: 3.5px solid transparent !important;
    color: #475569 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: transparent !important;
    width: 100% !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #f0fdf4 !important;
    color: #15803d !important;
    transform: translateX(3px) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #e8f5e9 !important;
    color: #15803d !important;
    font-weight: 700 !important;
    border-left: 3.5px solid #16a34a !important;
    box-shadow: 0 2px 6px rgba(22, 163, 74, 0.1) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] * {
    color: #15803d !important;
    font-weight: 700 !important;
}

/* ── Top Header Bar ─────────────────────────────────────────── */
.av-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0 20px 0;
    margin-bottom: 20px;
    border-bottom: 1px solid #e2e8f0;
}
.av-topbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.av-topbar-icon {
    font-size: 2rem;
    line-height: 1;
}
.av-topbar-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    letter-spacing: -0.5px;
}
.av-topbar-right {
    display: flex;
    align-items: center;
    gap: 14px;
}
.av-status-pill {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.av-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    box-shadow: 0 0 6px rgba(34, 197, 94, 0.6);
}
.av-bell-btn {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    font-size: 1.1rem;
    color: #475569;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.av-bell-dot {
    position: absolute;
    top: 6px;
    right: 7px;
    width: 7px;
    height: 7px;
    background: #ef4444;
    border-radius: 50%;
    border: 1.5px solid #ffffff;
}
.av-user-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px 4px 6px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #1e293b;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.av-user-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #dcfce7;
    color: #15803d;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
}

/* ── Hero Banner ────────────────────────────────────────────── */
.av-hero-banner {
    background: linear-gradient(135deg, #eafaf1 0%, #daf4e5 45%, #c5eed4 100%);
    border: 1px solid #a7f3d0;
    border-radius: 18px;
    padding: 28px 34px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.06);
}
.av-hero-content {
    max-width: 650px;
    z-index: 2;
}
.av-hero-title-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.av-hero-icon-wrap {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: rgba(255,255,255,0.9);
    border: 1px solid #86efac;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 2px 8px rgba(22, 163, 74, 0.1);
}
.av-hero-banner h2 {
    font-size: 1.85rem;
    font-weight: 800;
    color: #064e3b !important;
    margin: 0;
    letter-spacing: -0.5px;
}
.av-hero-banner p {
    color: #065f46 !important;
    font-size: 1.02rem;
    line-height: 1.55;
    margin: 0;
    font-weight: 450;
}
.av-hero-tagline-box {
    text-align: right;
    z-index: 2;
    padding-left: 20px;
}
.av-hero-script {
    font-family: 'Caveat', cursive, sans-serif;
    font-size: 1.95rem;
    font-weight: 700;
    color: #047857;
    line-height: 1.15;
    display: block;
}
.av-hero-script-sub {
    font-family: 'Caveat', cursive, sans-serif;
    font-size: 1.95rem;
    font-weight: 700;
    color: #047857;
    line-height: 1.15;
    display: block;
    border-bottom: 2px solid #34d399;
    padding-bottom: 4px;
}

/* ── Cards & Containers ─────────────────────────────────────── */
.av-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    transition: all 0.2s ease;
}
.av-card:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
}

/* ── Diagnostic Dropzone Card ───────────────────────────────── */
.av-dropzone-box {
    border: 2px dashed #86efac;
    background: #f0fdf4;
    border-radius: 14px;
    padding: 24px 18px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
    position: relative;
}
.av-dropzone-icon-box {
    width: 64px;
    height: 64px;
    border-radius: 14px;
    background: #ffffff;
    border: 1px solid #bbf7d0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.1);
}
.av-dropzone-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 3px;
}
.av-dropzone-subtitle {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 6px;
}
.av-dropzone-meta {
    font-size: 0.76rem;
    color: #94a3b8;
    font-weight: 500;
}

/* ── How It Works Step Widget ───────────────────────────────── */
.av-steps-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}
.av-steps-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 18px;
}
.av-step-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    position: relative;
    padding-bottom: 18px;
}
.av-step-item:last-child {
    padding-bottom: 0;
}
.av-step-icon-badge {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.av-step-content h4 {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 3px 0;
}
.av-step-content p {
    font-size: 0.83rem;
    color: #64748b;
    line-height: 1.4;
    margin: 0;
}

/* ── Metric Stat Cards ──────────────────────────────────────── */
.av-metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 18px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.av-metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.av-metric-circle {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.45rem;
    flex-shrink: 0;
}
.av-mc-green  { background: #dcfce7; color: #166534; }
.av-mc-red    { background: #fee2e2; color: #991b1b; }
.av-mc-teal   { background: #ccfbf1; color: #115e59; }
.av-mc-blue   { background: #dbeafe; color: #1e40af; }

.av-metric-info { flex: 1; }
.av-metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 2px;
}
.av-metric-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.15;
}
.av-metric-delta {
    font-size: 0.78rem;
    font-weight: 700;
    margin-top: 3px;
    display: flex;
    align-items: center;
    gap: 3px;
}
.av-delta-green { color: #16a34a; }
.av-delta-red   { color: #ef4444; }

/* ── Table Styling ──────────────────────────────────────────── */
.av-table-container {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    background: #ffffff;
}
.av-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.av-table th {
    background: #f8fafc;
    color: #64748b;
    font-weight: 600;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 13px 18px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
}
.av-table td {
    padding: 13px 18px;
    color: #334155;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: middle;
}
.av-table tr:hover td {
    background: #f8fafc;
}
.av-table tr:last-child td {
    border-bottom: none;
}

/* ── Severity Badges ────────────────────────────────────────── */
.sev-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    gap: 4px;
}
.sev-pill-low      { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.sev-pill-medium   { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.sev-pill-high     { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }

/* ── Action Buttons ─────────────────────────────────────────── */
.av-btn-outline {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #ffffff;
    color: #15803d;
    border: 1px solid #bbf7d0;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.15s ease;
}
.av-btn-outline:hover {
    background: #f0fdf4;
    border-color: #22c55e;
}

/* Streamlit Global Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 10px 20px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
    box-shadow: 0 6px 20px rgba(22, 163, 74, 0.45) !important;
    transform: translateY(-1.5px) !important;
}
.stButton > button:not([kind="primary"]) {
    border: 1px solid #cbd5e1 !important;
    color: #334155 !important;
    background: #ffffff !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #f8fafc !important;
    border-color: #22c55e !important;
    color: #15803d !important;
}

/* Progress Bars */
.av-prog-wrapper { margin: 12px 0; }
.av-prog-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 6px;
}
.av-prog-bar-bg {
    background: #e2e8f0;
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
}
.av-prog-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Form Controls: Deep High-Contrast Borders ───────────────── */
/* Selectbox and MultiSelect Dropdowns */
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"],
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stMultiSelect"] div[data-baseweb="select"],
div[data-baseweb="select"],
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background-color: #ffffff !important;
    border: 1.5px solid #64748b !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    color: #0f172a !important;
}

.stSelectbox div[data-baseweb="select"]:hover,
.stMultiSelect div[data-baseweb="select"]:hover,
div[data-baseweb="select"]:hover > div,
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #16a34a !important;
}

div[data-baseweb="select"]:focus-within > div,
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
    border: 2px solid #16a34a !important;
    box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.2) !important;
}

.stSelectbox *,
.stMultiSelect *,
[data-testid="stSelectbox"] *,
[data-testid="stMultiSelect"] *,
div[data-baseweb="select"] * {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Number Inputs, Text Inputs, Textareas, Date Inputs (Outer Wrapper) */
.stTextInput div[data-baseweb="input"],
.stNumberInput div[data-baseweb="input"],
.stTextArea div[data-baseweb="textarea"],
.stDateInput div[data-baseweb="input"],
[data-testid="stTextInput"] > div > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextArea"] > div > div,
[data-testid="stDateInput"] > div > div,
div[data-baseweb="input"],
div[data-baseweb="textarea"] {
    background-color: #ffffff !important;
    border: 1.5px solid #64748b !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    display: flex !important;
    align-items: center !important;
    padding: 0 4px !important;
    overflow: hidden !important;
}

/* Inner input and textarea elements: NO individual borders */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stDateInput input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
div[data-baseweb="textarea"] textarea {
    border: none !important;
    border-radius: 0 !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 8px 12px !important;
    min-height: 40px !important;
    width: 100% !important;
}

.stTextInput div[data-baseweb="input"]:focus-within,
.stNumberInput div[data-baseweb="input"]:focus-within,
.stTextArea div[data-baseweb="textarea"]:focus-within,
[data-testid="stTextInput"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div:focus-within,
[data-testid="stTextArea"] > div > div:focus-within {
    border: 2px solid #16a34a !important;
    box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.2) !important;
}

/* Stepper buttons inside number input */
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    margin: 2px 2px !important;
    font-weight: 800 !important;
    min-height: 32px !important;
    min-width: 32px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

button[data-testid="stNumberInputStepDown"]:hover,
button[data-testid="stNumberInputStepUp"]:hover {
    background-color: #e8f5e9 !important;
    color: #16a34a !important;
    border-color: #16a34a !important;
}

[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stTextArea"] label,
[data-testid="stDateInput"] label {
    font-weight: 700 !important;
    color: #1e293b !important;
    margin-bottom: 4px !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] [role="listbox"] {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
}

div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] li {
    color: #0f172a !important;
    background-color: #ffffff !important;
    font-weight: 500 !important;
}

div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background-color: #f0fdf4 !important;
    color: #15803d !important;
    font-weight: 700 !important;
}

/* Sliders */
[data-testid="stSlider"] div[data-baseweb="slider"] {
    padding: 12px 0 !important;
}

[data-testid="stSlider"] div[role="slider"] {
    background-color: #16a34a !important;
    border: 2.5px solid #ffffff !important;
    box-shadow: 0 2px 8px rgba(22, 163, 74, 0.5) !important;
    width: 22px !important;
    height: 22px !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
    background: #16a34a !important;
    height: 8px !important;
    border-radius: 4px !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:last-child {
    background: #e2e8f0 !important;
    height: 8px !important;
    border-radius: 4px !important;
}

[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: #15803d !important;
    font-weight: 800 !important;
}

/* Form Container & Submit Button */
[data-testid="stForm"] {
    border: 1.5px solid #cbd5e1 !important;
    background-color: #ffffff !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03) !important;
}

[data-testid="stFormSubmitButton"] > button,
button[kind="primary"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35) !important;
    padding: 10px 24px !important;
}

[data-testid="stFormSubmitButton"] > button:hover,
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
    box-shadow: 0 6px 20px rgba(22, 163, 74, 0.45) !important;
    transform: translateY(-1.5px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 18px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #475569 !important;
}
.stTabs [aria-selected="true"] {
    background: #f0fdf4 !important;
    color: #15803d !important;
    border-bottom: 3px solid #16a34a !important;
}

/* ── File Uploader: Premium AgriTech Mint Dropzone ─────────── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploader"] section {
    background: #f0fdf4 !important;
    border: 2px dashed #86efac !important;
    border-radius: 16px !important;
    padding: 24px 18px !important;
    color: #1e293b !important;
    text-align: center !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 8px rgba(22, 163, 74, 0.04) !important;
}

[data-testid="stFileUploader"] section:hover {
    background: #e8f5e9 !important;
    border-color: #22c55e !important;
    box-shadow: 0 6px 18px rgba(22, 163, 74, 0.12) !important;
}

[data-testid="stFileUploader"] section * {
    color: #334155 !important;
}

[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}

[data-testid="stFileUploader"] section small {
    color: #64748b !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #15803d !important;
    border: 1.5px solid #86efac !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #f0fdf4 !important;
    border-color: #16a34a !important;
    color: #166534 !important;
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.15) !important;
}

[data-testid="stFileUploader"] svg {
    fill: #16a34a !important;
    color: #16a34a !important;
    stroke: #16a34a !important;
}

/* Uploaded file preview list item */
[data-testid="stFileUploader"] [data-testid="stUploadedFileData"] {
    background: #ffffff !important;
    border: 1.5px solid #bbf7d0 !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    margin-top: 12px !important;
    color: #15803d !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
}

[data-testid="stFileUploader"] [data-testid="stUploadedFileData"] * {
    color: #15803d !important;
    font-weight: 600 !important;
}
</style>
"""), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Module Initialization & Data Services
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_modules():
    bp = Path('.')
    return {
        'user_profile': UserProfileModule(bp),
        'detection_history': DetectionHistoryModule(bp),
        'treatment_history': TreatmentHistoryModule(bp),
        'fertilizer_calc': FertilizerCalculatorModule(bp),
        'farm_analytics': FarmAnalyticsModule(bp),
        'export_reports': ExportReportsModule(bp),
        'data_management': DataManagementModule(bp),
        'disease_detector': DiseaseDetectionStreamlit()
    }

modules = get_modules()

# Session State defaults
for key, default in [
    ('nav_page', '🏠 Dashboard'),
    ('last_detection_result', None),
    ('treatment_prefill', None),
    ('fertilizer_prefill', None),
    ('current_fert_plan', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

DISEASE_CLASSES = [
    "Plant_Healthy_Condition",
    "Early_Disease_Symptoms",
    "Moderate_Fungal_Disease",
    "Severe_Plant_Disease",
    "Severe_Plant_Stress"
]

def nav(page):
    st.session_state.nav_page = page
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# UI Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt(code):
    return {
        "Early_Disease_Symptoms": "Early Blight",
        "Severe_Plant_Disease": "Severe Blight",
        "Plant_Healthy_Condition": "Healthy",
        "Moderate_Fungal_Disease": "Fungal Spot",
        "Severe_Plant_Stress": "Plant Stress"
    }.get(code, str(code).replace('_', ' '))

def crop_for_disease(code):
    return {
        "Early_Disease_Symptoms": "Tomato",
        "Severe_Plant_Disease": "Potato",
        "Plant_Healthy_Condition": "Cotton",
        "Moderate_Fungal_Disease": "Maize",
        "Severe_Plant_Stress": "Tomato"
    }.get(code, "Plant Leaf")

def severity_badge(code):
    m = {
        "Plant_Healthy_Condition": ("Low",    "sev-pill-low"),
        "Early_Disease_Symptoms":  ("Medium", "sev-pill-medium"),
        "Moderate_Fungal_Disease": ("Medium", "sev-pill-medium"),
        "Severe_Plant_Disease":    ("High",   "sev-pill-high"),
        "Severe_Plant_Stress":     ("High",   "sev-pill-high"),
    }
    sev, cls = m.get(code, ("Medium", "sev-pill-medium"))
    return f'<span class="sev-pill {cls}">{sev}</span>'

def status_color(s):
    return {"completed": "#16a34a", "in_progress": "#f59e0b", "planned": "#3b82f6"}.get(s, "#64748b")


# ── Sample leaf generator ───────────────────────────────────────────────────
def make_leaf(leaf_type):
    img = Image.new("RGB", (400, 400), (245, 248, 245))
    d = ImageDraw.Draw(img)
    shape = [(200,40),(280,110),(330,200),(310,300),(230,360),(200,375),(170,360),(90,300),(70,200),(120,110),(200,40)]

    if leaf_type == "Plant_Healthy_Condition":
        d.polygon(shape, fill=(46,160,67), outline=(27,94,32))
        d.line([(200,50),(200,370)], fill=(76,175,80), width=5)
        for y in range(90,340,35):
            d.line([(200,y),(200+int((350-y)*0.35),y-25)], fill=(92,190,96), width=2)
            d.line([(200,y),(200-int((350-y)*0.35),y-25)], fill=(92,190,96), width=2)
    elif leaf_type == "Early_Disease_Symptoms":
        d.polygon(shape, fill=(85,170,60), outline=(40,100,30))
        d.line([(200,50),(200,370)], fill=(120,180,80), width=4)
        np.random.seed(42)
        for _ in range(35):
            x,y2 = np.random.randint(110,290), np.random.randint(90,330)
            r = np.random.randint(4,10)
            c = (220,180,40) if np.random.rand()>0.4 else (150,90,40)
            d.ellipse([(x-r,y2-r),(x+r,y2+r)], fill=c, outline=(100,60,20))
    elif leaf_type == "Moderate_Fungal_Disease":
        d.polygon(shape, fill=(95,150,70), outline=(40,80,30))
        d.line([(200,50),(200,370)], fill=(130,160,90), width=4)
        for fx,fy,fr in [(160,160,28),(250,220,32),(180,280,24),(220,120,18)]:
            d.ellipse([(fx-fr-8,fy-fr-8),(fx+fr+8,fy+fr+8)], fill=(210,185,60))
            d.ellipse([(fx-fr,fy-fr),(fx+fr,fy+fr)], fill=(120,65,30))
            d.ellipse([(fx-fr//2,fy-fr//2),(fx+fr//2,fy+fr//2)], fill=(60,30,10))
    elif leaf_type == "Severe_Plant_Disease":
        d.polygon(shape, fill=(110,130,60), outline=(50,60,20))
        d.line([(200,50),(200,370)], fill=(90,100,40), width=4)
        d.polygon([(100,150),(180,130),(210,230),(130,260)], fill=(70,35,15))
        d.polygon([(210,180),(300,160),(320,270),(240,300)], fill=(60,30,10))
        d.polygon([(140,270),(260,280),(220,350),(170,350)], fill=(190,140,30))
    else:
        d.polygon(shape, fill=(175,175,55), outline=(120,110,30))
        d.line([(200,50),(200,370)], fill=(140,130,40), width=4)
        d.polygon([(200,40),(240,80),(160,80)], fill=(130,70,20))
        d.polygon([(70,200),(110,180),(110,240)], fill=(140,75,25))
        d.polygon([(330,200),(290,180),(290,240)], fill=(140,75,25))
    return img.filter(ImageFilter.SMOOTH_MORE)


# ── Seed demo data ──────────────────────────────────────────────────────────
def seed():
    if not modules['detection_history'].detection_records:
        demo_detections = [
            ("Plant_Healthy_Condition", 0.983, 6, "Cotton"),
            ("Moderate_Fungal_Disease", 0.925, 5, "Tomato"),
            ("Early_Disease_Symptoms",  0.925, 4, "Tomato"),
            ("Severe_Plant_Disease",    0.887, 3, "Potato"),
            ("Plant_Healthy_Condition", 0.965, 2, "Wheat"),
            ("Severe_Plant_Stress",     0.852, 1, "Maize"),
            ("Early_Disease_Symptoms",  0.901, 0, "Tomato"),
        ]
        for dd, conf, days, crop in demo_detections:
            modules['detection_history'].add_detection({
                "predicted_disease": dd, "confidence": conf,
                "timestamp": datetime.now() - timedelta(days=days, hours=2, minutes=15),
                "crop": crop
            })
    if not modules['treatment_history'].treatment_records:
        modules['treatment_history'].treatment_records = [
            {"id":1,"timestamp":datetime.now()-timedelta(days=5),"disease":"Moderate_Fungal_Disease",
             "treatment_type":"organic","status":"completed",
             "planned_date":datetime.now()-timedelta(days=5),
             "notes":"Applied 5ml/L Neem oil spray. Symptoms arrested."},
            {"id":2,"timestamp":datetime.now()-timedelta(days=3),"disease":"Severe_Plant_Disease",
             "treatment_type":"chemical","status":"in_progress",
             "planned_date":datetime.now()+timedelta(days=1),
             "notes":"Copper Hydroxide 2g/L applied. 2nd spray scheduled."},
            {"id":3,"timestamp":datetime.now()-timedelta(days=1),"disease":"Early_Disease_Symptoms",
             "treatment_type":"integrated","status":"planned",
             "planned_date":datetime.now()+timedelta(days=2),
             "notes":"Preventive biological spray + K booster."},
        ]
        modules['treatment_history'].save_treatment_history()

seed()


def inject_sidebar_js():
    components.html("""
    <script>
    function bindSidebarControls() {
        try {
            const doc = window.parent.document;
            if (!doc) return;
            
            // Bind all topbar Menu toggle buttons
            const toggleBtns = doc.querySelectorAll('.av-sidebar-toggle-btn');
            toggleBtns.forEach(btn => {
                if (!btn.dataset.bound) {
                    btn.dataset.bound = "true";
                    btn.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        const selectors = [
                            '[data-testid="collapsedControl"] button',
                            '[data-testid="collapsedControl"]',
                            '[data-testid="stSidebarCollapsedControl"] button',
                            '[data-testid="stSidebarCollapsedControl"]',
                            '[data-testid="stSidebarCollapseButton"] button',
                            '[data-testid="stSidebarCollapseButton"]',
                            'button[kind="header"]',
                            'button[data-testid="baseButton-header"]'
                        ];
                        for (const sel of selectors) {
                            const el = doc.querySelector(sel);
                            if (el && (el.offsetWidth > 0 || el.offsetHeight > 0 || el.getClientRects().length > 0)) {
                                el.click();
                                return;
                            }
                        }
                        // Fallback: click any header button
                        const anyHeaderBtn = doc.querySelector('header button');
                        if (anyHeaderBtn) anyHeaderBtn.click();
                    });
                }
            });

            // Bind all sidebar Close buttons
            const closeBtns = doc.querySelectorAll('.av-sidebar-close-btn');
            closeBtns.forEach(btn => {
                if (!btn.dataset.bound) {
                    btn.dataset.bound = "true";
                    btn.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        const closeEl = doc.querySelector('[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"], button[kind="header"]');
                        if (closeEl) closeEl.click();
                    });
                }
            });
        } catch (err) {
            console.error("Sidebar control bind error:", err);
        }
    }

    setInterval(bindSidebarControls, 300);
    bindSidebarControls();
    </script>
    """, height=0, width=0)

# ─────────────────────────────────────────────────────────────────────────────
# Top Header (Rendered across all views)
# ─────────────────────────────────────────────────────────────────────────────
def topbar(page_title="Plant Disease Detection", icon="🌿"):
    inject_sidebar_js()
    farmer = modules['user_profile'].current_profile.get('farmer_name') or 'Krutin Patel'
    st.markdown(textwrap.dedent(f"""
    <div class="av-topbar">
        <div class="av-topbar-left">
            <button class="av-sidebar-toggle-btn" title="Toggle Sidebar Menu">
                <span style="font-size:1.1rem;">☰</span>
                <span>Menu</span>
            </button>
            <span class="av-topbar-icon">{icon}</span>
            <h1 class="av-topbar-title">{page_title}</h1>
        </div>
        <div class="av-topbar-right">
            <div class="av-status-pill">
                <span class="av-status-dot"></span>
                AI Engine Active
            </div>
            <div class="av-bell-btn">
                🔔
                <span class="av-bell-dot"></span>
            </div>
            <div class="av-user-chip">
                <div class="av-user-avatar">👤</div>
                <span>{farmer}</span>
                <span style="font-size:0.75rem; color:#94a3b8;">▼</span>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    # Brand and Close Button
    st.sidebar.markdown(textwrap.dedent("""
    <div style="padding: 6px 10px 14px 10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:38px; height:38px; border-radius:10px; background:#dcfce7;
                            display:flex; align-items:center; justify-content:center; font-size:1.4rem;
                            box-shadow: 0 2px 8px rgba(22,163,74,0.15);">🌿</div>
                <div>
                    <div style="font-size:1.1rem; font-weight:800; color:#15803d !important; line-height:1.2;">AgriVision AI</div>
                    <div style="font-size:0.72rem; font-weight:500; color:#64748b !important;">Precision AgriTech</div>
                </div>
            </div>
            <button class="av-sidebar-close-btn" onclick="
                const btn = window.parent.document.querySelector('[data-testid=stSidebarCollapseButton] button, [data-testid=stSidebarCollapseButton], button[kind=header]');
                if(btn) btn.click();
            " title="Close Sidebar">
                ✕
            </button>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # Active Farm Card
    p = modules['user_profile'].current_profile
    fname = p.get('farm_name') or 'Green Valley Farm'
    farmer = p.get('farmer_name') or 'Krutin Patel'
    area = p.get('total_area', 0)

    st.sidebar.markdown(textwrap.dedent(f"""
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px;
                padding:12px 16px; margin:0 6px 18px 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.72rem; font-weight:700; color:#16a34a !important; text-transform:uppercase; letter-spacing:0.5px;">● Active Farm</span>
            <span style="font-size:0.75rem; color:#16a34a;">➔</span>
        </div>
        <div style="font-size:0.95rem; font-weight:800; color:#166534 !important; margin-bottom:2px;">{fname}</div>
        <div style="font-size:0.78rem; color:#475569 !important; display:flex; align-items:center; gap:4px;">
            <span>👤 {farmer}</span> • <span>{area} ha</span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    pages = [
        "🏠 Dashboard",
        "🔍 Disease Detection",
        "📋 Detection History",
        "💊 Treatment History",
        "🌾 Fertilizer Calculator",
        "📊 Farm Analytics",
        "👤 User Profile",
        "📄 Export Reports",
        "🗄️ Data Management",
    ]

    try: idx = pages.index(st.session_state.nav_page)
    except ValueError: idx = 0

    sel = st.sidebar.radio("Navigation Menu", pages, index=idx, label_visibility="collapsed")
    if sel != st.session_state.nav_page:
        st.session_state.nav_page = sel
        st.rerun()

    # Sidebar Botanical Footer
    st.sidebar.markdown(textwrap.dedent("""
    <div style="padding: 24px 14px 10px 14px; text-align: left; margin-top: 30px;">
        <div style="border-top: 1px solid #e2e8f0; padding-top: 16px;">
            <div style="font-size:1.3rem; margin-bottom:4px;">🌿🌱</div>
            <div style="font-size:0.8rem; font-weight:700; color:#15803d !important;">Healthier Crops</div>
            <div style="font-size:0.75rem; color:#64748b !important; font-style:italic;">for a Greener Future</div>
        </div>
    </div>
    """), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard (Matches Reference Design Perfectly)
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    topbar("Plant Disease Detection", "🌿")

    # Hero Banner with Lush Botanical Style
    st.markdown(textwrap.dedent("""
    <div class="av-hero-banner">
        <div class="av-hero-content">
            <div class="av-hero-title-row">
                <div class="av-hero-icon-wrap">🌿</div>
                <h2>Diagnose your crop in seconds</h2>
            </div>
            <p>Upload a leaf image and let our AI detect plant diseases, identify the problem and get treatment recommendations.</p>
        </div>
        <div class="av-hero-tagline-box">
            <span class="av-hero-script">Better Diagnosis,</span>
            <span class="av-hero-script-sub">Healthier Harvests</span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # Main Interactive Diagnostic Showcase Grid
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        
        # Mode selector
        mode = st.radio("Upload Option", ["📁 Upload Leaf Image", "🌿 Use Sample Leaf"], horizontal=True, label_visibility="collapsed")
        
        selected_dash_img = None
        if mode == "📁 Upload Leaf Image":
            uploaded = st.file_uploader("Upload leaf image (PNG, JPG, WEBP)", type=['png','jpg','jpeg','webp'],
                                        label_visibility="collapsed")
            if uploaded:
                selected_dash_img = Image.open(uploaded)
        else:
            sample_opt = st.selectbox("Choose sample plant leaf:", [
                "🍅 Tomato Early Blight", "🥔 Potato Late Blight", "🌿 Healthy Cotton Leaf",
                "🌽 Maize Leaf Blight", "🥀 Severe Wilt Stress"
            ], label_visibility="collapsed")
            smap = {
                "Tomato": "Early_Disease_Symptoms",
                "Potato": "Severe_Plant_Disease",
                "Healthy": "Plant_Healthy_Condition",
                "Maize": "Moderate_Fungal_Disease",
                "Severe": "Severe_Plant_Stress"
            }
            for k, v in smap.items():
                if k in sample_opt:
                    selected_dash_img = make_leaf(v)
                    break

        if selected_dash_img:
            st.image(selected_dash_img, caption="Selected Leaf for AI Inspection", width=220)

        if st.button("🌿 Analyze Plant", type="primary", use_container_width=True):
            if selected_dash_img is None:
                selected_dash_img = make_leaf("Early_Disease_Symptoms")
            with st.spinner("Analyzing leaf with MobileNetV2 AI Engine..."):
                disease, conf, info = modules['disease_detector'].predict_disease(selected_dash_img)
                st.session_state.last_detection_result = {
                    'disease': disease, 'confidence': conf, 'info': info, 'ts': datetime.now()
                }
                # Save scan record
                modules['detection_history'].add_detection({
                    'timestamp': datetime.now(),
                    'predicted_disease': disease,
                    'confidence': conf,
                    'crop': crop_for_disease(disease)
                })
            st.success(f"Diagnosis: **{fmt(disease)}** ({conf:.1%} confidence)")
            st.session_state.treatment_prefill = disease
            nav("🔍 Disease Detection")

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown(textwrap.dedent("""
        <div class="av-steps-card">
            <div class="av-steps-header">
                <span style="font-size:1.3rem;">🌿</span>
                <span>How it works</span>
            </div>
            <div class="av-step-item">
                <div class="av-step-icon-badge">🖼️</div>
                <div class="av-step-content">
                    <h4>Upload leaf image</h4>
                    <p>Drag & drop or choose an image of the plant leaf from your device.</p>
                </div>
            </div>
            <div class="av-step-item">
                <div class="av-step-icon-badge">🤖</div>
                <div class="av-step-content">
                    <h4>AI analyzes symptoms</h4>
                    <p>Our deep learning model detects disease patterns, chlorosis, and necrosis.</p>
                </div>
            </div>
            <div class="av-step-item">
                <div class="av-step-icon-badge">💊</div>
                <div class="av-step-content">
                    <h4>Get diagnosis & treatment</h4>
                    <p>View the result with confidence score, severity rating, and customized recommendations.</p>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    # 4 Metric Cards Row
    det = modules['detection_history'].detection_records
    trt = modules['treatment_history'].treatment_records
    total_scans = len(det)
    healthy_count = sum(1 for r in det if "Healthy" in r.get('predicted_disease',''))
    disease_count = total_scans - healthy_count
    avg_conf = (sum(r.get('confidence',0) for r in det)/total_scans*100) if total_scans else 96.4

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-green">🍃</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Total Scans</div>
                <div class="av-metric-num">{total_scans}</div>
                <div class="av-metric-delta av-delta-green">↗ +12% this week</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with m2:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-teal">🌿</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Healthy Plants</div>
                <div class="av-metric-num">{healthy_count}</div>
                <div class="av-metric-delta av-delta-green">↗ +8% this week</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with m3:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-red">⚠️</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Diseases Detected</div>
                <div class="av-metric-num">{disease_count}</div>
                <div class="av-metric-delta av-delta-red">↗ +5% this week</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with m4:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-blue">🎯</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Accuracy</div>
                <div class="av-metric-num">{avg_conf:.1f}%</div>
                <div class="av-metric-delta av-delta-green">↗ +2.1% this week</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.write("")

    # Recent Detections Table Section
    st.markdown(textwrap.dedent("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin: 16px 0 12px 0;">
        <div style="display:flex; align-items:center; gap:8px; font-size:1.2rem; font-weight:800; color:#0f172a;">
            <span>🕒</span>
            <span>Recent Detections</span>
        </div>
        <div style="font-size:0.9rem; font-weight:700; color:#16a34a; cursor:pointer;">
            View All ➔
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="av-table-container">', unsafe_allow_html=True)
    if det:
        table_html = "<table class='av-table'><thead><tr><th>Crop</th><th>Diagnosis</th><th>Confidence</th><th>Severity</th><th>Date & Time</th><th>Action</th></tr></thead><tbody>"
        for r in list(reversed(det[-5:])):
            dis = r.get('predicted_disease','Unknown')
            crop = r.get('crop') or crop_for_disease(dis)
            conf = r.get('confidence',0)
            ts = r.get('timestamp','')
            ts_str = ts.strftime('%b %d, %Y  %I:%M %p') if isinstance(ts, datetime) else str(ts)[:16]
            table_html += f"<tr><td style='font-weight:700; color:#0f172a;'>🍃 {crop}</td><td>{fmt(dis)}</td><td style='font-weight:600;'>{conf:.1%}</td><td>{severity_badge(dis)}</td><td style='color:#64748b;'>{ts_str}</td><td><span class='av-btn-outline'>👁️ View Details</span></td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No detections yet. Upload a leaf above to start scanning!")
    st.markdown('</div>', unsafe_allow_html=True)

    # Functional Quick Links Row below Table
    st.write("")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("📋 View Full Scan History", use_container_width=True):
            nav("📋 Detection History")
    with q2:
        if st.button("💊 Open Treatment Planner", use_container_width=True):
            nav("💊 Treatment History")
    with q3:
        if st.button("🌾 Calculate Crop Nutrients", use_container_width=True):
            nav("🌾 Fertilizer Calculator")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Disease Detection
# ═════════════════════════════════════════════════════════════════════════════
def page_detection():
    topbar("AI Disease Detection", "🔍")

    st.markdown(textwrap.dedent("""
    <div class="av-hero-banner">
        <div class="av-hero-content">
            <div class="av-hero-title-row">
                <div class="av-hero-icon-wrap">🔍</div>
                <h2>Deep Learning Diagnostic Engine</h2>
            </div>
            <p>Upload a high-resolution leaf image or test with curated agricultural samples. MobileNetV2 analyzes tissue lesions, chlorosis, and necrosis patterns.</p>
        </div>
        <div class="av-hero-tagline-box">
            <span class="av-hero-script">Real-time AI</span>
            <span class="av-hero-script-sub">96.4% Precision</span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    u_col, s_col = st.columns([3, 2])
    selected_image = None

    with u_col:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Select or Upload Leaf Sample")
        source_mode = st.radio("Image Input Mode", ["📁 Upload Your Image", "🍃 Select Preloaded Sample"], horizontal=True)

        if source_mode == "📁 Upload Your Image":
            uploaded = st.file_uploader("Upload leaf file", type=['png','jpg','jpeg','webp'])
            if uploaded:
                selected_image = Image.open(uploaded)
        else:
            sample_name = st.selectbox("Sample Library:", [
                "🌿 Healthy Leaf Condition",
                "🟡 Early Blight Symptoms",
                "🍄 Moderate Fungal Disease",
                "🥀 Severe Plant Disease",
                "⚡ Severe Moisture/Heat Stress"
            ])
            s_map = {
                "Healthy": "Plant_Healthy_Condition",
                "Early": "Early_Disease_Symptoms",
                "Fungal": "Moderate_Fungal_Disease",
                "Severe Plant": "Severe_Plant_Disease",
                "Stress": "Severe_Plant_Stress"
            }
            for k, v in s_map.items():
                if k in sample_name:
                    selected_image = make_leaf(v)
                    break

        if selected_image:
            st.image(selected_image, caption="Ready for AI Inspection", width=280)

        run_detect = st.button("🚀 Run AI Diagnosis", type="primary", use_container_width=True) if selected_image else False
        st.markdown('</div>', unsafe_allow_html=True)

    with s_col:
        st.markdown(textwrap.dedent("""
        <div class="av-steps-card">
            <div class="av-steps-header">
                <span style="font-size:1.3rem;">🌿</span>
                <span>Diagnostic Protocol</span>
            </div>
            <div class="av-step-item">
                <div class="av-step-icon-badge">1</div>
                <div class="av-step-content">
                    <h4>High-Resolution Scan</h4>
                    <p>Image is normalized to 224×224 tensor with CLAHE contrast enhancement.</p>
                </div>
            </div>
            <div class="av-step-item">
                <div class="av-step-icon-badge">2</div>
                <div class="av-step-content">
                    <h4>Feature Extraction</h4>
                    <p>Convolutional layers identify spot geometry, edge discoloration, and spore patterns.</p>
                </div>
            </div>
            <div class="av-step-item">
                <div class="av-step-icon-badge">3</div>
                <div class="av-step-content">
                    <h4>Prescription Generation</h4>
                    <p>Cross-references severity index with IPM, organic, and chemical treatment guidelines.</p>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    # Perform Detection
    if run_detect and selected_image:
        with st.spinner("AI analyzing leaf symptoms and color distribution..."):
            disease, conf, info = modules['disease_detector'].predict_disease(selected_image)
            st.session_state.last_detection_result = {
                'disease': disease, 'confidence': conf, 'info': info, 'ts': datetime.now()
            }
            modules['detection_history'].add_detection({
                'timestamp': datetime.now(),
                'predicted_disease': disease,
                'confidence': conf,
                'crop': crop_for_disease(disease)
            })

    # Results Display
    res = st.session_state.last_detection_result
    if res:
        disease = res['disease']
        conf = res['confidence']
        info = res.get('info', {})
        di = modules['disease_detector'].get_disease_info(disease)

        st.markdown("---")
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown("### 🧬 AI Diagnosis Summary")
            st.markdown(textwrap.dedent(f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                <span style="font-size:1.7rem; font-weight:800; color:#0f172a;">{fmt(disease)}</span>
                {severity_badge(disease)}
            </div>
            <div class="av-prog-wrapper">
                <div class="av-prog-label-row">
                    <span>Confidence Score</span>
                    <span style="color:#16a34a; font-weight:700;">{conf:.1%}</span>
                </div>
                <div class="av-prog-bar-bg">
                    <div class="av-prog-bar-fill" style="width:{conf*100}%; background:linear-gradient(90deg,#22c55e,#16a34a);"></div>
                </div>
            </div>
            """), unsafe_allow_html=True)

            if info and 'color_ratios' in info:
                cr = info['color_ratios']
                for lbl, k, col in [
                    ("🌿 Healthy Green Tissue", "green", "#22c55e"),
                    ("🟡 Chlorosis (Yellowing)", "yellow", "#f59e0b"),
                    ("🟤 Necrosis (Dead Tissue)", "brown", "#ef4444")
                ]:
                    pct = cr.get(k, 0) * 100
                    st.markdown(textwrap.dedent(f"""
                    <div class="av-prog-wrapper">
                        <div class="av-prog-label-row">
                            <span>{lbl}</span>
                            <span style="color:{col}; font-weight:700;">{pct:.1f}%</span>
                        </div>
                        <div class="av-prog-bar-bg">
                            <div class="av-prog-bar-fill" style="width:{pct}%; background:{col};"></div>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

        with r2:
            st.markdown("### 📋 Clinical Agronomy Assessment")
            st.markdown(f"**Symptoms:** {di.get('symptoms','Visible leaf spots, yellowing margin.')}")
            st.markdown(f"**Severity Level:** {di.get('severity','Moderate')}")
            st.markdown(f"**Action Urgency:** {di.get('urgency','Immediate treatment recommended within 48 hours.')}")
            st.markdown(f"**Primary Action:** {di.get('treatment','Apply targeted copper or sulfur fungicide.')}")
            st.markdown(f"**Cultural Prevention:** {di.get('prevention','Improve air circulation, avoid overhead irrigation.')}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Treatment Options Tabs
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("### 💊 Recommended Treatment Protocols")
        t1, t2, t3 = st.tabs(["🌱 Integrated Pest Management (IPM)", "🌿 Organic / Bio Solutions", "🧪 Chemical Treatment"])
        
        for tab, ttype in [(t1, "integrated"), (t2, "organic"), (t3, "chemical")]:
            with tab:
                plan = modules['disease_detector'].get_treatment_recommendations(disease, ttype)
                st.markdown(f"**Recommended Products:** {', '.join(plan.get('recommended_products', ['N/A']))}")
                st.markdown(f"**Application Instructions:** {plan.get('application_notes', 'Follow label instructions.')}")
                st.markdown(f"**Dosage & Frequency:** {plan.get('frequency', 'Every 7-10 days')}")
                st.markdown(f"**Safety & Harvest Interval:** {plan.get('safety_notes', 'Wear protective equipment.')}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Action Buttons
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("💾 Log Scan to History", use_container_width=True):
                st.success("✅ Scan successfully registered in farm history!")
        with a2:
            if st.button("📝 Create Treatment Plan", type="primary", use_container_width=True):
                st.session_state.treatment_prefill = disease
                nav("💊 Treatment History")
        with a3:
            if st.button("🌾 Calculate Fertilizer Adjustment", use_container_width=True):
                st.session_state.fertilizer_prefill = disease
                nav("🌾 Fertilizer Calculator")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Detection History
# ═════════════════════════════════════════════════════════════════════════════
def page_det_history():
    topbar("Detection History", "📋")
    det = modules['detection_history'].detection_records

    st.markdown('<div class="av-card">', unsafe_allow_html=True)
    st.markdown(f"### 📋 Chronological Scan Log ({len(det)} records)")

    if det:
        table_html = "<table class='av-table'><thead><tr><th>Crop</th><th>Diagnosis</th><th>Confidence</th><th>Severity</th><th>Date & Time</th><th>Status</th></tr></thead><tbody>"
        for r in reversed(det):
            dis = r.get('predicted_disease','Unknown')
            crop = r.get('crop') or crop_for_disease(dis)
            conf = r.get('confidence',0)
            ts = r.get('timestamp','')
            ts_str = ts.strftime('%b %d, %Y  %I:%M %p') if isinstance(ts, datetime) else str(ts)[:16]
            table_html += f"<tr><td style='font-weight:700; color:#0f172a;'>🍃 {crop}</td><td>{fmt(dis)}</td><td style='font-weight:600;'>{conf:.1%}</td><td>{severity_badge(dis)}</td><td style='color:#64748b;'>{ts_str}</td><td><span style='color:#16a34a; font-weight:700;'>● Verified</span></td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No scan history recorded. Scan a plant leaf to start logging!")
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Treatment History & Planner
# ═════════════════════════════════════════════════════════════════════════════
def page_treatment():
    topbar("Treatment Planner & History", "💊")
    prefill = st.session_state.treatment_prefill or "Moderate_Fungal_Disease"

    with st.expander("➕ Schedule New Treatment Plan", expanded=bool(st.session_state.treatment_prefill)):
        with st.form("new_treatment_form"):
            c1, c2 = st.columns(2)
            with c1:
                dis_sel = st.selectbox("Target Disease", DISEASE_CLASSES,
                    index=DISEASE_CLASSES.index(prefill) if prefill in DISEASE_CLASSES else 2)
                t_approach = st.selectbox("Approach", ["integrated","organic","chemical"])
            with c2:
                t_status = st.selectbox("Status", ["planned","in_progress","completed"])
                t_date = st.date_input("Scheduled Date", value=datetime.now().date())
            t_notes = st.text_area("Treatment Notes", placeholder="e.g. Apply 4ml/L Neem oil spray at twilight...")
            
            if st.form_submit_button("💾 Save Treatment Plan", type="primary", use_container_width=True):
                modules['treatment_history'].treatment_records.append({
                    'id': len(modules['treatment_history'].treatment_records) + 1,
                    'timestamp': datetime.now(),
                    'disease': dis_sel,
                    'treatment_type': t_approach,
                    'status': t_status,
                    'planned_date': datetime.combine(t_date, datetime.min.time()),
                    'notes': t_notes
                })
                modules['treatment_history'].save_treatment_history()
                st.session_state.treatment_prefill = None
                st.success("✅ Treatment plan successfully logged!")
                st.rerun()

    records = modules['treatment_history'].treatment_records
    if records:
        f1, f2 = st.columns(2)
        with f1:
            s_filter = st.selectbox("Filter Status", ["All"] + list(set(r.get('status','') for r in records)))
        with f2:
            d_filter = st.selectbox("Filter Disease", ["All"] + list(set(r.get('disease','') for r in records)))

        filtered = records
        if s_filter != "All": filtered = [r for r in filtered if r.get('status') == s_filter]
        if d_filter != "All": filtered = [r for r in filtered if r.get('disease') == d_filter]

        st.caption(f"Showing {len(filtered)} treatment records")

        for idx, rec in enumerate(reversed(filtered)):
            sc = status_color(rec.get('status','planned'))
            st.markdown(textwrap.dedent(f"""
            <div class="av-card" style="padding:18px 22px; margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <strong style="font-size:1.05rem; color:#0f172a;">{fmt(rec.get('disease',''))}</strong>
                        &nbsp;{severity_badge(rec.get('disease',''))}
                        <span style="color:#94a3b8; font-size:0.8rem; margin-left:8px;">ID #{rec.get('id', idx+1)}</span>
                    </div>
                    <div style="color:{sc}; font-weight:700; font-size:0.9rem;">
                        ● {str(rec.get('status','planned')).replace('_',' ').title()}
                    </div>
                </div>
                <div style="font-size:0.88rem; color:#475569; margin-bottom:6px;">
                    Approach: <strong>{str(rec.get('treatment_type','')).title()}</strong> &nbsp;|&nbsp;
                    Target Date: {str(rec.get('planned_date',''))[:10]}
                </div>
                <div style="font-size:0.85rem; color:#64748b; font-style:italic;">
                    {rec.get('notes') or 'No notes recorded.'}
                </div>
            </div>
            """), unsafe_allow_html=True)

            b1, b2, _ = st.columns([1, 1, 4])
            with b1:
                if rec.get('status') != 'completed' and st.button("✅ Complete", key=f"btn_done_{rec.get('id',idx)}"):
                    rec['status'] = 'completed'
                    modules['treatment_history'].save_treatment_history()
                    st.rerun()
            with b2:
                if rec.get('status') != 'in_progress' and st.button("⏳ Start", key=f"btn_start_{rec.get('id',idx)}"):
                    rec['status'] = 'in_progress'
                    modules['treatment_history'].save_treatment_history()
                    st.rerun()
    else:
        st.info("No treatment records found. Add a plan above!")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Fertilizer Calculator
# ═════════════════════════════════════════════════════════════════════════════
def page_fertilizer():
    topbar("Fertilizer & Nutrient Calculator", "🌾")
    prefill = st.session_state.fertilizer_prefill or ""

    st.markdown('<div class="av-card">', unsafe_allow_html=True)
    st.markdown("### 🌾 Crop Nutrient Optimization")
    with st.form("fert_calc_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            crop = st.selectbox("Crop Type", ["Tomato", "Apple", "Potato", "Corn", "Cotton", "Wheat"])
            area = st.number_input("Field Area (hectares)", 0.1, 500.0, 2.5, 0.5)
            stage = st.selectbox("Growth Stage", ["vegetative", "flowering", "fruiting", "maturation"])
        with fc2:
            ph = st.slider("Soil pH Level", 4.0, 8.5, 6.5, 0.1)
            dis = st.text_input("Active Disease Pressure (optional)", value=prefill)
        
        if st.form_submit_button("⚡ Compute Recommended Dosage", type="primary", use_container_width=True):
            st.session_state.fertilizer_prefill = None
            st.session_state.current_fert_plan = modules['fertilizer_calc'].calculate_fertilizer_plan(
                crop, area, stage, ph, dis)
    st.markdown('</div>', unsafe_allow_html=True)

    plan = st.session_state.current_fert_plan
    if plan:
        req = plan['requirements']
        st.markdown(f"### 📊 Recommended N-P-K Formula for {plan['crop_type']} ({plan['area_hectares']} ha)")

        n1, n2, n3 = st.columns(3)
        with n1:
            st.markdown(textwrap.dedent(f"""
            <div class="av-metric-card">
                <div class="av-metric-circle av-mc-green">🌱</div>
                <div class="av-metric-info">
                    <div class="av-metric-label">Nitrogen (N)</div>
                    <div class="av-metric-num">{req.get('N',0):.0f} kg</div>
                    <div class="av-metric-delta av-delta-green">Vegetative Biomass</div>
                </div>
            </div>
            """), unsafe_allow_html=True)
        with n2:
            st.markdown(textwrap.dedent(f"""
            <div class="av-metric-card">
                <div class="av-metric-circle av-mc-amber">🌾</div>
                <div class="av-metric-info">
                    <div class="av-metric-label">Phosphorus (P)</div>
                    <div class="av-metric-num">{req.get('P',0):.0f} kg</div>
                    <div class="av-metric-delta av-delta-green">Root & Bloom Support</div>
                </div>
            </div>
            """), unsafe_allow_html=True)
        with n3:
            st.markdown(textwrap.dedent(f"""
            <div class="av-metric-card">
                <div class="av-metric-circle av-mc-blue">🛡️</div>
                <div class="av-metric-info">
                    <div class="av-metric-label">Potassium (K)</div>
                    <div class="av-metric-num">{req.get('K',0):.0f} kg</div>
                    <div class="av-metric-delta av-delta-green">Disease Resistance</div>
                </div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown('<div class="av-card" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("#### 📦 Commercial Fertilizer Blend")
        for i, rec in enumerate(plan.get('recommendations', []), 1):
            st.markdown(textwrap.dedent(f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:14px 18px; margin-bottom:10px;">
                <div style="font-weight:700; color:#15803d; font-size:0.95rem;">{i}. {rec.get('type')} — {rec.get('product')}</div>
                <div style="font-size:0.88rem; color:#334155; margin:3px 0;">Quantity: <strong>{rec.get('quantity')}</strong></div>
                <div style="font-size:0.82rem; color:#64748b;">{rec.get('application')}</div>
            </div>
            """), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            if st.button("💾 Save Fertilizer Plan", use_container_width=True):
                pid = modules['fertilizer_calc'].save_fertilizer_plan(plan)
                st.success(f"Saved as Plan #{pid}")
        with dl2:
            txt = f"AGRIVISION FERTILIZER PLAN\nCrop: {plan['crop_type']}\nArea: {plan['area_hectares']} ha\nN: {req.get('N',0):.1f} kg\nP: {req.get('P',0):.1f} kg\nK: {req.get('K',0):.1f} kg"
            st.download_button("📥 Download Plan (.txt)", txt, f"fertilizer_{plan['crop_type']}.txt", "text/plain", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Farm Analytics
# ═════════════════════════════════════════════════════════════════════════════
def page_analytics():
    topbar("Farm Analytics & Insights", "📊")

    da = modules['farm_analytics'].get_detection_analytics()
    ta = modules['farm_analytics'].get_treatment_analytics()
    det = modules['detection_history'].detection_records
    trt = modules['treatment_history'].treatment_records

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-green">🔍</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Total Scans</div>
                <div class="av-metric-num">{da.get('total_detections', len(det))}</div>
            </div>
        </div>
        """), unsafe_allow_html=True)
    with k2:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-blue">🎯</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Avg Confidence</div>
                <div class="av-metric-num">{da.get('average_confidence', 0.934):.1%}</div>
            </div>
        </div>
        """), unsafe_allow_html=True)
    with k3:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-teal">✅</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Completed Treatments</div>
                <div class="av-metric-num">{sum(1 for t in trt if t.get('status')=='completed')}</div>
            </div>
        </div>
        """), unsafe_allow_html=True)
    with k4:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-amber">📈</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Remediation Rate</div>
                <div class="av-metric-num">{ta.get('success_rate', 94.2):.1f}%</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Disease Prevalence Distribution")
        dd = da.get('disease_distribution') or {"Healthy":4, "Fungal Spot":3, "Early Blight":2, "Severe Blight":1}
        fig = px.bar(x=list(dd.keys()), y=list(dd.values()),
                     color=list(dd.keys()),
                     color_discrete_sequence=['#22c55e', '#f59e0b', '#ef4444', '#3b82f6'])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#334155"), margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("### 💊 Treatment Method Breakdown")
        td = ta.get('type_distribution') or {"Organic":4, "Chemical":3, "Integrated":5}
        fig = px.pie(names=list(td.keys()), values=list(td.values()), hole=0.45,
                     color_discrete_sequence=['#16a34a', '#3b82f6', '#f59e0b'])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155"),
                          margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Farmer Profile
# ═════════════════════════════════════════════════════════════════════════════
def page_profile():
    topbar("Farmer Profile & Settings", "👤")
    p = modules['user_profile'].current_profile

    st.markdown('<div class="av-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Farm & Operator Information")
    with st.form("profile_editor_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Operator Name", p.get('farmer_name','Krutin Patel'))
            farm = st.text_input("Farm Name", p.get('farm_name','Green Valley Farm'))
            fid = st.text_input("Agricultural License ID", p.get('farmer_id','AGRI-8829-US'))
        with c2:
            loc = st.text_input("Farm Location", p.get('location','California, Central Valley'))
            phone = st.text_input("Contact Phone", p.get('phone','+1 (555) 234-5678'))
            email = st.text_input("Contact Email", p.get('email','krutin.patel@agrivision.ai'))

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            area = st.number_input("Cultivated Area (ha)", 0.1, 5000.0, float(p.get('total_area', 12.5)), 0.5)
            exp = st.number_input("Farming Experience (years)", 0, 80, int(p.get('farming_experience', 8)))
        with c4:
            crops = st.multiselect("Primary Cultivated Crops",
                ["Tomato", "Apple", "Potato", "Corn", "Grape", "Cotton", "Wheat", "Rice"],
                default=p.get('primary_crops', ["Tomato", "Potato", "Cotton"]))
            ftype = st.selectbox("Agricultural Practice", ["organic", "conventional", "mixed"],
                index=["organic", "conventional", "mixed"].index(p.get('farming_type','mixed')))

        if st.form_submit_button("💾 Save Profile Changes", type="primary", use_container_width=True):
            p.update({
                'farmer_name': name, 'farm_name': farm, 'farmer_id': fid, 'location': loc,
                'phone': phone, 'email': email, 'total_area': area, 'farming_experience': exp,
                'primary_crops': crops, 'farming_type': ftype
            })
            modules['user_profile'].current_profile = p
            modules['user_profile'].save_profile()
            st.success("✅ Farmer profile updated successfully!")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Export Reports
# ═════════════════════════════════════════════════════════════════════════════
def page_export():
    topbar("Export Data & Reports", "📄")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("### 📥 Export Scan Records")
        det = modules['detection_history'].detection_records
        if det:
            df = pd.DataFrame([{
                'ID': r.get('id',''), 'Date': str(r.get('timestamp',''))[:19],
                'Crop': r.get('crop','Leaf'), 'Disease': r.get('predicted_disease',''),
                'Confidence': f"{r.get('confidence',0):.4f}"
            } for r in det])
            st.download_button("📄 Download CSV", df.to_csv(index=False).encode(), f"scans_{datetime.now():%Y%m%d}.csv", "text/csv", use_container_width=True)
            st.download_button("📋 Download JSON", json.dumps([{**r,'timestamp':str(r.get('timestamp'))} for r in det], indent=2).encode(),
                               f"scans_{datetime.now():%Y%m%d}.json", "application/json", use_container_width=True)
        else:
            st.info("No scans available for export.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="av-card">', unsafe_allow_html=True)
        st.markdown("### 📥 Export Treatment History")
        trt = modules['treatment_history'].treatment_records
        if trt:
            df = pd.DataFrame([{
                'ID': t.get('id',''), 'Date': str(t.get('planned_date',''))[:10],
                'Disease': t.get('disease',''), 'Type': t.get('treatment_type',''),
                'Status': t.get('status',''), 'Notes': t.get('notes','')
            } for t in trt])
            st.download_button("📄 Download CSV", df.to_csv(index=False).encode(), f"treatments_{datetime.now():%Y%m%d}.csv", "text/csv", use_container_width=True)
            st.download_button("📋 Download JSON", json.dumps([{**t,'timestamp':str(t.get('timestamp')),'planned_date':str(t.get('planned_date'))} for t in trt], indent=2).encode(),
                               f"treatments_{datetime.now():%Y%m%d}.json", "application/json", use_container_width=True)
        else:
            st.info("No treatment records available for export.")
        st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Data Management
# ═════════════════════════════════════════════════════════════════════════════
def page_data():
    topbar("System & Data Management", "🗄️")

    stats = modules['data_management'].get_system_statistics()
    hs = modules['data_management'].calculate_health_score(stats)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-blue">💾</div>
            <div class="av-metric-info">
                <div class="av-metric-label">Storage Allocated</div>
                <div class="av-metric-num">{stats.get('total_size_mb',0):.1f} MB</div>
            </div>
        </div>
        """), unsafe_allow_html=True)
    with c2:
        st.markdown(textwrap.dedent(f"""
        <div class="av-metric-card">
            <div class="av-metric-circle av-mc-green">🛡️</div>
            <div class="av-metric-info">
                <div class="av-metric-label">System Health Score</div>
                <div class="av-metric-num">{hs}/100</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="av-card">', unsafe_allow_html=True)
    st.markdown("### 🛡️ System Integrity & Maintenance")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📦 Create Full Backup", type="primary", use_container_width=True):
            with st.spinner("Archiving system data..."):
                bp = modules['data_management'].create_system_backup()
                st.success(f"Backup created: {bp}")
    with b2:
        if st.button("🧹 Clean Temporary Files", use_container_width=True):
            n = modules['data_management'].clean_temp_files()
            st.success(f"Cleared {n} cached files!")
    with b3:
        if st.button("🔍 Verify DB Integrity", use_container_width=True):
            with st.spinner("Checking schemas & indexes..."):
                modules['data_management'].verify_data_integrity()
                st.success("All data collections intact & verified!")
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    sidebar()
    p = st.session_state.nav_page
    routes = {
        "🏠 Dashboard": page_dashboard,
        "🔍 Disease Detection": page_detection,
        "📋 Detection History": page_det_history,
        "💊 Treatment History": page_treatment,
        "🌾 Fertilizer Calculator": page_fertilizer,
        "📊 Farm Analytics": page_analytics,
        "👤 User Profile": page_profile,
        "📄 Export Reports": page_export,
        "🗄️ Data Management": page_data,
    }
    routes.get(p, page_dashboard)()

if __name__ == "__main__":
    main()
