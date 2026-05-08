# =====================================
# COPD Detection - Full App with Login
# =====================================

import streamlit as st
import tensorflow as tf
import numpy as np
import os
import json
import hashlib
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================
# Page Config
# =====================================

st.set_page_config(
    page_title="COPD Detection System",
    page_icon="🫁",
    layout="wide"
)

# =====================================
# File Paths
# =====================================

USERS_FILE       = 'users.json'
PREDICTIONS_FILE = 'predictions.json'

# =====================================
# Helper: Hash password
# =====================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================
# Helper: Load / Save Users
# =====================================

def load_users():
    if not os.path.exists(USERS_FILE):
        # Create default users
        default_users = {
            "doctor1": {
                "password": hash_password("doctor123"),
                "role": "doctor",
                "name": "Dr. Smith"
            },
            "patient1": {
                "password": hash_password("patient123"),
                "role": "patient",
                "name": "John Doe"
            },
            "patient2": {
                "password": hash_password("patient456"),
                "role": "patient",
                "name": "Jane Doe"
            }
        }
        save_users(default_users)
        return default_users
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

# =====================================
# Helper: Load / Save Predictions
# =====================================

def load_predictions():
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    with open(PREDICTIONS_FILE, 'r') as f:
        return json.load(f)

def save_prediction(username, result, confidence, raw_score):
    predictions = load_predictions()
    predictions.append({
        "username":   username,
        "result":     result,
        "confidence": round(confidence, 2),
        "raw_score":  round(raw_score, 4),
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)

# =====================================
# Load ML Model
# =====================================

@st.cache_resource
def load_model():
    if not os.path.exists('copd_model.h5'):
        return None
    return tf.keras.models.load_model('copd_model.h5')

model = load_model()

# =====================================
# Session State Init
# =====================================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in  = False
    st.session_state.username   = ''
    st.session_state.role       = ''
    st.session_state.name       = ''

# =====================================
# LOGIN PAGE
# =====================================

def login_page():
    st.markdown("""
        <style>
        .login-box {
            max-width: 420px;
            margin: 60px auto;
            padding: 40px;
            background: #0f172a;
            border-radius: 16px;
            border: 1px solid #1e3a5f;
        }
        .login-title {
            font-size: 2rem;
            font-weight: 700;
            color: #38bdf8;
            text-align: center;
            margin-bottom: 8px;
        }
        .login-subtitle {
            color: #94a3b8;
            text-align: center;
            margin-bottom: 24px;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-title">🫁 COPD Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Medical AI Diagnostic System</div>', unsafe_allow_html=True)
        st.markdown("---")

        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")

        if st.button("Login", use_container_width=True, type="primary"):
            users = load_users()
            if username in users:
                if users[username]['password'] == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.username  = username
                    st.session_state.role      = users[username]['role']
                    st.session_state.name      = users[username]['name']
                    st.rerun()
                else:
                    st.error("❌ Wrong password")
            else:
                st.error("❌ Username not found")

        st.markdown("---")
        st.markdown("**Default Accounts:**")
        st.markdown("""
        | Role | Username | Password |
        |------|----------|----------|
        | 🩺 Doctor | doctor1 | doctor123 |
        | 🧑 Patient | patient1 | patient123 |
        | 🧑 Patient | patient2 | patient456 |
        """)

# =====================================
# PREDICTION CHART
# =====================================

def show_prediction_chart(result, confidence, raw_score):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f172a')

    # --- Chart 1: Confidence Bar ---
    ax1 = axes[0]
    ax1.set_facecolor('#1e293b')

    categories  = ['COPD', 'Normal']
    copd_conf   = (1 - raw_score) * 100
    normal_conf = raw_score * 100
    values      = [copd_conf, normal_conf]
    colors      = ['#ef4444', '#22c55e']

    bars = ax1.bar(categories, values, color=colors, width=0.4, edgecolor='none')

    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{val:.1f}%',
            ha='center', va='bottom',
            color='white', fontsize=13, fontweight='bold'
        )

    ax1.set_ylim(0, 115)
    ax1.set_title('Prediction Confidence', color='white', fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel('Confidence (%)', color='#94a3b8')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#334155')
    ax1.spines['left'].set_color('#334155')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Highlight winner
    if result == 'COPD Detected':
        bars[0].set_edgecolor('#fbbf24')
        bars[0].set_linewidth(2)
    else:
        bars[1].set_edgecolor('#fbbf24')
        bars[1].set_linewidth(2)

    # --- Chart 2: Gauge / Donut ---
    ax2 = axes[1]
    ax2.set_facecolor('#1e293b')

    gauge_val   = confidence / 100
    color_fill  = '#ef4444' if result == 'COPD Detected' else '#22c55e'

    wedge_vals  = [gauge_val, 1 - gauge_val]
    wedge_cols  = [color_fill, '#1e3a5f']

    wedges, _ = ax2.pie(
        wedge_vals,
        colors=wedge_cols,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor='#0f172a', linewidth=3)
    )

    ax2.text(0, 0.1,  f'{confidence:.1f}%',
             ha='center', va='center',
             color='white', fontsize=22, fontweight='bold')
    ax2.text(0, -0.25, result,
             ha='center', va='center',
             color=color_fill, fontsize=11, fontweight='bold')

    ax2.set_title('Result Gauge', color='white', fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# =====================================
# HISTORY CHART (for doctor)
# =====================================

def show_history_chart(predictions):
    if len(predictions) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#0f172a')

    copd_count   = sum(1 for p in predictions if p['result'] == 'COPD Detected')
    normal_count = sum(1 for p in predictions if p['result'] == 'No COPD (Normal)')
    total        = len(predictions)

    # --- Chart 1: Pie chart of results ---
    ax1 = axes[0]
    ax1.set_facecolor('#1e293b')

    if copd_count > 0 or normal_count > 0:
        sizes  = [copd_count, normal_count]
        colors = ['#ef4444', '#22c55e']
        labels = [f'COPD\n{copd_count}', f'Normal\n{normal_count}']
        wedges, texts = ax1.pie(
            sizes, colors=colors, labels=labels,
            startangle=90,
            wedgeprops=dict(edgecolor='#0f172a', linewidth=2),
            textprops=dict(color='white', fontsize=11)
        )

    ax1.set_title('Overall Distribution', color='white', fontsize=13, fontweight='bold')

    # --- Chart 2: Confidence over time ---
    ax2 = axes[1]
    ax2.set_facecolor('#1e293b')

    conf_list   = [p['confidence'] for p in predictions]
    result_list = [p['result'] for p in predictions]
    bar_colors  = ['#ef4444' if r == 'COPD Detected' else '#22c55e' for r in result_list]

    ax2.bar(range(len(conf_list)), conf_list, color=bar_colors, edgecolor='none')
    ax2.axhline(y=80, color='#fbbf24', linestyle='--', alpha=0.7, label='80% threshold')
    ax2.set_ylim(0, 110)
    ax2.set_title('Confidence Per Scan', color='white', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Scan #', color='#94a3b8')
    ax2.set_ylabel('Confidence (%)', color='#94a3b8')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#334155')
    ax2.spines['left'].set_color('#334155')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    red_patch   = mpatches.Patch(color='#ef4444', label='COPD')
    green_patch = mpatches.Patch(color='#22c55e', label='Normal')
    ax2.legend(handles=[red_patch, green_patch], facecolor='#1e293b',
               labelcolor='white', fontsize=9)

    # --- Chart 3: Accuracy summary bar ---
    ax3 = axes[2]
    ax3.set_facecolor('#1e293b')

    avg_conf = np.mean(conf_list)
    high_conf = sum(1 for c in conf_list if c >= 80)
    low_conf  = total - high_conf

    bars = ax3.bar(
        ['Avg\nConfidence', 'High Conf\n(≥80%)', 'Low Conf\n(<80%)'],
        [avg_conf, high_conf / total * 100, low_conf / total * 100],
        color=['#38bdf8', '#22c55e', '#f97316'],
        edgecolor='none', width=0.5
    )

    for bar in bars:
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{bar.get_height():.1f}%',
            ha='center', color='white', fontsize=11, fontweight='bold'
        )

    ax3.set_ylim(0, 115)
    ax3.set_title('Accuracy Summary', color='white', fontsize=13, fontweight='bold')
    ax3.set_ylabel('%', color='#94a3b8')
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('#334155')
    ax3.spines['left'].set_color('#334155')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# =====================================
# PATIENT PAGE
# =====================================

def patient_page():
    st.markdown(f"### 👋 Welcome, {st.session_state.name}")
    st.markdown("---")

    if model is None:
        st.error("⚠️ Model not loaded. Please contact your administrator.")
        return

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📤 Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded X-Ray", use_container_width=True)

    if uploaded_file:
        # Preprocess
        img_resized = img.resize((224, 224)).convert('RGB')
        img_array   = np.array(img_resized, dtype=np.float32)
        img_array   = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("🔍 Analyzing..."):
            prediction = model.predict(img_array)

        raw_score  = float(prediction[0][0])
        is_copd    = raw_score < 0.5
        result     = "COPD Detected" if is_copd else "No COPD (Normal)"
        confidence = (1 - raw_score) * 100 if is_copd else raw_score * 100

        # Save to history
        save_prediction(st.session_state.username, result, confidence, raw_score)

        with col_result:
            st.subheader("📊 Result")
            if is_copd:
                st.error(f"🔴 **{result}**")
            else:
                st.success(f"🟢 **{result}**")

            st.metric("Confidence", f"{confidence:.2f}%")
            st.caption("⚠️ For educational purposes only. Consult a doctor.")

        st.markdown("---")
        st.subheader("📈 Prediction Analysis")
        show_prediction_chart(result, confidence, raw_score)

    # Patient's own history
    st.markdown("---")
    st.subheader("📋 My Scan History")
    all_preds    = load_predictions()
    my_preds     = [p for p in all_preds if p['username'] == st.session_state.username]

    if len(my_preds) == 0:
        st.info("No scans yet. Upload an image above to get started.")
    else:
        st.markdown(f"**Total scans:** {len(my_preds)}")
        show_history_chart(my_preds)

        for p in reversed(my_preds):
            color = "🔴" if p['result'] == 'COPD Detected' else "🟢"
            st.markdown(
                f"{color} **{p['result']}** — Confidence: `{p['confidence']}%` — {p['timestamp']}"
            )

# =====================================
# DOCTOR PAGE
# =====================================

def doctor_page():
    st.markdown(f"### 👨‍⚕️ Welcome, {st.session_state.name}")
    st.markdown("---")

    all_preds = load_predictions()
    users     = load_users()
    patients  = {u: d for u, d in users.items() if d['role'] == 'patient'}

    # ---- Summary Cards ----
    total      = len(all_preds)
    copd_count = sum(1 for p in all_preds if p['result'] == 'COPD Detected')
    norm_count = total - copd_count
    avg_conf   = np.mean([p['confidence'] for p in all_preds]) if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Scans",    total)
    c2.metric("🔴 COPD Cases",     copd_count)
    c3.metric("🟢 Normal Cases",   norm_count)
    c4.metric("📊 Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    # ---- Overall charts ----
    st.subheader("📈 Overall Statistics")
    if total > 0:
        show_history_chart(all_preds)
    else:
        st.info("No predictions recorded yet.")

    st.markdown("---")

    # ---- Per-patient view ----
    st.subheader("👥 Patient Records")

    selected_patient = st.selectbox(
        "Select a patient to view their scans:",
        options=["All Patients"] + list(patients.keys()),
        format_func=lambda u: f"{users[u]['name']} ({u})" if u != "All Patients" else "All Patients"
    )

    if selected_patient == "All Patients":
        filtered = all_preds
    else:
        filtered = [p for p in all_preds if p['username'] == selected_patient]

    if len(filtered) == 0:
        st.info("No scans found for this selection.")
    else:
        st.markdown(f"**Showing {len(filtered)} scan(s)**")

        if selected_patient != "All Patients":
            show_history_chart(filtered)

        # Table
        st.markdown("#### Scan Records")
        header = st.columns([2, 2, 2, 2, 2])
        header[0].markdown("**Patient**")
        header[1].markdown("**Result**")
        header[2].markdown("**Confidence**")
        header[3].markdown("**Raw Score**")
        header[4].markdown("**Date & Time**")

        for p in reversed(filtered):
            row = st.columns([2, 2, 2, 2, 2])
            name = users[p['username']]['name'] if p['username'] in users else p['username']
            row[0].write(name)
            if p['result'] == 'COPD Detected':
                row[1].error("🔴 COPD")
            else:
                row[1].success("🟢 Normal")
            row[2].write(f"{p['confidence']}%")
            row[3].write(f"{p['raw_score']}")
            row[4].write(p['timestamp'])

    st.markdown("---")

    # ---- Upload scan for a patient ----
    st.subheader("📤 Run a Scan for a Patient")

    if model is None:
        st.error("⚠️ Model not loaded.")
        return

    selected_for_scan = st.selectbox(
        "Select patient:",
        options=list(patients.keys()),
        format_func=lambda u: f"{users[u]['name']} ({u})",
        key="scan_patient"
    )

    uploaded_file = st.file_uploader(
        "Upload X-Ray Image",
        type=['jpg', 'jpeg', 'png'],
        key="doctor_upload"
    )

    if uploaded_file:
        img         = Image.open(uploaded_file)
        img_resized = img.resize((224, 224)).convert('RGB')
        img_array   = np.array(img_resized, dtype=np.float32)
        img_array   = np.expand_dims(img_array, axis=0) / 255.0

        col_img, col_res = st.columns(2)
        with col_img:
            st.image(img, caption="Uploaded X-Ray", use_container_width=True)

        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)

        raw_score  = float(prediction[0][0])
        is_copd    = raw_score < 0.5
        result     = "COPD Detected" if is_copd else "No COPD (Normal)"
        confidence = (1 - raw_score) * 100 if is_copd else raw_score * 100

        save_prediction(selected_for_scan, result, confidence, raw_score)

        with col_res:
            st.subheader("Result")
            if is_copd:
                st.error(f"🔴 **{result}**")
            else:
                st.success(f"🟢 **{result}**")
            st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("📈 Prediction Analysis")
        show_prediction_chart(result, confidence, raw_score)

# =====================================
# MAIN ROUTER
# =====================================

if not st.session_state.logged_in:
    login_page()
else:
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 🫁 COPD System")
        st.markdown("---")
        role_icon = "🩺" if st.session_state.role == "doctor" else "🧑"
        st.markdown(f"**{role_icon} {st.session_state.name}**")
        st.markdown(f"Role: `{st.session_state.role.capitalize()}`")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ''
            st.session_state.role      = ''
            st.session_state.name      = ''
            st.rerun()

    if st.session_state.role == 'doctor':
        doctor_page()
    else:
        patient_page()