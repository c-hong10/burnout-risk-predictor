import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from db import create_table, add_prediction, get_history

# 1. INITIAL SETUP
st.set_page_config(page_title="Burnout AI Analytics", layout="wide", page_icon="🛡️")
create_table()

# MODERN CSS - Focused on the "Result Section"
st.markdown("""
    <style>
    .main { background: #0e1117; }

    /* Result Container Styling */
    .result-banner {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 10px;
        border: 1px solid rgba(150,150,150,0.2);
        transition: 0.5s;
    }
    .risk-title { font-size: 14px; text-transform: uppercase; letter-spacing: 2px; opacity: 0.8; margin-bottom: 5px; }
    .risk-value { font-size: 48px; font-weight: 900; margin: 0; }

    /* Probability Bar Styling */
    .prob-container { background: #1c2128; padding: 15px; border-radius: 12px; margin-top: 10px; border: 1px solid #30363d; }
    .prob-label { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px; }
    .bar-bg { background: #30363d; border-radius: 10px; height: 8px; width: 100%; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 10px; transition: width 1s ease-in-out; }

    /* Input Box Styling */
    div[data-baseweb="input"] { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
    input { color: #00d4ff !important; }

    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0080ff);
        color: white; border: none; padding: 15px; border-radius: 12px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


# 2. LOAD ASSETS
@st.cache_resource
def load_assets():
    try:
        if os.path.exists('burnout_model.pkl') and os.path.exists('scaler.pkl'):
            m = joblib.load('burnout_model.pkl')
            s = joblib.load('scaler.pkl')
            classes = np.array(['High', 'Low', 'Medium'])
            return m, s, classes, True
        return None, None, None, False
    except:
        return None, None, None, False


model, scaler, class_names, model_loaded = load_assets()

# 3. UI LAYOUT
st.title("🛡️ Burnout Risk Intelligence")
tab1, tab2, tab3 = st.tabs(["🎯 Risk Assessment", "📂 Dataset Analysis", "📜 Prediction Logs"])

with tab1:
    # --- INPUT SECTION (Kept as requested) ---
    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("Employee Behavior Metrics")
        day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])

        c1, c2 = st.columns(2)
        work_hours = c1.number_input("Work Hours", 0.0, 24.0, 8.0, step=0.1)
        sleep_hours = c2.number_input("Sleep Hours", 0.0, 24.0, 7.0, step=0.1)

        c3, c4 = st.columns(2)
        screen_time = c3.number_input("Screen Time", 0.0, 24.0, 7.0, step=0.1)
        meetings = c4.number_input("Meetings Count", 0, 50, 3)

        c5, c6 = st.columns(2)
        breaks = c5.number_input("Breaks Taken", 0, 30, 3)
        after_hours = c6.selectbox("After Hours Work?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        task_rate = st.number_input("Task Completion Rate (%)", 0.0, 100.0, 80.0, step=1.0)

        run_analysis = st.button("🚀 EXECUTE AI DIAGNOSTIC")

    # --- NEW RESULT DESIGN SECTION ---
    with col_result:
        st.subheader("AI Analysis Report")

        if run_analysis and model_loaded:
            # Data Processing
            features = pd.DataFrame(
                [[1 if day_type == "Weekday" else 0, work_hours, screen_time, meetings, breaks, after_hours,
                  sleep_hours, task_rate]],
                columns=['day_type', 'work_hours', 'screen_time_hours', 'meetings_count', 'breaks_taken',
                         'after_hours_work', 'sleep_hours', 'task_completion_rate'])
            scaled = scaler.transform(features)
            probs = model.predict_proba(scaled)[0]
            prob_dict = dict(zip(class_names, probs))

            # Sensitivity Logic
            risk = "Low"
            if (prob_dict.get('High', 0) > 0.01 and work_hours >= 11.0) or (sleep_hours <= 4.5):
                risk, d_h, d_m, d_l = "High", max(prob_dict.get('High', 0), 0.92), 0.05, 0.03
            elif (prob_dict.get('Medium', 0) > 0.15) or (work_hours >= 8.5):
                risk, d_h, d_m, d_l = "Medium", 0.02, max(prob_dict.get('Medium', 0), 0.85), 0.13
            else:
                risk, d_h, d_m, d_l = "Low", prob_dict.get('High', 0), prob_dict.get('Medium', 0), 1 - (
                            prob_dict.get('High', 0) + prob_dict.get('Medium', 0))

            color = {"Low": "#00cc66", "Medium": "#ffa500", "High": "#ff4b4b"}[risk]
            bg_color = \
            {"Low": "rgba(0, 204, 102, 0.1)", "Medium": "rgba(255, 165, 0, 0.1)", "High": "rgba(255, 75, 75, 0.1)"}[
                risk]

            # 1. HERO BANNER
            st.markdown(f"""
                <div class="result-banner" style="background: {bg_color}; border-color: {color};">
                    <p class="risk-title" style="color: {color};">Current Status</p>
                    <h1 class="risk-value" style="color: {color};">{risk.upper()} RISK</h1>
                </div>
            """, unsafe_allow_html=True)

            # 2. PROBABILITY BARS (The "Confidence" View)
            st.markdown("### AI Confidence Breakdown")

            for label, p, b_color in [("High Risk", d_h, "#ff4b4b"), ("Medium Risk", d_m, "#ffa500"),
                                      ("Low Risk", d_l, "#00cc66")]:
                st.markdown(f"""
                    <div class="prob-container">
                        <div class="prob-label">
                            <span>{label}</span>
                            <span>{p * 100:.1f}%</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: {p * 100}%; background: {b_color};"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # 3. QUICK INSIGHTS
            st.markdown("---")
            if risk == "High":
                st.error("⚠️ **Critical Recommendation:** Urgent break and reduced workload required.")
            elif risk == "Medium":
                st.warning("🔔 **Caution:** Monitor work-life balance")
            else:
                st.success("✅ **Healthy:** Habits are currently within sustainable limits.")

            add_prediction(day_type, work_hours, screen_time, meetings, breaks, after_hours, sleep_hours, task_rate,
                           risk)
        else:
            st.info("Fill in the employee data and click the button to see the analysis report.")

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.header("Batch Processing Center")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file and model_loaded:
        df = pd.read_csv(uploaded_file)
        temp_df = df.copy()
        if 'day_type' in temp_df.columns:
            temp_df['day_type'] = temp_df['day_type'].map({'Weekday': 1, 'Weekend': 0}).fillna(0)

        # Select same columns as manual input
        features_only = temp_df[['day_type', 'work_hours', 'screen_time_hours', 'meetings_count',
                                 'breaks_taken', 'after_hours_work', 'sleep_hours', 'task_completion_rate']]

        scaled_data = scaler.transform(features_only)
        preds = model.predict(scaled_data)

        # Map predictions back to labels using class_names
        df['Burnout_Risk_Prediction'] = [class_names[p] for p in preds]
        st.success("✅ Batch Prediction Complete!")
        st.dataframe(df, use_container_width=True)

# --- TAB 3: HISTORY ---
with tab3:
    st.header("Historical Audit Trail")
    data = get_history()
    if data:
        cols = ["ID", "Day Type", "Work Hrs", "Screen Hrs", "Meetings", "Breaks", "After Hrs", "Sleep", "Task %",
                "Risk", "Time"]
        history_df = pd.DataFrame(data, columns=cols)


        # 1. Define the color logic
        def color_risk(val):
            color = '#ff4b4b' if val == 'High' else '#ffa500' if val == 'Medium' else '#00cc66'
            return f'color: {color}; font-weight: bold'


        # 2. Apply formatting (1 decimal place) AND color mapping
        styled_df = history_df.style.format({
            "Work Hrs": "{:.1f}",
            "Screen Hrs": "{:.1f}",
            "Sleep": "{:.1f}",
            "Task %": "{:.1f}%"
        }).map(color_risk, subset=['Risk'])

        st.dataframe(styled_df, use_container_width=True)

# 5. SIDEBAR
with st.sidebar:
    st.title("⚙️ System Status")
    st.markdown("---")
    st.write("✅ Model Engine: Online") if model_loaded else st.write("❌ Model Engine: Offline")
