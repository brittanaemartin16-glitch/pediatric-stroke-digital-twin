from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
import streamlit as st


st.set_page_config(
    page_title="Pediatric Stroke Digital Twin",
    page_icon="🧠",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
PATIENT_IMAGE = ROOT / "patient-digital-twin.png"

st.markdown(
    """
    <style>
    .stApp {background: #f5f8fa; color: #18344a;}
    .block-container {max-width: 1250px; padding-top: 1.7rem;}
    h1, h2, h3 {color: #123b5d;}
    .tag {display:inline-block; padding:.35rem .7rem; border-radius:999px;
          background:#dff3ef; color:#147568; font-weight:700; font-size:.82rem;}
    .notice {background:#fff7e7; border:1px solid #edcc84; border-radius:12px;
             padding:.8rem 1rem; color:#614b20;}
    .result {background:white; border:1px solid #d6e3ea; border-radius:16px;
             padding:1rem 1.2rem; min-height:132px; box-shadow:0 4px 15px #18344a0d;}
    .result p {margin:0; color:#5b7181; font-size:.9rem;}
    .result strong {font-size:2.1rem; color:#123b5d;}
    .small {color:#627786; font-size:.84rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def clamp(value):
    return max(3, min(92, round(value)))


def calculate(age, lesion, seizure_burden, eeg, treatment_response,
              cortical, multifocal, genetic):
    base = (
        7
        + max(0, 12 - age) * 0.75
        + lesion * 0.22
        + seizure_burden * 7
        + eeg * 0.16
        + (12 if cortical else 0)
        + (9 if multifocal else 0)
        + (5 if genetic else 0)
        - treatment_response * 0.11
    )
    trajectory = {
        "Discharge": clamp(base - 7),
        "6 months": clamp(base),
        "24 months": clamp(base + seizure_burden * 2 + eeg * 0.04),
    }
    factors = {
        "Cortical involvement": 12 if cortical else 0,
        "Acute seizure burden": seizure_burden * 7,
        "EEG abnormality": eeg * 0.16,
        "Lesion characteristics": lesion * 0.22,
        "Younger age": max(0, 12 - age) * 0.75,
        "Multifocal infarction": 9 if multifocal else 0,
        "Exploratory genetic signal": 5 if genetic else 0,
    }
    factors = dict(sorted(factors.items(), key=lambda item: item[1], reverse=True))
    response = round(treatment_response * 0.72 + (100 - eeg) * 0.18)
    return trajectory, factors, response


def patient_visual(lesion, eeg, cortical, multifocal, risk):
    image = Image.open(PATIENT_IMAGE).convert("RGBA")
    image = ImageEnhance.Color(image).enhance(1.03)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    color = (43, 183, 168, 145) if risk < 30 else (242, 162, 58, 160)
    if risk >= 60:
        color = (239, 103, 93, 175)

    cx = int(image.width * (0.58 if cortical else 0.52))
    cy = int(image.height * (0.285 if cortical else 0.34))
    radius = int(22 + lesion * 0.42)
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color)
    if multifocal:
        r2 = max(15, radius // 2)
        draw.ellipse((cx-95-r2, cy+35-r2, cx-95+r2, cy+35+r2), fill=color)

    points = []
    start_x, end_x = int(image.width * 0.41), int(image.width * 0.89)
    baseline = int(image.height * 0.38)
    amplitude = 7 + eeg * 0.18
    for x in range(start_x, end_x, 5):
        wave = np.sin((x-start_x) / 13) + 0.45 * np.sin((x-start_x) / 4.4)
        points.append((x, baseline + int(amplitude * wave)))
    draw.line(points, fill=(40, 184, 205, 225), width=3)
    return Image.alpha_composite(image, overlay).convert("RGB")

st.title("Pediatric Stroke Digital Twin")
st.write(
    "Explore how clinical, imaging, EEG, genetic, and treatment information could be "
    "combined for one fictional child. Changing the inputs updates the brain display, "
    "the post-stroke epilepsy research estimate, and the explanation of the estimate."
)
st.markdown(
    '<div class="notice"><b>Synthetic research prototype:</b> This demonstration uses '
    'illustrative scoring and fictional values. It cannot diagnose epilepsy, predict a '
    'real child’s outcome, or recommend treatment.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Fictional patient PT-017")
    st.caption("Change these values to update the digital twin.")
    age = st.slider("Age at stroke", 1, 16, 8)
    lesion = st.slider("Lesion index (%)", 5, 90, 38)
    seizure_burden = st.slider("Acute seizure burden", 0, 5, 2)
    eeg = st.slider("EEG abnormality index", 0, 100, 46)
    treatment_response = st.slider("Observed treatment response", 0, 100, 55)
    cortical = st.toggle("Cortical involvement", True)
    multifocal = st.toggle("Multifocal infarction", False)
    genetic = st.toggle("Exploratory genetic signal", False)

trajectory, factors, response = calculate(
    age, lesion, seizure_burden, eeg, treatment_response,
    cortical, multifocal, genetic,
)

stage = st.radio("Recovery stage", list(trajectory), horizontal=True)
risk = trajectory[stage]
category = "Lower" if risk < 30 else "Moderate" if risk < 60 else "Higher"

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.subheader("Living digital profile")
    st.image(
        patient_visual(lesion, eeg, cortical, multifocal, risk),
        caption="The highlighted brain region and EEG line update with the selected inputs.",
        use_container_width=True,
    )
with right:
    st.subheader("Updated research forecast")
    a, b = st.columns(2)
    with a:
        st.markdown(
            f'<div class="result"><p>Estimated PSE research risk</p><strong>{risk}%</strong>'
            f'<br><span class="tag">{category}</span></div>', unsafe_allow_html=True
        )
    with b:
        st.markdown(
            f'<div class="result"><p>Candidate-response signal</p><strong>{response}%</strong>'
            '<br><span class="small">Research hypothesis only</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Dynamic recovery trajectory")
    chart_data = pd.DataFrame(
        {"Recovery point": list(trajectory.keys()), "Illustrative estimate": list(trajectory.values())}
    ).set_index("Recovery point")
    st.bar_chart(chart_data, color="#2a9d8f", horizontal=True)

st.markdown("---")
exp_col, framework_col = st.columns([1.2, 1], gap="large")
with exp_col:
    st.subheader("What shaped this estimate?")
    st.caption("A SHAP-style explanation ranks which selected factors affected this demonstration most.")
    factor_df = pd.DataFrame(
        {"Factor": list(factors.keys())[:6], "Relative contribution": list(factors.values())[:6]}
    ).set_index("Factor")
    st.bar_chart(factor_df, color="#247ba0", horizontal=True)
with framework_col:
    st.subheader("How it connects to the full project")
    st.markdown(
        """
        - **Theme A:** identifies possible molecular treatments and models hydrogel delivery.
        - **Theme B:** uses the fictional child’s information to update epilepsy-risk and treatment-response research estimates.
        - **Theme C:** examines whether distance, transportation, caregiver awareness, or EEG access could delay follow-up care.
        """
    )
    st.info(
        "A real version would require de-identified multicenter pediatric data, model training, "
        "external validation, calibration, fairness checks, and clinical oversight."
    )

