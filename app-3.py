from pathlib import Path

import altair as alt
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
PATIENT_IMAGE = ROOT / "patient-digital-twin-natural.png"
CT_IMAGE = ROOT / "synthetic-pediatric-ct.png"

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
    .hero {background:linear-gradient(135deg,#123b5d,#176b78); color:white;
           padding:1.8rem 2rem; border-radius:20px; margin-bottom:1rem;}
    .hero h1 {color:white; margin:0 0 .5rem 0; font-size:2.5rem;}
    .hero p {font-size:1rem; line-height:1.45; margin:0; max-width:920px;}
    .section-label {color:#147568; font-weight:800; letter-spacing:.08em;
                    text-transform:uppercase; font-size:.78rem; margin-top:1.3rem;}
    .project-card {background:white; border:1px solid #d6e3ea; border-radius:16px;
                   padding:1rem 1.1rem; min-height:175px; box-shadow:0 4px 15px #18344a0d;}
    .project-card h3 {font-size:1.05rem; margin:.25rem 0 .55rem 0;}
    .project-card p {color:#536b7a; line-height:1.45; font-size:.91rem;}
    .number {display:inline-flex; width:30px; height:30px; align-items:center;
             justify-content:center; border-radius:50%; background:#dff3ef;
             color:#147568; font-weight:800;}
    .definition {background:#eef7f8; border-left:5px solid #2a9d8f;
                 padding:.85rem 1rem; border-radius:8px; color:#254b59;}
    .interpret {background:white; border:1px solid #d6e3ea; border-radius:14px;
                padding:1rem 1.2rem; color:#385363; line-height:1.5;}
    .flow {display:grid; grid-template-columns:1fr auto 1fr auto 1fr; gap:.7rem;
           align-items:center; margin:.7rem 0 1.2rem;}
    .flow-card {background:white; border:1px solid #d6e3ea; border-radius:16px;
                padding:1rem; text-align:center; min-height:125px;}
    .flow-card .icon {font-size:2rem; display:block; margin-bottom:.25rem;}
    .flow-card b {color:#123b5d; font-size:1.05rem;}
    .flow-card small {display:block; color:#627786; margin-top:.35rem;}
    .arrow {font-size:1.8rem; color:#2a9d8f; font-weight:800;}
    .update {background:#eaf7f4; border:1px solid #9bd8cd; border-radius:14px;
             padding:.85rem 1rem; color:#234c52; font-size:1rem;}
    @media (max-width: 800px) {
      .flow {grid-template-columns:1fr;}
      .arrow {transform:rotate(90deg); text-align:center;}
    }
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


def ct_visual(lesion, cortical, multifocal, risk):
    """Create a responsive overlay on the synthetic CT-style image."""
    image = Image.open(CT_IMAGE).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    color = (43, 183, 168, 145) if risk < 30 else (242, 162, 58, 165)
    if risk >= 60:
        color = (239, 103, 93, 185)

    cx = int(image.width * (0.66 if cortical else 0.38))
    cy = int(image.height * (0.43 if cortical else 0.56))
    radius = int(max(12, image.width * (0.025 + lesion * 0.00075)))
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color)

    if multifocal:
        r2 = max(9, radius // 2)
        cx2, cy2 = int(image.width * 0.39), int(image.height * 0.39)
        draw.ellipse((cx2-r2, cy2-r2, cx2+r2, cy2+r2), fill=color)

    return Image.alpha_composite(image, overlay).convert("RGB")


st.markdown(
    '<div class="hero"><h1>Pediatric Stroke Digital Twin</h1>'
    '<p>One patient. Multiple data sources. A changing view of recovery.</p></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="notice"><b>Synthetic research prototype:</b> This demonstration uses '
    'illustrative scoring and fictional values. It cannot diagnose epilepsy, predict a '
    'real child’s outcome, or recommend treatment.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">From Discovery to Patient Care</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="flow">'
    '<div class="flow-card"><span class="icon">🧪</span><b>A · Discover</b><small>Targets + hydrogel delivery</small></div>'
    '<div class="arrow">→</div>'
    '<div class="flow-card"><span class="icon">🧠</span><b>B · Predict</b><small>Patient-specific digital twin</small></div>'
    '<div class="arrow">→</div>'
    '<div class="flow-card"><span class="icon">🏥</span><b>C · Deliver</b><small>Access to follow-up care</small></div>'
    '</div>', unsafe_allow_html=True,
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

st.markdown('<div class="section-label">Interactive demonstration</div>', unsafe_allow_html=True)
st.header("Patient-specific profile")
stage = st.radio("Select a recovery time point", list(trajectory), horizontal=True)
risk = trajectory[stage]
category = "Lower" if risk < 30 else "Moderate" if risk < 60 else "Higher"

active_factors = [name for name, value in factors.items() if value > 0]
top_factor = active_factors[0] if active_factors else "No major selected factor"
if risk < 30:
    update_text = f"Lower estimate at {stage}. {top_factor} has the strongest influence."
elif risk < 60:
    update_text = f"Moderate estimate at {stage}. The main driver is {top_factor.lower()}."
else:
    update_text = f"Higher estimate at {stage}. {top_factor} contributes most to this result."
st.markdown(f'<div class="update"><b>Live update:</b> {update_text}</div>', unsafe_allow_html=True)

left, middle = st.columns(2, gap="large")
with left:
    st.subheader("Digital patient")
    st.image(
        patient_visual(lesion, eeg, cortical, multifocal, risk),
        caption="Highlight = lesion estimate · Cyan line = EEG pattern",
        use_container_width=True,
    )
with middle:
    st.subheader("Synthetic CT view")
    st.image(
        ct_visual(lesion, cortical, multifocal, risk),
        caption="Highlight changes with lesion size, location, pattern, and risk · Not a real scan",
        use_container_width=True,
    )

st.markdown('<div class="section-label">Model outputs</div>', unsafe_allow_html=True)
st.header("Live research forecast")
a, b, c = st.columns(3, gap="large")
with a:
    st.markdown(
        f'<div class="result"><p>Estimated PSE research risk at {stage}</p><strong>{risk}%</strong>'
        f'<br><span class="tag">{category}</span></div>', unsafe_allow_html=True
    )
with b:
    st.markdown(
        f'<div class="result"><p>Candidate-response signal</p><strong>{response}%</strong>'
        '<br><span class="small">Research hypothesis only</span></div>',
        unsafe_allow_html=True,
    )
with c:
    st.markdown(
        f'<div class="result"><p>Strongest selected contributor</p>'
        f'<strong style="font-size:1.35rem">{top_factor}</strong>'
        '<br><span class="small">Highest relative contribution</span></div>',
        unsafe_allow_html=True,
    )

trajectory_col, meaning_col = st.columns([1.25, .75], gap="large")
with trajectory_col:
    st.subheader("Risk trajectory across recovery")
    chart_data = pd.DataFrame(
        {"Recovery point": list(trajectory.keys()), "Illustrative estimate": list(trajectory.values())}
    ).set_index("Recovery point")
    st.bar_chart(chart_data, color="#2a9d8f", horizontal=True)
with meaning_col:
    st.subheader("Current reading")
    st.markdown(
        f'<div class="interpret"><b>{stage}:</b> {category} research range.<br><br>'
        f'<b>Main influence:</b> {top_factor}.<br><br>'
        '<span class="small">Illustration only—not a diagnosis.</span></div>', unsafe_allow_html=True
    )

st.markdown('<div class="section-label">Explainable artificial intelligence</div>', unsafe_allow_html=True)
st.header("What shaped the estimate?")

factor_df = pd.DataFrame(
    {"Factor": list(factors.keys())[:7], "Relative contribution": list(factors.values())[:7]}
)
factor_chart = (
    alt.Chart(factor_df)
    .mark_bar(cornerRadiusEnd=6, color="#247ba0", size=27)
    .encode(
        x=alt.X("Relative contribution:Q", title="Relative contribution", axis=alt.Axis(labelFontSize=14, titleFontSize=15)),
        y=alt.Y(
            "Factor:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelFontSize=15, labelLimit=310, labelPadding=10),
        ),
        tooltip=["Factor:N", alt.Tooltip("Relative contribution:Q", format=".1f")],
    )
    .properties(height=390)
)
st.altair_chart(factor_chart, use_container_width=True)
st.markdown(
    f'<div class="update"><b>Chart update:</b> {top_factor} currently has the largest influence. '
    'Longer bars show greater influence—not proof of cause.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Model pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="flow">'
    '<div class="flow-card"><span class="icon">📥</span><b>Inputs</b><small>Clinical · MRI/CT · EEG · Genetic</small></div>'
    '<div class="arrow">→</div>'
    '<div class="flow-card"><span class="icon">⚙️</span><b>Model</b><small>Combine · Compare · Explain</small></div>'
    '<div class="arrow">→</div>'
    '<div class="flow-card"><span class="icon">📊</span><b>Outputs</b><small>Risk · Response · Key factors</small></div>'
    '</div>', unsafe_allow_html=True,
)

with st.expander("Research limitations and next steps"):
    st.write(
        "This is a synthetic demonstration, not a clinical tool. A real version would require "
        "multicenter pediatric data, ethical approval, external validation, fairness testing, "
        "secure data handling, and clinical oversight."
    )

st.caption(
    "Team 7 Pediatric Stroke Research Project · Interactive values and images are synthetic and "
    "provided for research communication only."
)
