import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load Dataset
def get_clean_data():
    try:
        df = pd.read_csv("dataset/clean_data.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please check if 'clean_data.csv' is in the dataset folder.")
        return pd.DataFrame()

# -- Sidebar Title --
def add_sidebar():
    st.sidebar.header("Cell Nuclei Mearsurments")
    data = get_clean_data()

    # -- Column names of sidebar --
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}
    # loop through labels to create sliders for each label.
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())     # Default value
        )
    return input_dict

# --- Getting scaled values for my Radar Chart ---
def get_scaled_values(input_dict):
    data = get_clean_data()
    # Remove Diagnosis column
    X = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}  # What we are returning
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

# -------- Creating our radar chart --------
def get_radar_chart(input_data):
    # Adding scaled values to our radar chart
    input_data = get_scaled_values(input_data)

    categories = [
        'Radius', 'Texture', 'Perimeter',
        'Area', 'Smoothness', 'Compactness',
        'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension'
        ]
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta = categories,
        fill = 'toself',
        name = 'Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
            input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta = categories,
        fill = 'toself',
        name = 'Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
    r = [
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta = categories,
        fill = 'toself',
        name = 'Worst Value'
    ))

    fig.update_layout(
        polar = dict(
            radialaxis = dict(
                visible = True,
                range = [0, 1]
            )),
        showlegend = True
    )
    return fig

# --- Importing our model ---
def add_predictions(input_data):
    model = pickle.load(open("model/log_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    # Convertin our input data dictionary to an array
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    # Model predicting
    pred = model.predict(input_array_scaled)

    # Styled container
    st.markdown("""
        <style>
        .card {
            background-color: #111827;
            padding: 25px 20px;
            border-radius: 15px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            color: white;
            margin-top: 20px;
            font-family: 'Segoe UI', sans-serif;
        }
        .card h3 {
            color: #facc15;
            margin-bottom: 10px;
        }
        .card p {
            margin: 10px 0;
        }
        .card code {
            background-color: #1f2937;
            padding: 3px 6px;
            border-radius: 5px;
        }
        </style>
        <div class="card">
    """, unsafe_allow_html=True)

    # Header Title
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if pred[0] == 0:
        st.write("Benign")
    else:
        st.write("Malicious")

    st.write("Probability of being benign:", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious:", model.predict_proba(input_array_scaled)[0][1])
    st.write("This app can assit medic professions in making a diagnosis, but not be used as a subtitute for a professional diagnosis.")

    # END of styled container
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title = "Breast Canncer Predictor",
        page_icon = ":female-doctor:",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )
    input_data = add_sidebar()

    # -- Title & Description --
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write(
        "Welcome to the Breast Cancer Predictor App. Please connect this app to your cytology lab to help diagnose "
        "breast cancer from your tissue sample. This app uses a machine learning model to predict whether a breast mass "
        "is **benign** or **malignant** based on the lab measurements it receives. "
    )
        
    # -- Columns --
    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

if __name__ == "__main__":

    main()
