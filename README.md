# 🩺 Breast Cancer Predictor App

This is a **Streamlit web application** that predicts whether a breast tumor is **Benign** or **Malignant** based on diagnostic features extracted from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## 🚀 Live Demo

You can try the app here 👉 [Streamlit App Link](https://appbreastcancerpredict-46eaj2q7xymwdihelc2tjr.streamlit.app)  

## 🧠 Model Summary

The prediction model was trained using the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which includes features such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Concavity
- Concave Points
- Symmetry
- Fractal Dimension

These features are used to classify the tumor into one of two categories:
- `Benign` (non-cancerous)
- `Malignant` (cancerous)

---

## 🗂️ Project Structure

StreamLit_App_Cancer/
│
├── app.py                      # 🔸 Main Streamlit application
├── dataset/
│   └── clean_data.csv          # 🔹 Preprocessed dataset used in the app
│
├── model/
│   └── cancer_model.pkl        # 🔹 Trained ML model (e.g. Logistic Regression, Random Forest)
│
├── myenv/                      # 🔸 Virtual environment (optional – don't include in GitHub)
│
├── requirements.txt            # 🔸 List of Python packages required to run the app
├── README.md   
