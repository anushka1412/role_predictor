import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Load Model and Encoder
# ----------------------------
model = joblib.load("career_prediction_model.pkl")
role_encoder = joblib.load("role_encoder.pkl")

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Career Recommendation System")
st.write("Select your interest level for each skill and click **Predict Career**.")

# ----------------------------
# Interest Mapping
# ----------------------------
interest_map = {
    "Not Interested": 0,
    "Beginner": 1,
    "Intermediate": 2,
    "Professional": 3
}

# --------------------------------------------------
# Replace these names with your dataset column names
# --------------------------------------------------

skills = [
    "Python",
    "Java",
    "C++",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "Data Analysis",
    "Statistics",
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Cloud Computing",
    "Cyber Security",
    "Networking",
    "Linux",
    "Communication Skills"
]

# ----------------------------
# Input Section
# ----------------------------

user_input = []

col1, col2 = st.columns(2)

for i, skill in enumerate(skills):

    if i % 2 == 0:
        choice = col1.selectbox(
            skill,
            list(interest_map.keys()),
            key=skill
        )
    else:
        choice = col2.selectbox(
            skill,
            list(interest_map.keys()),
            key=skill
        )

    user_input.append(interest_map[choice])

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Career"):

    input_df = pd.DataFrame([user_input], columns=skills)

    prediction = model.predict(input_df)

    role = role_encoder.inverse_transform(prediction)

    st.success(f"Recommended Career Role: **{role[0]}**")
