import streamlit as st
import joblib
import pandas as pd




st.set_page_config(
    page_title="Career Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

h1{
    text-align:center;
    color:#4CAF50;
}

.block-container{
    padding-top:2rem;
}

div.stButton > button{
    width:100%;
    height:60px;
    border-radius:15px;
    background:#4CAF50;
    color:white;
    font-size:22px;
    font-weight:bold;
}

div.stButton > button:hover{
    background:#45a049;
}

[data-testid="stMetric"]{
    border:2px solid #4CAF50;
    border-radius:15px;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )

    st.title("Career Guide")

    st.write("""
Welcome!

Select your interest level for each skill.

The AI model will recommend the most suitable career.
""")

    st.info("Developed using Machine Learning")

# ----------------------------
# Load Model and Encoder
# ----------------------------
model = joblib.load("career_prediction_model.pkl")
role_encoder = joblib.load("role_encoder.pkl")
skills = list(model.feature_names_in_)
# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Career Recommendation System",
    layout="wide"
)

st.title(" Career Recommendation System")
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
    "Database Fundamentals",
    "Computer Architecture",
    "Distributed Computing Systems",
    "Cyber Security",
    "Networking",
    "Software Development",
    "Programming Skills",
    "Project Management",
    "Computer Forensics Fundamentals",
    "Technical Communication",
    "AI ML",
    "Software Engineering",
    "Business Analysis",
    "Communication skills",
    "Data Science",
    "Troubleshooting skills",
    "Graphics Designing"
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
