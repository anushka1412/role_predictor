import streamlit as st
import joblib
import pandas as pd





st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#0f172a);
}

/* Title */
h1{
    color:#38BDF8;
    text-align:center;
    font-size:52px;
    font-weight:bold;
}

/* Subtitle */
p{
    font-size:18px;
}

/* Main Container */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#111827,#1E3A8A);
}

/* Dropdown Box */
div[data-baseweb="select"] > div{
    background:#1E293B !important;
    border:2px solid #38BDF8 !important;
    border-radius:12px !important;
    color:white !important;
}

/* Dropdown Hover */
div[data-baseweb="select"]:hover > div{
    border:2px solid #22C55E !important;
}

/* Labels */
label{
    color:#E2E8F0 !important;
    font-size:17px !important;
    font-weight:600 !important;
}

/* Button */
div.stButton > button{

    width:100%;
    height:60px;

    background:linear-gradient(90deg,#2563EB,#06B6D4);

    color:white;

    border:none;

    border-radius:15px;

    font-size:22px;

    font-weight:bold;

    transition:0.3s;
}

/* Button Hover */

div.stButton > button:hover{

    background:linear-gradient(90deg,#9333EA,#EC4899);

    transform:scale(1.03);

    box-shadow:0px 0px 15px #9333EA;

}

/* Success Box */

div[data-testid="stAlert"]{

    background:linear-gradient(90deg,#16A34A,#22C55E);

    color:white;

    border-radius:15px;

    font-size:22px;

}

/* Metric Cards */

[data-testid="stMetric"]{

    background:#1E293B;

    border-radius:15px;

    padding:20px;

    border:2px solid #38BDF8;

}

hr{

border:1px solid #334155;

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

    st.balloons()

    st.markdown(
    f"""
    <div style="
        background: linear-gradient(90deg, #2563EB, #06B6D4);
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
    ">
        <h2 style="color:white;">🎯 Recommended Career</h2>
        <h1 style="color:white;">{role[0]}</h1>
    </div>
    """,
    unsafe_allow_html=True
    )

    <h2 style="color:white;">
    Recommended Career
    </h2>

    <h1 style="color:white;">
    {role[0]}
    </h1>

    </div>
    """, unsafe_allow_html=True)
