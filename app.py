import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Career Predictor", 
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.main-title{
text-align:center;
font-size:45px;
font-weight:bold;
color:white;
margin-bottom:10px;
}

.sub-title{
text-align:center;
font-size:18px;
color:#dddddd;
margin-bottom:30px;
}

.prediction-box{
background:#00c853;
padding:20px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:white;
box-shadow:0px 0px 20px #00ff88;
}

.stButton>button{
width:100%;
background:#ff9800;
color:white;
font-size:20px;
border-radius:10px;
height:55px;
border:none;
}

.stButton>button:hover{
background:#ff5722;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("career_prediction_model.pkl")
encoder = joblib.load("role_encoder.pkl")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("dataset9000.csv")

feature_columns = df.columns[:-1]

# -------------------------------
# Title
# -------------------------------
st.markdown("<div class='main-title'> AI Career Prediction System</div>", unsafe_allow_html=True)

st.markdown("<div class='sub-title'>Predict the best career based on your skills</div>", unsafe_allow_html=True)

# -------------------------------
# Layout
# -------------------------------
left,right = st.columns(2)

inputs = []

with left:

    st.subheader(" Enter Your Skills")

    for feature in feature_columns[:len(feature_columns)//2]:
        value = st.slider(feature,0,100,50)
        inputs.append(value)

with right:

    st.subheader("📊 Continue")

    for feature in feature_columns[len(feature_columns)//2:]:
        value = st.slider(feature,0,100,50)
        inputs.append(value)

# -------------------------------
# Prediction
# -------------------------------
if st.button(" Predict Career"):

    data = np.array(inputs).reshape(1,-1)

    prediction = model.predict(data)[0]

    career = encoder.inverse_transform([prediction])[0]

    try:
        confidence = model.predict_proba(data).max()*100
    except:
        confidence = None

    st.balloons()

    st.markdown(
        f"""
        <div class='prediction-box'>
        Predicted Career <br><br>
         {career}
        </div>
        """,
        unsafe_allow_html=True
    )

    if confidence:
        st.progress(int(confidence))
        st.success(f"Confidence : {confidence:.2f}%")

    st.info("Keep improving your skills to become even better in this career!")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
"""
<center>
Made with ❤️ using Streamlit | AI Career Prediction
</center>
""",
unsafe_allow_html=True
)
