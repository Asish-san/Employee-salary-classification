import streamlit as st
import pandas as pd
import joblib
import requests

# Custom CSS for gradients, fonts, and layout
st.markdown("""
    <style>
    body {background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);}
    .main {background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);}
    .stButton>button {background-color: #43c6ac; color: white; font-weight: bold;}
    .stDownloadButton>button {background-color: #43c6ac; color: white;}
    .css-1v0mbdj {background: linear-gradient(90deg, #43c6ac 0%, #f8ffae 100%);}
    .footer {text-align: center; font-size: 18px; margin-top: 40px;}
    .github-link {color: #24292f; font-weight: bold;}
    .logo {height: 60px;}
    </style>
""", unsafe_allow_html=True)

# App logo and title
st.image("assets/logo.png", width=80, caption="Employee Salary Classification")
st.title('💼 Employee Salary Classification App')
st.markdown('<h3 style="color:#43c6ac;">Predict whether a person earn >50K or <50k based on input features!</h3>', unsafe_allow_html=True)

# Load trained model and model comparison image
model = joblib.load('best_model.pkl')
model_comparison_img_url = "assets/output.png"  # Example image
# Sidebar inputs
st.sidebar.header('👤 Input Employee Details')
age = st.sidebar.slider('Age', 18, 65, 30)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
education = st.sidebar.selectbox('Education Level', [
    'Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc', 'Some-college'
])
job_title = st.sidebar.selectbox('Job Title', [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
    'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
    'Protective-serv', 'Armed-Forces'
])
experience = st.sidebar.slider('Years of Experience', 0, 40, 5)

# Input DataFrame (must match training features)
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [1 if gender == 'Male' else 0],
    'Education Level': [education],
    'Job Title': [job_title],
    'Years of Experience': [experience]
})

st.write('### 🔎 Input Data')
st.dataframe(input_df, use_container_width=True)

# Model comparison image and R2 score
st.markdown('---')
st.image(model_comparison_img_url, caption='Model Comparison', use_column_width=True)
if 'best_model_name' in globals() and 'best_r2' in globals():
    st.markdown(f"<h4>🏆 <span style='color:#43c6ac'>Best Model:</span> {best_model_name} | <span style='color:#43c6ac'>R² Score:</span> {best_r2:.4f}</h4>", unsafe_allow_html=True)

# Predict button with animation
if st.button('🚀 Predict Salary'):
    salary_pred = model.predict(input_df)[0]
    # Real-time USD to INR conversion
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD')
        usd_to_inr = response.json()['rates']['INR']
        salary_inr = salary_pred * usd_to_inr
        st.success(f'💰 Predicted Salary: ${salary_pred:,.2f} USD  |  ₹{salary_inr:,.2f} INR')
        st.balloons()
    except Exception:
        st.success(f'💰 Predicted Salary: ${salary_pred:,.2f} USD')
        st.warning('⚠️ Could not fetch real-time USD to INR conversion.')

# Batch prediction
st.markdown('---')
st.markdown('#### 📂 Batch Prediction')
uploaded_file = st.file_uploader('Upload a CSV file for batch prediction', type='csv')
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write('Uploaded data preview:', batch_data.head())
    batch_preds = model.predict(batch_data)
    # Real-time conversion for batch
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD')
        usd_to_inr = response.json()['rates']['INR']
        batch_data['PredictedSalaryUSD'] = batch_preds
        batch_data['PredictedSalaryINR'] = batch_preds * usd_to_inr
    except Exception:
        batch_data['PredictedSalaryUSD'] = batch_preds
        batch_data['PredictedSalaryINR'] = 'N/A'
    st.write('✅ Predictions:')
    st.dataframe(batch_data.head(), use_container_width=True)
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button('⬇️ Download Predictions CSV', csv, file_name='predicted_salaries.csv', mime='text/csv')

# Animations and emojis
st.markdown("""
    <div style='text-align:center;'>
        <span style='font-size:40px;'>🎉✨🚀💼</span>
    </div>
""", unsafe_allow_html=True)

# Footer with author info and logo
st.markdown("""
    <hr>
    <div class='footer'>
        <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' class='logo'>
        <br>
        <span>Created by <b>Asish Kumar</b> | 
        <a class='github-link' href='https://github.com/Asish-san' target='_blank'>Follow me on GitHub</a>
        <br>
        <span style='color:#43c6ac;'>Streamlit Web App</span> &nbsp; <span style='font-size:24px;'>🌐</span>
        </span>
    </div>
""", unsafe_allow_html=True)
