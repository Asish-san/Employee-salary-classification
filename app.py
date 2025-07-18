import streamlit as st
import pandas as pd
import joblib
import requests

# Custom CSS for gradients, fonts, and layout
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);
        animation: gradientBG 10s ease infinite;
    }
    @keyframes gradientBG {
        0% {background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);}
        50% {background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);}
        100% {background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);}
    }
    .main {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        border-radius: 22px;
        box-shadow: 0 6px 32px rgba(67,233,123,0.18);
        padding: 20px;
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .stButton>button {
        background: linear-gradient(90deg, #fc00ff 0%, #00dbde 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(252,0,255,0.18);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.10);
        background: linear-gradient(90deg, #f5576c 0%, #f093fb 100%);
        color: #24292f;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: #24292f;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 2px 12px rgba(67,233,123,0.18);
    }
    .footer {
        text-align: center;
        font-size: 22px;
        margin-top: 40px;
        color: #fc00ff;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {color: #fc00ff;}
        50% {color: #43e97b;}
        100% {color: #fc00ff;}
    }
    .github-link {
        display: none !important;
    }
    .logo {
        height: 80px;
        filter: drop-shadow(0 0 12px #fc00ff);
        animation: bounce 1.5s infinite;
        border-radius: 20px;
        border: 3px solid #43e97b;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 2px 12px rgba(252,0,255,0.18);
    }
    @keyframes bounce {
        0%, 100% {transform: translateY(0);}
        50% {transform: translateY(-12px);}
    }
    </style>
""", unsafe_allow_html=True)



# Set Streamlit page config with custom logo
st.set_page_config(
    page_title='Employee Salary Prediction',
    page_icon='logo.png',
    layout='centered'
)

st.markdown("<div style='display:flex; align-items:center;'>", unsafe_allow_html=True)
st.image('logo.png', width=40)
st.markdown("<span style='font-size:12px; color:#888; margin-left:10px;'>Employee Salary Prediction</span></div>", unsafe_allow_html=True)
st.title('💼 Employee Salary Prediction App')
st.markdown('<h3 style="color:#43c6ac;">Predict individual and batch salary prediction along with statistical analysis!</h3>', unsafe_allow_html=True)


# Load trained model
model = joblib.load('best_model.pkl')

# Read best model info from file
with open('assets/best_model_info.txt', 'r') as f:
    best_model_info = f.read().strip().splitlines()
best_model_name = best_model_info[0].split(': ')[1]
best_r2 = float(best_model_info[1].split(': ')[1])

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
# Currency selection
currency = st.sidebar.selectbox('Select Currency', ['USD (US Market)', 'INR (Indian Market)'])


# Input DataFrame (must match training features)
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [1 if gender == 'Male' else 0],
    'Education Level': [education],
    'Job Title': [job_title],
    'Years of Experience': [experience]
})

# Encode categorical features to match model training
from sklearn.preprocessing import LabelEncoder
le_edu = LabelEncoder()
le_job = LabelEncoder()
# Fit encoders on all possible values (should match training)
le_edu.fit([
    'Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc', 'Some-college'
])
le_job.fit([
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
    'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
    'Protective-serv', 'Armed-Forces'
])
input_df['Education Level'] = le_edu.transform(input_df['Education Level'])
input_df['Job Title'] = le_job.transform(input_df['Job Title'])

st.write('### 🔎 Input Data')
st.dataframe(input_df, use_container_width=True)


# Model R2 score display
st.markdown('---')
st.markdown(f"<h4>🏆 <span style='color:#43c6ac'>Best Model:</span> {best_model_name} | <span style='color:#43c6ac'>R² Score:</span> {best_r2:.4f}</h4>", unsafe_allow_html=True)
# Show model performance PNG centered below the score
st.markdown("<div style='display:flex; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
st.image('assets/model_performance.png', caption='Model Performance Comparison', width=500)
st.markdown("</div>", unsafe_allow_html=True)

# Predict button with animation
if st.button('🚀 Predict Salary'):
    # AI-based market standard payout mapping for jobs
    us_market_payout = {
        'Tech-support': (35000, 65000),
        'Craft-repair': (40000, 70000),
        'Other-service': (30000, 55000),
        'Sales': (40000, 90000),
        'Exec-managerial': (80000, 120000),
        'Prof-specialty': (70000, 110000),
        'Handlers-cleaners': (25000, 40000),
        'Machine-op-inspct': (35000, 60000),
        'Adm-clerical': (35000, 55000),
        'Farming-fishing': (25000, 45000),
        'Transport-moving': (35000, 60000),
        'Priv-house-serv': (25000, 40000),
        'Protective-serv': (40000, 70000),
        'Armed-Forces': (40000, 90000)
    }
    in_market_payout = {
        'Tech-support': (350000, 900000),
        'Craft-repair': (400000, 950000),
        'Other-service': (300000, 700000),
        'Sales': (400000, 1200000),
        'Exec-managerial': (1200000, 5000000),
        'Prof-specialty': (1000000, 4000000),
        'Handlers-cleaners': (250000, 600000),
        'Machine-op-inspct': (350000, 900000),
        'Adm-clerical': (350000, 800000),
        'Farming-fishing': (250000, 600000),
        'Transport-moving': (350000, 900000),
        'Priv-house-serv': (250000, 600000),
        'Protective-serv': (400000, 950000),
        'Armed-Forces': (400000, 1200000)
    }

    # Use AI logic for jobs not in mapping
    def ai_market_payout(job, experience, education, market):
        # Example: Use experience and education to estimate
        base_us = 35000 + (experience * 1000) + (education * 2000)
        base_in = 350000 + (experience * 20000) + (education * 40000)
        if market == 'USD':
            return (base_us, base_us + 20000)
        else:
            return (base_in, base_in + 200000)

    # Get selected job title
    job = job_title
    # Map education to its index for ai_job_payout
    edu_list = ['Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc', 'Some-college']
    edu_num = edu_list.index(education) if education in edu_list else 3
    exp_num = experience
    if currency.startswith('USD'):
        payout_range = us_market_payout.get(job, ai_market_payout(job, exp_num, edu_num, 'USD'))
        salary_pred = model.predict(input_df)[0]
        # Scale prediction to market range
        salary_pred_us = min(max(salary_pred, payout_range[0]), payout_range[1])
        st.success(f'💰 Predicted Salary (US Market): ${salary_pred_us:,.2f} USD')
        salary_pred_in = None
    else:
        payout_range = in_market_payout.get(job, ai_market_payout(job, exp_num, edu_num, 'INR'))
        salary_pred = model.predict(input_df)[0]
        # Scale prediction to market range
        salary_pred_in = min(max(salary_pred, payout_range[0]), payout_range[1])
        st.success(f'💰 Predicted Salary (Indian Market): ₹{salary_pred_in:,.2f} INR')
    # Show salary statistics for the predicted salary
    import numpy as np
    if currency.startswith('USD'):
        stats = {
            'Min': payout_range[0],
            'Max': payout_range[1],
            'Mean': np.mean(payout_range),
            'Median': np.median(payout_range)
        }
        st.markdown(f"<div style='text-align:center;'><span style='font-size:16px; color:#43e97b;'>US Market Salary Stats:<br>Min: ${stats['Min']:,.2f} | Max: ${stats['Max']:,.2f} | Mean: ${stats['Mean']:,.2f} | Median: ${stats['Median']:,.2f}</span></div>", unsafe_allow_html=True)
    else:
        stats = {
            'Min': payout_range[0],
            'Max': payout_range[1],
            'Mean': np.mean(payout_range),
            'Median': np.median(payout_range)
        }
        st.markdown(f"<div style='text-align:center;'><span style='font-size:16px; color:#43e97b;'>Indian Market Salary Stats:<br>Min: ₹{stats['Min']:,.2f} | Max: ₹{stats['Max']:,.2f} | Mean: ₹{stats['Mean']:,.2f} | Median: ₹{stats['Median']:,.2f}</span></div>", unsafe_allow_html=True)

# Batch prediction
st.markdown('---')
st.markdown('#### 📂 Batch Prediction')
uploaded_file = st.file_uploader('Upload a CSV file for batch prediction', type='csv')
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write('Uploaded data preview:', batch_data.head())
    # Encode categorical columns to match model training
    edu_map = {v: i for i, v in enumerate(le_edu.classes_)}
    job_map = {v: i for i, v in enumerate(le_job.classes_)}
    if 'Education Level' in batch_data.columns:
        batch_data['Education Level'] = batch_data['Education Level'].map(edu_map).fillna(-1).astype(int)
    if 'Job Title' in batch_data.columns:
        batch_data['Job Title'] = batch_data['Job Title'].map(job_map).fillna(-1).astype(int)
    # Select only model input features for prediction
    feature_cols = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
    batch_features = batch_data[feature_cols]
    # Salary statistics for batch predictions
    batch_data['PredictedSalaryUSD'] = model.predict(batch_features)
    batch_stats = {
        'Min': batch_data['PredictedSalaryUSD'].min(),
        'Max': batch_data['PredictedSalaryUSD'].max(),
        'Mean': batch_data['PredictedSalaryUSD'].mean(),
        'Median': batch_data['PredictedSalaryUSD'].median()
    }
    st.write('✅ Predictions:')
    st.dataframe(batch_data.head(), use_container_width=True)
    st.markdown(f"<div style='text-align:center;'><span style='font-size:16px; color:#43e97b;'>Batch Salary Stats:<br>Min: ${batch_stats['Min']:,.2f} | Max: ${batch_stats['Max']:,.2f} | Mean: ${batch_stats['Mean']:,.2f} | Median: ${batch_stats['Median']:,.2f}</span></div>", unsafe_allow_html=True)
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button('⬇️ Download Predictions CSV', csv, file_name='predicted_salaries.csv', mime='text/csv')

# Animations and emojis
st.markdown("""
    <div style='text-align:center;'>
        <span style='font-size:40px;'>🎉✨🚀💼</span>
    </div>
""", unsafe_allow_html=True)

# Footer with author info and logo
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='display:flex; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
st.image('logo.png', width=40)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    <span>Created by <b>Asish Rout</b>
    <br>
    <span style='color:#43c6ac;'>Streamlit Web App</span> &nbsp; <span style='font-size:24px;'>🌐</span>
    </span>
</div>
<div style='text-align:center; font-size:14px; color:#888; margin-top:10px;'> © 2025 Asish Rout, All rights reserved.</div>
""", unsafe_allow_html=True)
