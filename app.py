import streamlit as st
import pandas as pd
import joblib
import requests

# Custom CSS for gradients, fonts, and layout
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #f6d365 0%, #fdc094 25%, #fda085 50%, #a1c4fd 75%, #c2e9fb 100%);
        animation: gradientBG 12s ease infinite;
    }
    @keyframes gradientBG {
        0% {background: linear-gradient(120deg, #f6d365 0%, #fdc094 25%, #fda085 50%, #a1c4fd 75%, #c2e9fb 100%);}
        25% {background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);}
        50% {background: linear-gradient(120deg, #43e97b 0%, #38f9d7 100%);}
        75% {background: linear-gradient(120deg, #fc00ff 0%, #00dbde 100%);}
        100% {background: linear-gradient(120deg, #f6d365 0%, #fdc094 25%, #fda085 50%, #a1c4fd 75%, #c2e9fb 100%);}
    }
    .footer {
        text-align: center;
        font-size: 22px;
        margin-top: 40px;
        background: linear-gradient(90deg, #fc00ff, #43e97b, #f093fb, #f5576c);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientText 3s ease-in-out infinite;
    }
    @keyframes gradientText {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .gradient-box {
        background: linear-gradient(135deg, #fc00ff 0%, #43e97b 100%);
        border-radius: 24px;
        padding: 32px 24px;
        margin-bottom: 24px;
        opacity: 0.75;
        box-shadow: 0 4px 24px rgba(252,0,255,0.12);
    }
    .gradient-input {
        background: linear-gradient(135deg, #f8ffae 0%, #fc00ff 100%);
        border-radius: 18px;
        padding: 18px 12px;
        opacity: 0.75;
        box-shadow: 0 2px 12px rgba(252,0,255,0.10);
        margin-bottom: 18px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2193b0 0%, #6dd5ed 100%);
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(252,0,255,0.18);
        transition: transform 0.2s;
        color: #222;
    }
        background: linear-gradient(90deg, #fc00ff 0%, #f6d365 50%, #43c6ac 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(252,0,255,0.18);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.10);
        background: linear-gradient(90deg, #f5576c 0%, #f093fb 50%, #43e97b 100%);
        color: #24292f;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #43e97b 0%, #f8ffae 50%, #fc00ff 100%);
        color: #24292f;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 2px 12px rgba(67,233,123,0.18);
    }
    .logo {
        height: 80px;
        filter: drop-shadow(0 0 12px #fc00ff);
        animation: bounce 1.5s infinite;
        border-radius: 20px;
        border: 3px solid #43e97b;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #fc00ff 100%);
        box-shadow: 0 2px 12px rgba(252,0,255,0.18);
    }
    @keyframes bounce {
        0%, 100% {transform: translateY(0);}
        50% {transform: translateY(-12px);}
    }
    </style>
""", unsafe_allow_html=True)



# Set Streamlit page config
st.set_page_config(
    page_title='Employee Salary Prediction',
    page_icon='assets/logo.png',
    layout='centered'
)

st.markdown("<div style='display:flex; align-items:center;'>", unsafe_allow_html=True)
st.image('logo.png', width=40)
st.markdown("<span style='font-size:12px; color:#888; margin-left:10px;'>Employee Salary Prediction</span></div>", unsafe_allow_html=True)
st.title('💼 Employee Salary Prediction App')
st.markdown('<h3 style="color:#43c6ac;">Predict salary, compare models, and see real-time USD to INR conversion!</h3>', unsafe_allow_html=True)

# Add custom CSS
hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

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
# Show model performance
st.markdown("<div style='display:flex; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
st.image('assets/model_performance.png', caption='Model Performance Comparison', width=500)
st.markdown("</div>", unsafe_allow_html=True)

# Predict button
if st.button('🚀 Predict Salary'):
    # payout mapping for jobs
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
    def ai_job_payout(job, market):
        # Example: Use experience and education to estimate
        edu_map = {
            'HS-grad': 1,
            'Assoc': 2,
            'Some-college': 3,
            'Bachelors': 4,
            'Masters': 5,
            'PhD': 6
        }
        edu_num = edu_map.get(education, 3)  # Default to 'Some-college' if not found
        base_us = 35000 + (experience * 1000) + (edu_num * 2000)
        base_in = 350000 + (experience * 20000) + (edu_num * 40000)
        if market == 'USD':
            return (base_us, base_us + 20000)
        else:
            return (base_in, base_in + 200000)

    # Get selected job title
    job = job_title
    salary_pred = model.predict(input_df)[0]
    # Get real-time USD to INR rate
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD')
        usd_to_inr = response.json()['rates']['INR']
    except Exception:
        usd_to_inr = 83.0  # fallback value
    # US market prediction
    payout_range_us = us_market_payout.get(job, ai_job_payout(job, 'USD'))
    salary_pred_us = min(max(salary_pred, payout_range_us[0]), payout_range_us[1])
    # Indian market prediction
    payout_range_in = in_market_payout.get(job, ai_job_payout(job, 'INR'))
    salary_pred_in = min(max(salary_pred, payout_range_in[0]), payout_range_in[1])
    # Convert between currencies
    salary_us_to_inr = salary_pred_us * usd_to_inr
    salary_in_to_us = salary_pred_in / usd_to_inr
    if currency.startswith('USD'):
        st.success(f'💰 Predicted Salary (US Market): ${salary_pred_us:,.2f} USD')
        st.info(f'Equivalent in INR: ₹{salary_us_to_inr:,.2f}')
    else:
        st.success(f'💰 Predicted Salary (Indian Market): ₹{salary_pred_in:,.2f} INR')
        st.info(f'Equivalent in USD: ${salary_in_to_us:,.2f}')
    st.markdown(f"<div style='text-align:center;'><span style='font-size:16px; color:#43e97b;'>Real-time USD to INR Rate: <b>₹{usd_to_inr:,.2f}</b></span></div>", unsafe_allow_html=True)
    st.balloons()

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
    if 'Gender' in batch_data.columns:
        batch_data['Gender'] = batch_data['Gender'].map({'Male': 1, 'Female': 0})
    # Align columns to match model training features
    expected_cols = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
    for col in expected_cols:
        if col not in batch_data.columns:
            batch_data[col] = 0  # Default value for missing columns
    batch_data = batch_data[expected_cols]
    batch_preds = model.predict(batch_data)
    # Real-time conversion for batch
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD')
        usd_to_inr = response.json()['rates']['INR']
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
        def ai_job_payout(job, edu, exp, market):
            edu_map = {
                'HS-grad': 1,
                'Assoc': 2,
                'Some-college': 3,
                'Bachelors': 4,
                'Masters': 5,
                'PhD': 6
            }
            edu_num = edu_map.get(edu, 3)
            base_us = 35000 + (exp * 1000) + (edu_num * 2000)
            base_in = 350000 + (exp * 20000) + (edu_num * 40000)
            if market == 'USD':
                return (base_us, base_us + 20000)
            else:
                return (base_in, base_in + 200000)

        pred_usd = []
        pred_inr = []
        # For each row, apply market capping and AI logic if needed
        for idx, row in batch_data.iterrows():
            # Get original job/edu/exp from uploaded data if available
            job = row.get('Job Title', None)
            edu = row.get('Education Level', None)
            exp = row.get('Years of Experience', 0)
            # Try to map back encoded values to string for job/edu
            if isinstance(job, int) and 0 <= job < len(le_job.classes_):
                job_str = le_job.classes_[job]
            elif isinstance(job, str):
                job_str = job
            else:
                job_str = None
            if isinstance(edu, int) and 0 <= edu < len(le_edu.classes_):
                edu_str = le_edu.classes_[edu]
            elif isinstance(edu, str):
                edu_str = edu
            else:
                edu_str = None
            # Get model prediction
            pred = batch_preds[idx]
            # US market payout
            payout_us = us_market_payout.get(job_str, ai_job_payout(job_str, edu_str, exp, 'USD'))
            pred_us = min(max(pred, payout_us[0]), payout_us[1])
            # IN market payout
            payout_in = in_market_payout.get(job_str, ai_job_payout(job_str, edu_str, exp, 'INR'))
            pred_in = min(max(pred, payout_in[0]), payout_in[1])
            # Convert both ways
            pred_us_to_inr = pred_us * usd_to_inr
            pred_in_to_us = pred_in / usd_to_inr
            pred_usd.append(pred_us)
            pred_inr.append(pred_in)
            # Store both conversions for each row
            if 'PredictedSalaryUSD' not in batch_data.columns:
                batch_data['PredictedSalaryUSD'] = 0
            if 'PredictedSalaryINR' not in batch_data.columns:
                batch_data['PredictedSalaryINR'] = 0
            if 'PredictedSalaryUSD_from_INR' not in batch_data.columns:
                batch_data['PredictedSalaryUSD_from_INR'] = 0
            if 'PredictedSalaryINR_from_USD' not in batch_data.columns:
                batch_data['PredictedSalaryINR_from_USD'] = 0
            batch_data.at[idx, 'PredictedSalaryUSD'] = pred_us
            batch_data.at[idx, 'PredictedSalaryINR'] = pred_in
            batch_data.at[idx, 'PredictedSalaryUSD_from_INR'] = pred_in_to_us
            batch_data.at[idx, 'PredictedSalaryINR_from_USD'] = pred_us_to_inr
    st.write('✅ Predictions:')
    st.dataframe(batch_data.head(), use_container_width=True)
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button('⬇️ Download Predictions CSV', csv, file_name='predicted_salaries.csv', mime='text/csv')
except Exception as e:
        st.error(f"Batch prediction failed: {e}")


st.markdown("""
    <div style='text-align:center;'>
        <span style='font-size:40px;'>🎉✨🚀💼</span>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='display:flex; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
st.image('logoasish.png', width=40)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    <span>Created by <b>Asish Kumar</b> |
    <br>
    <span style='color:#43c6ac;'>Streamlit Web App</span> &nbsp; <span style='font-size:24px;'>🌐</span>
    <br>
    <span style='font-size:14px;'><b>© 2025 Asish Kumar. All rights reserved.</b></span>
    </span>
</div>
""", unsafe_allow_html=True)
