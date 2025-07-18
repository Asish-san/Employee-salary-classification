# Employee Salary Prediction App

This project provides an end-to-end solution for predicting employee salaries using machine learning. It includes:
- Data analysis and visualization in a Jupyter notebook
- Model training and comparison
- A Streamlit web app for interactive salary prediction and batch processing

## Features
- feature engineering
- Model comparison (Linear Regression, Random Forest, Gradient Boosting)
- Interactive prediction for individual employees
- Batch prediction for CSV uploads
- Real-time USD to INR conversion
- Market-standard salary capping for US and Indian markets

## File Structure
- `salary.csv`: Dataset used for training
- `salary_prediction.ipynb`: Jupyter notebook for data analysis and model training
- `best_model.pkl`: Saved trained model
- `assets/`: Contains logo and model performance images
- `app.py`: Streamlit app for salary prediction
- `README.md`: Project documentation

## How to Run Locally
1. **Clone the repository or download the files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install:
   ```bash
   pip install streamlit pandas scikit-learn matplotlib seaborn joblib
   ```
3. **Ensure the following files are present in the project directory**:
   - `app.py`
   - `best_model.pkl`
   - `assets/logo_salary_app.png`
   - `assets/model_performance.png`
   - `assets/best_model_info.txt`
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
5. **Open the app in your browser** ([click here](https://employee-salary-predictor-2025.streamlit.app/))

## How to Use
- **Individual Prediction**: Use the sidebar to enter employee details and select currency (USD/INR). Click "Predict Salary" to see the result.
- **Batch Prediction**: Upload a CSV file with employee data. Download the predictions as a CSV.

## Deployment
- The app is compatible with Streamlit Cloud. Ensure all assets and model files are uploaded to the cloud workspace.
- For best results, keep image and model files in the same directory as `app.py` or in the `assets/` folder.

## Customization
- UI colors, gradients, and logo can be changed in `app.py` CSS section.
- Salary ranges and market standards can be updated in the prediction logic.

## Troubleshooting
- **Image/File Errors**: Ensure all referenced files exist and paths are correct.
- **Model Errors**: Retrain the model in the notebook if the dataset changes.
- **Streamlit Cloud**: Upload all required files and assets.

## Author
Created by **Asish Kumar**
- [GitHub Profile](https://github.com/Asish-san)

---
Enjoy exploring and predicting employee salaries with a beautiful, interactive app!

---
© 2025 Asish Kumar. All rights reserved.
This project and its contents are protected under GitHub's copyright and anti-plagiarism system. Any unauthorized copying, distribution, or use may be detected and flagged by GitHub's automated tools. For more information, see [GitHub Copyright Policy](https://docs.github.com/en/site-policy/content-removal-policies/github-copyright-policy).
