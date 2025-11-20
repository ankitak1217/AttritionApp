import streamlit as st
import joblib, json, pandas as pd
from pathlib import Path

st.set_page_config(page_title='Attrition RF - Demo', layout='centered')
st.title('Attrition Random Forest - Prediction UI')

MODEL_PATH = Path('model.pkl')
META_PATH = Path('metadata.json')
if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("model.pkl or metadata.json missing; place them in the app folder.")
    st.stop()

try:
    model = joblib.load(str(MODEL_PATH))
except Exception as e:
    st.error(f"Failed to load model.pkl: {e}")
    st.stop()

with open(str(META_PATH), 'r') as mf:
    meta = json.load(mf)

features = meta.get('feature_names', ["Age", "Attrition", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"])
label_map = meta.get('label_mapping', {"0": "0", "1": "1"})

st.sidebar.header('Input mode')
mode = st.sidebar.radio('Mode', ['Manual single', 'Upload CSV'])

if mode == 'Manual single':
    st.header('Manual input - single observation')
    inputs = {}
    for feat in features:
        # you can adjust min/max/step for known ranges
        inputs[feat] = st.number_input(feat, value=0.0)
    if st.button('Predict'):
        X = pd.DataFrame([inputs], columns=features)
        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        except Exception as e:
            st.error(f'Prediction error: {e}')
        else:
            out_label = label_map.get(str(pred), str(pred)) if label_map else str(pred)
            st.success(f'Predicted class: {out_label}')
            if proba is not None:
                st.write('Probabilities:')
                for i,p in enumerate(proba):
                    cname = label_map.get(str(i), f'class_{i}') if label_map else f'class_{i}'
                    st.write(f'{cname}: {p:.4f}')

else:
    st.header('Batch prediction - upload CSV')
    uploaded = st.file_uploader('Upload CSV with feature columns', type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f'Missing columns in uploaded CSV: {missing}')
        else:
            X = df[features]
            try:
                preds = model.predict(X)
                proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            except Exception as e:
                st.error(f'Prediction error: {e}')
            else:
                out = X.copy()
                out['prediction'] = [label_map.get(str(p), str(p)) for p in preds]
                if proba is not None:
                    for i in range(proba.shape[1]):
                        out[f'prob_class_{i}'] = proba[:, i]
                st.dataframe(out.head())
                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions CSV', csv, 'predictions.csv', 'text/csv')

st.caption('This app loads model.pkl and metadata.json from the app folder.')