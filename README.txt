Files generated:
- model.pkl        : trained RandomForest model (joblib)
- metadata.json    : feature names + label mapping
- app.py           : Streamlit UI (template) to load the model + metadata
- README.txt       : this file

Feature names detected: ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
Label mapping: {'0': '0', '1': '1'}

To run locally:
1) pip install streamlit pandas numpy scikit-learn joblib
2) streamlit run app.py

If you plan to deploy on Streamlit Cloud, include a requirements.txt listing at minimum:
streamlit
pandas
numpy
scikit-learn
joblib