import streamlit as st
import pandas as pd
import joblib

# --- Saved files load karo ---
model = joblib.load('LR_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')   # training X ke columns (same order me)

NUM_COLS = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

st.title('Heart Disease Prediction by Shahid')
st.markdown('Please provide the following details')

age = st.slider('Age', 18, 100, 40)
sex = st.selectbox('Sex', ['M', 'F'],
                   format_func=lambda v: 'Male' if v == 'M' else 'Female')
chest_pain = st.selectbox('Chest pain type', ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.slider('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.number_input('Cholesterol', 100, 600, 200)
fasting_bs = st.selectbox('Fasting blood sugar > 120 mg/dl', [0, 1],
                          format_func=lambda v: 'Yes' if v == 1 else 'No')
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Max Heart Rate', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Angina', ['Y', 'N'],
                               format_func=lambda v: 'Yes' if v == 'Y' else 'No')
oldpeak = st.slider('Oldpeak', 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if st.button('Predict'):
    # Keys training columns ke naam se EXACT match honi chahiye.
    # get_dummies(drop_first=True) me baseline category (F, ASY, LVH, N, Down)
    # ka column hota hi nahi, wo neeche reindex me apne aap hat jata hai.
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        f'Sex_{sex}': 1,
        f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Missing columns 0 se bharo, extra hatao, order training jaisa karo
    input_df = input_df.reindex(columns=columns, fill_value=0).astype(float)

    # Scaler jin columns pe fit hua tha, sirf unhi ko scale karo
    scale_cols = list(getattr(scaler, 'feature_names_in_', NUM_COLS))
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    st.subheader(f'Estimated risk: {probability:.0%}')
    if prediction == 1:
        st.error('High Risk of Heart Disease')
    else:
        st.success('Low Risk of Heart Disease')

    st.caption('Educational project. Ye medical diagnosis nahi hai.')