# Heart-Disease-Prediction-Machine-Learning

# Heart Disease Prediction using Machine Learning

I built this to predict whether a patient is likely to have heart disease from basic clinical data. I trained five classification models, compared them, and used the Logistic Regression model in a small Streamlit app.

Live app: [Streamlit link]

## Dataset

The data is heart.csv from Kaggle ([Kaggle link]). It has 918 patients and 11 features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak and ST_Slope. The target is HeartDisease (1 for heart disease, 0 for none).

## Approach

Cholesterol and RestingBP had 0 values that cannot be real, so I treated them as missing. I replaced them with NaN and filled them with the median.

I split the data first, then fitted the scaler and encoder on the training set only, so nothing from the test set leaked in. I trained and compared these models:

- Logistic Regression
- KNN
- Naive Bayes
- Decision Tree
- SVM
- [Random Forest / XGBoost, if used]

I looked mainly at recall, because missing a sick patient is worse than a false alarm. I also checked precision, F1 and ROC-AUC.

## Results

| Model | Accuracy | F1 |

| Logistic Regression | 0.8641 || 0.8792 |
| KNN | 0.8533 | | 0.8708 |
| Naive Bayes | 0.8478 | | 0.8614 |
| Decision Tree | 0.8152 || 0.8317 | 
| SVM | 0.8587 || 0.8762 |

Best model: Logistic Regression (accuracy 0.8641, F1 0.8792).

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app needs `LR_heart.pkl`, `scaler.pkl` and `columns.pkl` in the same folder.

## What I learned

My first version scaled the whole dataset before splitting and then converted it to int. That wrecked features like Cholesterol and leaked test data into training. After I fixed it, my results changed. I now split first and check my data after every step. I also learned that a 0 in a medical column is not always a real value.

## Limitations

The dataset is small, with only 918 patients. The model is not clinically validated. This is an educational project and not medical advice.

## Contact

- GitHub: [github.com/ShahidMullaji](https://github.com/ShahidMullaji)
- LinkedIn: [linkedin.com/in/mohammad-shahid-7bba45290](https://linkedin.com/in/mohammad-shahid-7bba45290)
