import urllib
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load, dump
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------
# Step 1: Load Dataset
# -------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                 "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    try:
        df = pd.read_csv(url, names=col_names)
    except urllib.error.URLError as e:
        st.error(f"Failed to load data. Error: {e}")
        return None

    # Replace 0 values with NaN in specific columns
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

    # Fill missing values with median
    df.fillna(df.median(), inplace=True)

    return df

data = load_data()
if data is None:
    st.stop()

# -------------------------
# Step 2: Exploratory Data Analysis
# -------------------------
st.title("Diabetes Prediction Web App")
st.write("### Exploratory Data Analysis (EDA)")

if st.checkbox("Show Dataset"):
    st.dataframe(data)

if st.checkbox("Show Summary Statistics"):
    st.write(data.describe())

if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

# -------------------------
# Step 3: Model Training
# -------------------------
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel="linear", probability=True),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
}


# Train and Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1-Score": f1}

# Find Best Model based on F1-Score
best_model_name = max(results, key=lambda x: results[x]["F1-Score"])
best_metrics = results[best_model_name]

st.write(f"### Best Model: *{best_model_name}*")
st.write(f"🔹 Accuracy: *{best_metrics['Accuracy']:.2f}*")
st.write(f"🔹 Precision: *{best_metrics['Precision']:.2f}*")
st.write(f"🔹 Recall: *{best_metrics['Recall']:.2f}*")
st.write(f"🔹 F1-Score: *{best_metrics['F1-Score']:.2f}*")

# -------------------------
# Step 4: Hyperparameter Tuning (For Best Model)
# -------------------------
best_model = models[best_model_name]

if best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
elif best_model_name == "Decision Tree":
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
elif best_model_name == "SVM":
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
elif best_model_name == "Logistic Regression":
    param_grid = {'C': [0.1, 1, 10]}
elif best_model_name == "AdaBoost":
    param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]}

# Run Grid Search
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get Optimized Model
optimized_model = grid_search.best_estimator_
st.write(f"Optimized {best_model_name} Model F1-Score: {grid_search.best_score_:.2f}")

# Save Best Model
dump(optimized_model, "diabetes_model.pkl")

# -------------------------
# Step 5: User Input for Prediction
# -------------------------
st.title("Make a Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])

if gender == "Female":
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
else:
    pregnancies = 0

glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Load Model and Predict
    model = load("diabetes_model.pkl")
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.error("The patient is likely to have diabetes.")
    else:
        st.success("The patient is unlikely to have diabetes.")
