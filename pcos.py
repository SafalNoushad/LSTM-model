import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer  
from imblearn.over_sampling import SMOTE

# 📌 Load Dataset
file_path = "pcos/PCOS.csv"
df = pd.read_csv(file_path)

# 📌 Standardize Column Names
df.columns = df.columns.str.strip()
print("🔍 Available Columns:", df.columns.tolist())  # Debugging Step

# 📌 Handle Incorrect Data (Replace invalid values with NaN)
df.replace(["#NAME?", "?", "-", "unknown", "a"], np.nan, inplace=True)

# 📌 Feature Selection (Ensure Columns Exist)
selected_features = [
    "Age (yrs)", "BMI", "Weight (Kg)", "Cycle length(days)", "Pregnant(Y/N)",
    "FSH(mIU/mL)", "LH(mIU/mL)", "TSH (mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)", 
    "Vit D3 (ng/mL)", "PRG(ng/mL)", "RBS(mg/dl)", "Weight gain(Y/N)", 
    "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", 
    "Fast food (Y/N)", "Reg.Exercise(Y/N)"
]

# ✅ Remove missing columns from selected features
df = df.dropna(axis=1, how="all")  # Drop columns that are fully empty
selected_features = [col for col in selected_features if col in df.columns]

# 📌 Create Feature Matrix & Target Variable
X = df[selected_features]
y = df["PCOS (Y/N)"].astype(int)  

# ✅ Handle NaN Values Using Imputer (Only for Numeric Columns)
imputer = SimpleImputer(strategy="mean")  
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 📌 Check for Remaining NaN (Debugging Step)
if X_imputed.isnull().sum().sum() > 0:
    print("⚠️ Warning: Missing values still exist after imputation!")

# 📌 Handle Class Imbalance Using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 📌 Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 📌 Predictions & Evaluation
y_pred = rf_model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
