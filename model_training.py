# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# --- Configuration ---
DATA_PATH = 'MultipleFiles/adult 3.csv'
MODEL_DIR = 'saved_models'
TARGET_COLUMN = 'income'

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Load Data ---
print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please check the path.")
    exit()

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# --- 2. Data Preprocessing ---
print("\nStarting data preprocessing...")

# Identify categorical and numerical features
# Exclude 'fnlwgt' as it's often considered a sampling weight and not a direct feature for individual prediction
numerical_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]

# Handle missing values represented by '?'
for col in df.columns:
    if df[col].dtype == 'object': # Only process object (string) columns
        df[col] = df[col].replace('?', pd.NA) # Replace '?' with pandas' NA for proper handling

# Impute missing categorical values with the mode
for col in categorical_features:
    if col in df.columns:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
        print(f"Imputed missing values in '{col}' with mode: '{mode_value}'")

# Drop rows with any remaining missing numerical values (shouldn't be any if data is clean)
df.dropna(inplace=True)
print(f"Dataset shape after handling missing values: {df.shape}")

# Separate features (X) and target (y)
X = df.drop(columns=[TARGET_COLUMN, 'fnlwgt']) # Drop fnlwgt as it's not a predictive feature
y = df[TARGET_COLUMN].apply(lambda x: 1 if x.strip() == '>50K' else 0) # Convert target to 0/1

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Target value counts:\n{y.value_counts()}")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # handle_unknown='ignore' for unseen categories in prediction

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any, though none expected here
)
print("Preprocessing pipeline created.")

# --- 3. Model Training ---
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

print("Building and training the Logistic Regression model pipeline...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42)) # liblinear is good for small datasets
])

model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate Model ---
print("\nEvaluating the model...")
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# --- 5. Model Persistence ---
print("\nSaving the trained model and preprocessor...")
joblib.dump(model_pipeline, os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'))
print(f"Model saved to {os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')}")

print("\nModel training script finished.")
