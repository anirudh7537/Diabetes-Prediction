import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def main():
    # 1. LOAD THE DATA
    # Ensure 'diabetes.csv' is in the same folder as this Python script
    print("Loading dataset...")
    try:
      diabetes_dataset = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found. Please ensure it is in the same directory.")
        return

    # 2. EXPLORATORY DATA ANALYSIS (Optional but good for verification)
    print("\n--- Dataset Info ---")
    print("Shape of dataset:", diabetes_dataset.shape)
    print("\nFirst 5 rows:\n", diabetes_dataset.head())
    print("\nOutcome distribution:\n", diabetes_dataset['Outcome'].value_counts())
    print("\nMean values grouped by Outcome:\n", diabetes_dataset.groupby('Outcome').mean())

    # 3. SEPARATE FEATURES AND LABELS
    X = diabetes_dataset.drop(columns='Outcome')
    Y = diabetes_dataset['Outcome']

    # 4. DATA STANDARDIZATION
    # Standardizing data helps the SVM model perform better
    scaler = StandardScaler()
    scaler.fit(X)
    X_standardized = scaler.transform(X)
    
    # Update X with the standardized data
    X = X_standardized

    # 5. TRAIN/TEST SPLIT
    # Splitting 80% of data for training, 20% for testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    print(f"\nData split complete: Training size = {X_train.shape[0]}, Test size = {X_test.shape[0]}")

    # 6. MODEL TRAINING
    print("\nTraining the Support Vector Machine (SVM) model...")
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    # 7. MODEL EVALUATION
    # Accuracy on training data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print(f"Accuracy on training data: {training_data_accuracy * 100:.2f}%")

    # Accuracy on test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print(f"Accuracy on test data: {test_data_accuracy * 100:.2f}%")

  # 8. PREDICTIVE SYSTEM FOR A NEW INSTANCE
    print("\n--- Running Prediction System ---")
    input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

    # Convert to a DataFrame with explicit feature names to prevent the UserWarning
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data_df = pd.DataFrame([input_data], columns=feature_names)

    # Standardize the input data using the SAME scaler
    std_data = scaler.transform(input_data_df)
    print("Standardized Input Data:", std_data)

    # Make the prediction
    prediction = classifier.predict(std_data)
    
    if prediction[0] == 0:
        print('\nResult: The person is NOT diabetic.')
    else:
        print('\nResult: The person IS diabetic.')

if __name__ == "__main__":
    main()