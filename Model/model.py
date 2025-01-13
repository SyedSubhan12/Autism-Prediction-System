import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib

# Function to preprocess the dataset
def preprocess_data(file_path, target_column):
    """
    Preprocess the dataset by reading it, handling missing values, and splitting features and target.
    :param file_path: Path to the dataset file (CSV format).
    :param target_column: Name of the target column.
    :return: features (X) and target (y) as separate DataFrames.
    """
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)

    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in the dataset!")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y

# Function to train the model
def train_model_pipeline(train_file, target_column):
    """
    Train a Random Forest model using the training dataset and save the pipeline to a file.
    :param train_file: Path to the training dataset file (CSV format).
    :param target_column: Name of the target column.
    """
    # Preprocess the training data
    X, y = preprocess_data(train_file, target_column)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Define the model and hyperparameter grid
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Validate the model
    y_val_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print("Validation Accuracy:", accuracy)
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

    # Save the trained model to a file
    joblib.dump(best_model, 'model_pipeline.pkl')
    print("Model pipeline saved to 'model_pipeline.pkl'.")

# Function to evaluate the model
def evaluate_model_pipeline(test_file):
    """
    Load the saved model pipeline and evaluate it using the test dataset.
    :param test_file: Path to the test dataset file (CSV format).
    """
    # Load the test data
    test_data = pd.read_csv(test_file)
    test_data.dropna(inplace=True)

    # Load the saved model pipeline
    model = joblib.load('model_pipeline.pkl')

    # Predict on test data
    predictions = model.predict(test_data)
    print("Predictions on test data:", predictions)

    # Save predictions to a file
    test_data['Predictions'] = predictions
    test_data.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'.")

# Example usage
if __name__ == "__main__":
    train_file = "train_transformed.csv"  # Replace with your training dataset path
    test_file = "test_transformed.csv"    # Replace with your test dataset path
    target_column = "Class/ASD"       # Replace with your target column name

    # Train the model
    train_model_pipeline(train_file, target_column)

    # Evaluate the model
    evaluate_model_pipeline(test_file)
