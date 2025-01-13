import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
def load_data(file_path):
    """Loads the dataset and returns a DataFrame."""
    return pd.read_csv(file_path)

# Drop unnecessary columns
def drop_columns(df, columns_to_drop):
    """Drops specified columns from the DataFrame."""
    return df.drop(columns=columns_to_drop, axis=1)

# Handle missing or placeholder values
def handle_missing_values(df, placeholder="?"):
    """Replaces placeholders with 'Unknown' and handles missing values."""
    return df.replace(placeholder, 'Unknown')

# Encode categorical features
def encode_categorical_columns(df, categorical_columns):
    """Encodes categorical columns using LabelEncoder."""
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# Scale numerical features
def scale_numerical_columns(df, numerical_columns):
    """Scales numerical columns using StandardScaler."""
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

# Save transformed data to a file
def save_transformed_data(df, output_file):
    """Saves the transformed data to a CSV file."""
    df.to_csv(output_file, index=False)

# Main transformation pipeline
def transform_data(file_path, output_file):
    """Executes the full data transformation pipeline."""
    df = load_data(file_path)

    # Step 1: Drop unnecessary columns
    columns_to_drop = ['ID', 'age_desc', 'relation']
    df = drop_columns(df, columns_to_drop)

    # Step 2: Handle missing or placeholder values
    df = handle_missing_values(df)

    # Step 3: Encode categorical features
    categorical_columns = ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before']
    df, label_encoders = encode_categorical_columns(df, categorical_columns)

    # Step 4: Scale numerical features
    numerical_columns = ['age', 'result']
    df, scaler = scale_numerical_columns(df, numerical_columns)

    # Save the transformed dataset
    save_transformed_data(df, output_file)

    return label_encoders, scaler

# Example usage
file_path = 'test.csv'  # Replace with the correct file path
output_file = 'test_transformed.csv'  # Specify the output file name
label_encoders, scaler = transform_data(file_path, output_file)
