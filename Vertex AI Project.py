from google.cloud import aiplatform
import pandas as pd
import os

# Load dataset from the local CSV file
csv_file_path = r"C:\Users\kimho\Downloads\archive\car_price_prediction_.csv"  # Correct path to your file

def load_dataset(file_path):
    try:
        dataset_local = pd.read_csv(file_path, delimiter=',')
        print("Dataset loaded successfully.")
        print("First few rows of the dataset:\n", dataset_local.head())  # Check if the data loads correctly
        print("Column names before renaming:", dataset_local.columns)  # Check column names to confirm proper loading
        return dataset_local
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        exit()

# Rename columns (uppercase with underscores)
def rename_columns(df):
    df.columns = [col.strip().upper().replace(' ', '_') for col in df.columns]
    return df

# Clean and process data locally
def clean_data(df):
    df = rename_columns(df)
    if 'CAR_PRICE' not in df.columns:
        print("Error: 'CAR_PRICE' column not found after renaming.")
        exit()
    
    # Coerce 'CAR_PRICE' to numeric
    df['CAR_PRICE'] = pd.to_numeric(df['CAR_PRICE'], errors='coerce')

    # Check for missing values and handle them (e.g., drop rows)
    missing_values_count = df['CAR_PRICE'].isnull().sum()
    if missing_values_count > 0:
        print(f"Warning: {missing_values_count} non-numeric or missing values found in the 'CAR_PRICE' column. Dropping these rows.")
        df = df.dropna(subset=['CAR_PRICE'])
    
    return df

# Save the updated CSV to the local file
def save_cleaned_data(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Cleaned dataset saved to {file_path}.")

# Initialize Vertex AI
def initialize_vertex_ai():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\kimho\Downloads\Vertex AI\ai-project-1-439012-15a06614f903.json"
    aiplatform.init(project="ai-project-1-439012", location="us-central1")

# Create Dataset for AutoML using data from Google Cloud Storage
def create_dataset(display_name, gcs_uri):
    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=gcs_uri
    )
    print(f"Dataset created: {dataset.display_name} with ID: {dataset.resource_name}")
    return dataset

# Train a Model using the created dataset
def train_model(display_name, dataset, target_column):
    optimization_prediction_type = "regression"  # Regression for car price prediction
    optimization_objective = "minimize-mae"  # Minimize Mean Absolute Error

    model = aiplatform.AutoMLTabularTrainingJob(
        display_name=display_name,
        optimization_prediction_type=optimization_prediction_type,
        optimization_objective=optimization_objective
    )

    try:
        model = model.run(
            dataset=dataset,
            model_display_name=display_name,
            target_column=target_column
        )
        print(f"Model training complete: {model.display_name}")
        return model
    except Exception as e:
        print(f"Model training failed: {e}")
        exit()

# Deploy the trained model to an endpoint
def deploy_model(model):
    endpoint = model.deploy(
        deployed_model_display_name="market_risk_model_deployed",  # Correct parameter
        machine_type="n1-standard-4"  # Specify the machine type for deployment
    )

    print(f"Model deployed to endpoint with ID: {endpoint.resource_name}")
    return endpoint


# Make predictions using the deployed model
def make_prediction(endpoint, instances):
    response = endpoint.predict(instances=instances)
    print("Prediction Results:", response.predictions)
    return response.predictions

def main():
    # Load and clean dataset
    dataset_local = load_dataset(csv_file_path)
    dataset_local = clean_data(dataset_local)
    save_cleaned_data(dataset_local, csv_file_path)

    # Initialize Vertex AI
    initialize_vertex_ai()

    # Define GCS URI for the dataset (replace with your actual GCS URI)
    gcs_uri = "gs://car_price_data_kaggle/car_price_prediction_.csv"  # Replace with your actual GCS path

    # Create Dataset from GCS
    dataset = create_dataset(
        display_name="market_risk_dataset",
        gcs_uri=gcs_uri
    )

    # Train a Model using the created dataset
    model = train_model(
        display_name="market_risk_model",
        dataset=dataset,
        target_column="CAR_PRICE"
    )

    # Deploy the trained model
    endpoint = deploy_model(model)

    # Example prediction input (ensure feature names match the renamed columns)
    prediction_input = [{
        "MILEAGE": 10.5,
        "ENGINE_SIZE": 1.6,
        "BRAND": "toyota",  
        "TRANSMISSION": "manual",  
        "CONDITION": "new",  
        "FUEL_TYPE": "petrol",  
        "YEAR": 2018,
        "MODEL": "camry"  
    }]

    # Call the prediction function
    make_prediction(endpoint, prediction_input)

if __name__ == "__main__":
    main()
