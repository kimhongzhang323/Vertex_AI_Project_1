# data_preprocessing.py
import pandas as pd

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Check for missing values in the target column
    missing_values = df['Price'].isnull().sum()
    print(f'Missing values in Price column: {missing_values}')

    # Optionally drop rows with missing labels
    df.dropna(subset=['Price'], inplace=True)

    # Save the preprocessed data back to CSV
    processed_file_path = file_path.replace('.csv', '_processed.csv')
    df.to_csv(processed_file_path, index=False)
    
    return processed_file_path

# For testing purposes, if you run this script directly
if __name__ == "__main__":
    file_path = "gs://car_price_data_kaggle/car_price_prediction_.csv"
    preprocess_data(file_path)
