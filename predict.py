import gradio as gr
import pandas as pd
from google.cloud import aiplatform

# Load dataset from the local CSV file
csv_file_path = r"C:\Users\kimho\Downloads\archive\car_price_prediction_.csv"  # Correct path to your file

try:
    # Load the CSV file
    data = pd.read_csv(csv_file_path, delimiter=',')
    print("Dataset loaded successfully.")
    print("First few rows of the dataset:\n", data.head())  # Check if the data loads correctly
    print("Column names before renaming:", data.columns)  # Check column names to confirm proper loading

    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

except FileNotFoundError:
    print(f"Error: The file at {csv_file_path} was not found.")
    exit()

# Ensure the 'YEAR' column is treated as strings
data['YEAR'] = data['YEAR'].astype(str)

# Convert ENGINE_SIZE to string to avoid AttributeError when using .str accessor
data['ENGINE_SIZE'] = data['ENGINE_SIZE'].astype(str)

# Check the data types of each column for debugging
print("Data types:\n", data.dtypes)

# Extract unique values for dropdowns, handling any NaN values and stripping spaces
brands = data['BRAND'].dropna().str.strip().unique().tolist()
years = sorted(data['YEAR'].dropna().unique().tolist())  # Use uppercase 'YEAR'
engine_sizes = sorted(data['ENGINE_SIZE'].dropna().str.strip().unique().tolist())  # Now this should work
fuel_types = sorted(data['FUEL_TYPE'].dropna().str.strip().unique().tolist())
transmissions = sorted(data['TRANSMISSION'].dropna().str.strip().unique().tolist())
conditions = sorted(data['CONDITION'].dropna().str.strip().unique().tolist())
models = sorted(data['MODEL'].dropna().str.strip().unique().tolist())

# Define the prediction function
def make_prediction(car_id, brand, year, engine_size, fuel_type, transmission, mileage, condition, model):
    try:
        # Ensure CAR_ID is numeric
        car_id = int(car_id)
    except ValueError:
        return "Error: CAR_ID must be a numeric value.", None, None, None

    # Prepare the input for prediction
    prediction_input = [
        {
            "CAR_ID": str(car_id),
            "BRAND": brand,
            "YEAR": year,
            "ENGINE_SIZE": engine_size,
            "FUEL_TYPE": fuel_type,
            "TRANSMISSION": transmission,
            "MILEAGE": mileage,
            "CONDITION": condition,
            "MODEL": model
        }
    ]

    # Debugging output
    print("Prediction Input:", prediction_input)

    # Initialize the endpoint with the correct endpoint name
    endpoint = aiplatform.Endpoint(endpoint_name="projects/ai-project-1-439012/locations/us-central1/endpoints/8384665666698870784")

    # Make the prediction
    try:
        response = endpoint.predict(instances=prediction_input)
        print("API Response:", response)  # Full response for debugging

        if hasattr(response, 'predictions') and len(response.predictions) > 0:
            prediction = response.predictions[0]
            predicted_price = prediction.get("predicted_price")
            upper_bound = prediction.get("upper_bound")
            time_taken = prediction.get("time_taken")
            prediction_graph = prediction.get("prediction_graph")

            print("Predicted Price:", predicted_price)  # Check value
            print("Prediction Graph:", prediction_graph)  # Check value

            return predicted_price, upper_bound, time_taken, prediction_graph
        else:
            error_message = response.error if hasattr(response, 'error') else 'No predictions found.'
            return f"Error: No predictions found in response. Details: {error_message}", None, None, None
    except Exception as e:
        return f"Prediction error: {str(e)}", None, None, None




# Define the Gradio interface
interface = gr.Interface(
    fn=make_prediction,
    inputs=[
        gr.Textbox(label="Car ID"),  # Textbox for Car ID input
        gr.Dropdown(label="Brand", choices=brands),  # Dropdown for Brand
        gr.Dropdown(label="Year", choices=years),  # Dropdown for Year
        gr.Dropdown(label="Engine Size", choices=engine_sizes),  # Dropdown for Engine Size
        gr.Dropdown(label="Fuel Type", choices=fuel_types),  # Dropdown for Fuel Type
        gr.Dropdown(label="Transmission", choices=transmissions),  # Dropdown for Transmission
        gr.Textbox(label="Mileage"),  # Textbox for Mileage input
        gr.Dropdown(label="Condition", choices=conditions),  # Dropdown for Condition
        gr.Dropdown(label="Model", choices=models)  # Dropdown for Model
    ],
    outputs=[
        gr.Textbox(label="Predicted Car Price"),
        gr.Textbox(label="Upper Bound of Confidence Interval"),
        gr.Textbox(label="Time Taken to Predict"),
        gr.Image(label="Prediction Graph")
    ],
    title="Car Price Prediction",
    description="Enter the car details to predict the car price using a machine learning model hosted on Vertex AI."
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(share=True)
