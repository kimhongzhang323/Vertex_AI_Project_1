# Car Price Prediction Project
Learning Based Project

## Overview
This project aims to predict car prices based on various features such as brand, model, year, engine size, fuel type, and mileage using Google Cloud's Vertex AI and AutoML capabilities. The dataset used for training the model is sourced from a CSV file containing historical car prices.

## Features
- Predicts car prices using regression algorithms.
- Utilizes Google Cloud's Vertex AI for model training and deployment.
- Provides easy integration for making predictions via an API endpoint.

## Table of Contents
1. [Technologies](#technologies)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Training the Model](#training-the-model)
6. [Deployment](#deployment)
7. [Making Predictions](#making-predictions)
8. [License](#license)

## Technologies
- Python
- Google Cloud Platform (GCP)
- Vertex AI
- Pandas
- Scikit-learn

## Dataset
The dataset used for this project is in CSV format and includes the following columns:
- `Car ID`
- `Brand`
- `Year`
- `Engine Size`
- `Fuel Type`
- `Transmission`
- `Mileage`
- `Condition`
- `Price` (Target variable)
- `Model`

**Sample Dataset Path:**
\car_price_prediction_.csv

## Installation
To run this project, you need to have Python and the required libraries installed. Use the following command to install the necessary packages:

pip install google-cloud-aiplatform pandas scikit-learn
## Set Up Google Cloud
Create a Google Cloud account and set up a project.
Enable the Vertex AI API in your GCP project.
Set up authentication by downloading your service account key (JSON file) and setting the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of this key.

## Usage
Clone the repository or download the project files.
Update the paths and settings in the script as necessary.
Run the main script to load the dataset, train the model, and deploy it.
## Example Command
python Vertex_AI_Project.py 

## Training the Model
The model is trained using AutoML for tabular regression. The target variable for prediction is the Price column. The training process is handled by the train_model function, which utilizes the AutoMLTabularTrainingJob from the Vertex AI library.

## Deployment
Once trained, the model is deployed to a Vertex AI endpoint for making predictions. The deployment process is handled by the deploy_model function, which routes incoming prediction requests to the deployed model.

## Making Predictions
Predictions can be made by sending instances (feature data) to the deployed endpoint using the make_prediction function. Here's an example of how to format your input:

prediction_input = [
    {"Mileage": 10.5, "Engine Size": 1.6, "Brand": "Toyota"}
]
## License
This project is licensed under the MIT License. See the LICENSE file for more information.
