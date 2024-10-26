from google.cloud import aiplatform

# Initialize Vertex AI platform
aiplatform.init(project="ai-project-1-439012", location="us-central1")

# Deploy the trained model
def deploy_model(model_id):
    # Load the model by its ID
    model = aiplatform.Model(model_id)

    # Deploy the model to an endpoint
    endpoint = model.deploy(
        deployed_model_display_name="market_risk_model_deployed",
        machine_type="n1-standard-4"
    )

    print(f"Model deployed to endpoint with ID: {endpoint.resource_name}")
    return endpoint

# Example usage
if __name__ == "__main__":
    model_id = "2532506231768088576"  # Use your trained model ID
    deploy_model(model_id)
