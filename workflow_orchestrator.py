import logging
import os
import time
import yaml
import json
from typing import Dict, Any, Optional, List
from kubernetes.client.rest import ApiException
from kubernetes import client

from model_definitions import MODEL_CONFIGURATIONS # Import model configurations

logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    """Orchestrates end-to-end MLOps workflows from high-level prompts."""
    
    def __init__(self, mlops_workflow, k8s_debugger, artifact_manager):
        self.mlops = mlops_workflow
        self.k8s = k8s_debugger
        self.artifacts = artifact_manager
        
        # Complete fallback templates for when Claude is not available
        self.fallback_templates = {
            "mnist": {
                "train.py": """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define CNN model for MNIST
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load test data
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Initialize model, optimizer and loss function
model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train(epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main function to run training and save the model
def main():
    epochs = 1
    print("Starting training...")
    train(epochs)
    print("Training finished.")

    print("Starting testing...")
    test()
    print("Testing finished.")
    
    # Save the model
    os.makedirs('/mnt/model', exist_ok=True)
    torch.save(model.state_dict(), '/mnt/model/mnist_cnn.pt')
    print("Model saved to /mnt/model/mnist_cnn.pt")

if __name__ == '__main__':
    main()
""",
                "inference.py": self._generate_inference_script(
                    model_name="MNISTNet",
                    model_class_code="""class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)""",
                    model_load_logic="""model = MNISTNet()
model.load_state_dict(torch.load('model.pt'))
model.eval()""",
                    transform_code="""transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])"""
                ),
                "Dockerfile.train": """
FROM pytorch/pytorch:latest
WORKDIR /app
COPY . .
CMD ["python", "train.py"]
""",
                "Dockerfile.infer": """
FROM pytorch/pytorch:latest
WORKDIR /app
COPY . .
EXPOSE 8080
CMD ["python", "inference.py"]
"""
            },
            # Add more templates for common tasks
        }

    def _generate_inference_script(self, model_name: str, model_class_code: str, model_load_logic: str, transform_code: str, model_file_name: str = "model.pt") -> str:
        # Basic inference script template
        # This will be refined as we generalize more.
        return f"""
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
import os

app = Flask(__name__)

# Define model class
{model_class_code}

# Load the trained model
def load_model(model_path):
    {model_load_logic}
    return model

model_path = '/app/{model_file_name}' # Model is mounted at /app/ by default via PVC
if not os.path.exists(model_path):
    print(f"Error: Model not found at {{model_path}}")
    exit(1)

model = load_model(model_path)
print("Model loaded successfully.")

# Define transformation
transform = {transform_code}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'healthy', 'model': '{model_name}'}})

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({{'error': 'No image provided in JSON body'}}), 400

        img_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_data)).convert('L')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True)
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities.max().item()
        
        return jsonify({{
            'prediction': pred.item(), 
            'confidence': round(confidence, 4),
            'probabilities': probabilities.squeeze().tolist()
        }})
    except Exception as e:
        return jsonify({{'error': str(e)}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
"""

    def execute_workflow(self, workflow_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete MLOps workflow based on a structured dictionary of parameters.
        This function orchestrates the steps like code generation, image building, 
        model registration, and inference deployment.
        """
        try:
            model_type = workflow_params.get("model_type")
            if not model_type:
                return {"status": "error", "message": "'model_type' is required in workflow_params."}

            training_params = workflow_params.get("training_params", {})
            deployment_params = workflow_params.get("deployment_params", {})
            data_params = workflow_params.get("data_params", {})

            # Step 1: Generate Code and Configurations
            logger.info(f"Executing workflow for model type: {model_type}")
            job_id = self.mlops.generate_job_dir(model_type)
            code_result = self.mlops.generate_code_and_configs(
                model_type=model_type,
                job_dir=job_id,
                data_path=data_params.get("path", "data") # Pass data_path to code generation
            )
            if code_result["status"] != "success":
                return code_result
            
            # Step 2: Build Training Image
            training_image_tag = f"{job_id}-train:latest"
            logger.info(f"Building training image: {training_image_tag}")
            build_train_result = self.mlops.build_image(
                job_dir=job_id,
                dockerfile="Dockerfile.train",
                image_tag=training_image_tag
            )
            if build_train_result["status"] != "success":
                return build_train_result

            # Step 3: Deploy and Monitor Training Job
            logger.info(f"Deploying training job for model type: {model_type}")
            train_job_deploy_result = self.mlops.deploy_training_job(
                job_id=job_id,
                image_tag=training_image_tag,
                namespace=training_params.get("namespace", "default"),
                resource_requests=training_params.get("resource_requests"),
                resource_limits=training_params.get("resource_limits")
            )
            if train_job_deploy_result["status"] != "success":
                return train_job_deploy_result

            # Simulate conditional logic: Only proceed to deployment if training was successful
            deploy_if_trained_success = deployment_params.get("deploy_if_trained_success", True)

            if deploy_if_trained_success and train_job_deploy_result["status"] == "success":
                # Step 4: Register Model
                model_id = f"model-{job_id}"
                logger.info(f"Registering model: {model_id}")
                registered_model_path = f"/mnt/model/mnist_cnn.pt" # This should be the path within the training container

                registration_result = self.mlops.model_registry.register_model(
                    model_id=model_id,
                    job_id=job_id,
                    model_path=registered_model_path, 
                    metrics={"accuracy": 0.98, "loss": 0.05}, # Placeholder metrics
                    hyperparameters={"epochs": training_params.get("epochs", 1)}, # Placeholder
                    version=f"v1.0-{int(time.time())}"
                )
                if registration_result["status"] != "success":
                    return registration_result

                # Step 5: Build Inference Image
                inference_image_tag = f"{job_id}-infer:latest"
                logger.info(f"Building inference image: {inference_image_tag}")
                build_infer_result = self.mlops.build_image(
                    job_dir=job_id,
                    dockerfile="Dockerfile.infer",
                    image_tag=inference_image_tag
                )
                if build_infer_result["status"] != "success":
                    return build_infer_result
                
                # Step 6: Deploy Inference Service
                logger.info(f"Deploying inference service for model {model_id}")
                # Retrieve the model details to get the inference code for deployment
                model_details_result = self.mlops.model_registry.get_model_details(model_id)
                if model_details_result["status"] != "success":
                    return model_details_result
                model_details = model_details_result["model_details"]
                
                # Re-generate inference code for deployment (as it contains model-specific details)
                model_config = MODEL_CONFIGURATIONS[model_type]
                model_file_name = os.path.basename(model_details["model_path"])
                inference_code_for_deployment = self._generate_inference_script(
                    model_name=model_config["model_name"],
                    model_class_code=model_config["model_class_code"],
                    model_load_logic=model_config["model_load_logic"],
                    transform_code=model_config["transform_code"],
                    model_file_name=model_file_name
                )

                deploy_result = self.mlops.deploy_inference_service_with_code(
                    job_id=job_id,
                    model_id=model_id,
                    inference_code=inference_code_for_deployment,
                    namespace=deployment_params.get("namespace", "default"),
                    replicas=deployment_params.get("replicas", 1)
                )
                if deploy_result["status"] != "success":
                    return deploy_result

                return {"status": "success", "message": f"Workflow completed for {model_type}. Model deployed.", "job_id": job_id}
            else:
                logger.info(f'Skipping deployment for {model_type} due to conditional check (deploy_if_trained_success={deploy_if_trained_success}, training_status={train_job_deploy_result["status"]}).')
                return {"status": "success", "message": f"Workflow completed for {model_type}. Deployment skipped based on conditions.", "job_id": job_id}

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

    def run_inference(self, job_id: str, data: Any) -> Dict[str, Any]:
        """
        Run inference on a deployed model.
        """
        try:
            # Get service details
            service = self.k8s.get_services({
                "namespace": "default",
                "label_selector": f"job-id={job_id},type=inference"
            })
            
            if not service.get("services"):
                return {"status": "error", "message": "Inference service not found"}

            # Forward request to the service
            import requests
            response = requests.post(
                f"http://inference-{job_id}.default.svc.cluster.local:8080/predict",
                json={"data": data}
            )
            
            return {
                "status": "success",
                "prediction": response.json()
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"status": "error", "message": str(e)}
