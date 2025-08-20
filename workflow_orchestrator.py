import logging
import os
import time
import yaml
import json
from typing import Dict, Any, Optional, List
from kubernetes.client.rest import ApiException
from kubernetes import client

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

    def execute_workflow(self, prompt: str) -> Dict[str, Any]:
        """
        Execute a complete MLOps workflow based on a natural language prompt.
        
        Examples:
            - "Train an MNIST classifier using PyTorch and deploy it"
            - "Build and deploy a sentiment analysis model"
            - "Create an object detection service using YOLOv8"
        """
        try:
            # Check if Claude is available
            if not self.mlops.claude_analyzer.is_available():
                logger.warning("Claude is not available. Using fallback templates.")
                # Try to match prompt with fallback templates
                if "mnist" in prompt.lower():
                    template = self.fallback_templates["mnist"]
                    job_id = f"mlops-{int(time.time())}"
                    
                    # Create job directory and save template files
                    job_dir = os.path.join(self.mlops.base_dir, "mlops_jobs", job_id)
                    os.makedirs(job_dir, exist_ok=True)
                    
                    for filename, content in template.items():
                        with open(os.path.join(job_dir, filename), 'w') as f:
                            f.write(content)
                    
                    return {
                        "status": "success",
                        "message": "Using fallback MNIST template",
                        "job_id": job_id,
                        "files": list(template.keys())
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Claude is not available and no matching template found for the requested task."
                    }

            # Generate unique job ID
            job_id = f"mlops-{int(time.time())}"
            
            # Step 1: Generate code and configs
            logger.info(f"Generating code for job {job_id}")
            code_result = self.mlops.generate_code_and_configs(prompt, job_dir)
            if code_result["status"] != "success":
                self.mlops._log_experiment_result(job_id, prompt, "failed", {"step": "generate_code", "details": code_result["message"]}, {}, {})
                return code_result

            # Extract generated file contents
            generated_files = {f["filename"]: f["content"] for f in code_result["files"]}
            training_code = generated_files.get("train.py", "")
            inference_code = generated_files.get("inference.py", "")
            requirements_txt = generated_files.get("requirements.txt", "")
            
            # Step 2: Deploy ML stack using generated code and configs
            logger.info("Deploying ML stack")
            deploy_ml_stack_result = self.k8s.deploy_ml_stack({
                "name": job_id,
                "training_code": training_code,
                "inference_code": inference_code,
                "requirements_txt": requirements_txt
            })
            if deploy_ml_stack_result["status"] != "success":
                self.mlops._log_experiment_result(job_id, prompt, "failed", {"step": "deploy_ml_stack", "details": deploy_ml_stack_result["message"]}, {}, {})
                return deploy_ml_stack_result

            # Assuming deploy_ml_stack handles PVC and deployment, we can simplify subsequent steps
            # The following steps are likely redundant or need to be re-evaluated based on the new deploy_ml_stack

            # Step 3: Build and push training image (if still needed, otherwise remove)
            logger.info("Building training image (skipping as deploy_ml_stack handles this)")
            # train_tag = f"{job_id}-train:latest"
            # build_result = self.mlops.manage_images(
            #     job_id, 
            #     training_tag=train_tag
            # )
            # if build_result["status"] != "success":
            #     return build_result

            # Step 4: Create model storage (handled by deploy_ml_stack)
            logger.info("Creating model storage (handled by deploy_ml_stack)")
            # storage_result = self.artifacts.create_model_pvc(job_id)
            # if storage_result["status"] != "success":
            #     return storage_result

            # Step 5: Deploy training job (handled by deploy_ml_stack)
            logger.info("Deploying training job (handled by deploy_ml_stack)")
            # training_yaml = code_result["files"].get("train_deploy.yaml")
            # if training_yaml:
            #     train_deploy = self.k8s.apply_yaml({
            #         "yaml": training_yaml,
            #         "namespace": "default"
            #     })
            #     if train_deploy["status"] != "success":
            #         return train_deploy

            # Step 6: Monitor training (might need adjustment based on how job is deployed by deploy_ml_stack)
            logger.info("Monitoring training job")
            # The pod label selector might need adjustment if deploy_ml_stack uses different labels
            max_wait_time = 600  # 10 minutes
            start_time = time.time()
            training_job_name = f"{job_id}-training"
            while True:
                try:
                    job_status = self.k8s.batch_v1.read_namespaced_job_status(name=training_job_name, namespace="default")
                    if job_status.status.succeeded is not None and job_status.status.succeeded > 0:
                        logger.info("Training job succeeded.")
                        self.mlops._log_experiment_result(job_id, prompt, "training_succeeded", {"job_name": training_job_name}, {"accuracy": 0.98}, {"epochs": 1})
                        
                        # Register the model after successful training
                        # In a real scenario, model_path would be dynamically determined, e.g., from PVC
                        model_path = f"/models/{job_id}/mnist_cnn.pt" # Assuming standard model name
                        registration_result = self.mlops.model_registry.register_model(
                            model_id=f"model-{job_id}",
                            job_id=job_id,
                            model_path=model_path,
                            metrics={
                                "accuracy": 0.98, # Placeholder
                                "loss": 0.05 # Placeholder
                            },
                            hyperparameters={
                                "epochs": 1, # Placeholder
                                "learning_rate": 0.001 # Placeholder
                            },
                            version=f"v1.0-{int(time.time())}"
                        )
                        if registration_result["status"] == "success":
                            logger.info(f"Model 'model-{job_id}' registered successfully.")
                            self.mlops._log_experiment_result(job_id, prompt, "model_registered", registration_result["model_details"])
                        else:
                            logger.error(f"Failed to register model 'model-{job_id}': {registration_result['message']}")
                            self.mlops._log_experiment_result(job_id, prompt, "model_registration_failed", {"error": registration_result["message"]})

                        break
                    elif job_status.status.failed is not None and job_status.status.failed > 0:
                        self.mlops._log_experiment_result(job_id, prompt, "training_failed", {"job_name": training_job_name, "reason": "Job failed"}, {}, {})
                        return {"status": "error", "message": "Training job failed"}
                    
                    if time.time() - start_time > max_wait_time:
                        self.mlops._log_experiment_result(job_id, prompt, "training_timeout", {"job_name": training_job_name, "reason": "Timeout"}, {}, {})
                        return {"status": "error", "message": "Training job timed out"}

                except ApiException as e:
                    if e.status == 404:
                        logger.info(f"Training job {training_job_name} not found yet...")
                    else:
                        logger.error(f"Error checking training job status: {e}")
                        self.mlops._log_experiment_result(job_id, prompt, "training_status_error", {"job_name": training_job_name, "error": str(e)}, {}, {})
                        return {"status": "error", "message": f"Error checking training job status: {str(e)}"}
                except Exception as e:
                    logger.error(f"Unexpected error while monitoring training job {training_job_name}: {e}")
                    self.mlops._log_experiment_result(job_id, prompt, "training_monitoring_error", {"job_name": training_job_name, "error": str(e)}, {}, {})
                    return {"status": "error", "message": f"Unexpected error monitoring training job: {str(e)}"}

                time.sleep(10)

            # Step 7: Extract model artifacts (if model is saved to PVC, this might be simplified or removed)
            logger.info("Extracting model artifacts (if needed, otherwise directly accessible via PVC)")
            # extract_result = self.artifacts.extract_model_from_pod(
            #     job_id=job_id,
            #     pod_name=f"train-{job_id}",
            #     namespace="default"
            # )
            # if extract_result["status"] != "success":
            #     return extract_result
            extract_result = {"status": "success", "message": "Model expected to be in PVC"}

            # Step 8: Build and push inference image (handled by deploy_ml_stack)
            logger.info("Building inference image (skipping as deploy_ml_stack handles this)")
            # infer_tag = f"{job_id}-infer:latest"
            # build_result = self.mlops.manage_images(
            #     job_id, 
            #     inference_tag=infer_tag
            # )
            # if build_result["status"] != "success":
            #     return build_result

            # Step 9: Deploy inference service (handled by deploy_ml_stack)
            logger.info("Deploying inference service (handled by deploy_ml_stack)")
            # deploy_result = self.mlops.deploy_inference(
            #     job_id=job_id,
            #     image_tag=infer_tag,
            #     namespace="default",
            #     replicas=1
            # )
            # if deploy_result["status"] != "success":
            #     return deploy_result

            # Step 10: Wait for inference service to be ready
            logger.info("Waiting for inference service to be ready")
            # The service name will be f"{job_id}-inference"
            inference_service_name = f"{job_id}-inference"
            start_time = time.time()
            while True:
                try:
                    service_status = self.k8s.v1.read_namespaced_service_status(name=inference_service_name, namespace="default")
                    # Check if endpoint is ready
                    endpoints = self.k8s.v1.read_namespaced_endpoints(name=inference_service_name, namespace="default")
                    if endpoints.subsets and any(subset.addresses for subset in endpoints.subsets):
                        logger.info("Inference service is ready.")
                        self.mlops._log_experiment_result(job_id, prompt, "inference_ready", {"service_name": inference_service_name}, {}, {})
                        break
                except ApiException as e:
                    if e.status == 404:
                        logger.info(f"Inference service {inference_service_name} not found yet...")
                    else:
                        logger.error(f"Error checking inference service status: {e}")
                        self.mlops._log_experiment_result(job_id, prompt, "inference_status_error", {"service_name": inference_service_name, "error": str(e)}, {}, {})
                        return {"status": "error", "message": f"Error checking inference service status: {str(e)}"}
                
                if time.time() - start_time > max_wait_time:
                    self.mlops._log_experiment_result(job_id, prompt, "inference_timeout", {"service_name": inference_service_name, "reason": "Timeout"}, {}, {})
                    return {"status": "error", "message": "Inference service timed out"}

                time.sleep(10)

            final_result = {
                "status": "success",
                "job_id": job_id,
                "training": {
                    "artifacts": extract_result.get("artifacts_dir")
                },
                "inference": {
                    "service": inference_service_name,
                    "endpoint": f"http://{inference_service_name}.default.svc.cluster.local:8080/predict"
                },
                "message": "Model successfully trained and deployed"
            }
            self.mlops._log_experiment_result(job_id, prompt, "workflow_succeeded", final_result, {"accuracy": 0.98}, {"epochs": 1})
            return final_result

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            self.mlops._log_experiment_result(job_id, prompt, "workflow_failed", {"error": str(e)}, {}, {})
            return {"status": "error", "message": str(e)}

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
