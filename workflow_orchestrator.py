import logging
import os
import time
import yaml
from typing import Dict, Any, Optional, List

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
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_
""",
                "inference.py": """
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Define CNN model for MNIST (same as in train.py)
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

# Load the trained model
model = MNISTNet()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
    
    return jsonify({'prediction': pred.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
""",
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
            code_result = self.mlops.generate_code_and_configs(prompt, job_id)
            if code_result["status"] != "success":
                return code_result
                
            # Step 2: Build and push training image
            logger.info("Building training image")
            train_tag = f"{job_id}-train:latest"
            build_result = self.mlops.manage_images(
                job_id, 
                training_tag=train_tag
            )
            if build_result["status"] != "success":
                return build_result

            # Step 3: Create model storage
            logger.info("Creating model storage")
            storage_result = self.artifacts.create_model_pvc(job_id)
            if storage_result["status"] != "success":
                return storage_result

            # Step 4: Deploy training job
            logger.info("Deploying training job")
            training_yaml = code_result["files"].get("train_deploy.yaml")
            if training_yaml:
                train_deploy = self.k8s.apply_yaml({
                    "yaml": training_yaml,
                    "namespace": "default"
                })
                if train_deploy["status"] != "success":
                    return train_deploy

            # Step 5: Monitor training
            logger.info("Monitoring training job")
            while True:
                pod_status = self.k8s.get_pods({
                    "namespace": "default",
                    "label_selector": f"job-id={job_id},type=training"
                })
                if pod_status.get("pods", []):
                    pod = pod_status["pods"][0]
                    if pod["status"] == "Succeeded":
                        break
                    elif pod["status"] == "Failed":
                        return {"status": "error", "message": "Training job failed"}
                time.sleep(30)

            # Step 6: Extract model artifacts
            logger.info("Extracting model artifacts")
            extract_result = self.artifacts.extract_model_from_pod(
                job_id=job_id,
                pod_name=f"train-{job_id}",
                namespace="default"
            )
            if extract_result["status"] != "success":
                return extract_result

            # Step 7: Build and push inference image
            logger.info("Building inference image")
            infer_tag = f"{job_id}-infer:latest"
            build_result = self.mlops.manage_images(
                job_id, 
                inference_tag=infer_tag
            )
            if build_result["status"] != "success":
                return build_result

            # Step 8: Deploy inference service
            logger.info("Deploying inference service")
            deploy_result = self.mlops.deploy_inference(
                job_id=job_id,
                image_tag=infer_tag,
                namespace="default",
                replicas=1
            )
            if deploy_result["status"] != "success":
                return deploy_result

            # Step 9: Wait for inference service to be ready
            logger.info("Waiting for inference service to be ready")
            while True:
                status = self.mlops.get_inference_status(job_id)
                if status["status"] == "success":
                    if status["deployment"]["ready_replicas"] == status["deployment"]["available_replicas"]:
                        break
                time.sleep(10)

            return {
                "status": "success",
                "job_id": job_id,
                "training": {
                    "image": train_tag,
                    "artifacts": extract_result.get("artifacts_dir")
                },
                "inference": {
                    "image": infer_tag,
                    "service": f"inference-{job_id}",
                    "endpoint": f"http://inference-{job_id}.default.svc.cluster.local:8080/predict"
                },
                "message": "Model successfully trained and deployed"
            }

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
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
