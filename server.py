#!/usr/bin/env python3

import os
import sys
import logging
from typing import Dict, Any, List, Optional  # Add this import line
import textwrap # Import textwrap
import time

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if (current_dir not in sys.path):
    sys.path.append(current_dir)

from fastmcp import FastMCP
from kubernetes import client, config
from k8s_debugger import KubernetesDebugger
from claude_analyzer import ClaudeAnalyzer
from dvc_manager import DVCManager
from mlops_workflow import MLOpsWorkflow
from artifact_manager import ArtifactManager
from workflow_orchestrator import WorkflowOrchestrator
from storage_backends import KubernetesPvcBackend # Corrected this import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mcp_instance = None  # Global variable to hold the FastMCP instance
k8s_client_initialized = False # Global flag for Kubernetes client status

def setup_kubernetes(kubeconfig_content: Optional[str] = None):
    """Setup Kubernetes configuration. Returns True if successful, False otherwise."""
    global k8s_client_initialized
    try:
        if kubeconfig_content:
            # Save kubeconfig content to a temporary file and load it
            kubeconfig_path = os.path.join(current_dir, ".kubeconfig_temp")
            with open(kubeconfig_path, "w") as f:
                f.write(kubeconfig_content)
            config.load_kube_config(config_file=kubeconfig_path)
            logger.info("Loaded kubeconfig from provided content successfully")
        else:
            config.load_kube_config()
            logger.info("Loaded kubeconfig successfully")
        k8s_client_initialized = True
        return True
    except Exception as e1:
        logger.warning(f"Failed to load kubeconfig: {e1}")
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster config successfully")
            k8s_client_initialized = True
            return True
        except Exception as e2:
            logger.error(f"Failed to load Kubernetes config: {e2}")
            k8s_client_initialized = False
            return False

def initialize_mcp_server(kubeconfig_content: Optional[str] = None):
    """Initialize and configure the MCP server dynamically."""
    global mcp_instance, k8s_client_initialized
    if mcp_instance:
        logger.info("MCP server already initialized.")
        return {"status": "success", "message": "MCP server already initialized.", "initialized": True}

    try:
        logger.info("Attempting to initialize MCP server...")
        # Set up Kubernetes
        k8s_available = setup_kubernetes(kubeconfig_content)
        if not k8s_available:
            logger.warning("Kubernetes API is not available. MCP server will start with limited functionality.")
            return {"status": "error", "message": "Kubernetes API is not available.", "initialized": False}

        # Set up Claude API
        claude_api_key = os.environ.get("CLAUDE_API_KEY")
        if not claude_api_key:
            logger.warning("No Claude API key provided. Claude analysis features will be disabled.")

        # Initialize the Claude analyzer
        claude_analyzer = ClaudeAnalyzer(claude_api_key)

        # Initialize Kubernetes debugger (pass k8s_available flag)
        k8s_debugger = KubernetesDebugger(claude_analyzer, k8s_available=k8s_available)

        # Initialize DVC manager (assume repo_dir is current_dir)
        dvc_manager = DVCManager(repo_dir=current_dir)

        # Initialize MLOps workflow orchestrator
        mlops_workflow = MLOpsWorkflow(claude_analyzer, base_dir=current_dir)

        # Initialize Kubernetes client (moved from nested function)
        v1 = None
        try:
            config.load_kube_config()
            v1 = client.CoreV1Api()
            logger.info("Kubernetes config loaded successfully.")
        except config.ConfigException:
            logger.warning("Failed to load kubeconfig: Invalid kube-config file. No configuration found.")
            try:
                config.load_incluster_config()
                v1 = client.CoreV1Api()
                logger.info("Kubernetes in-cluster config loaded successfully.")
            except config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                logger.warning("Kubernetes API is not available. MCP server will start with limited functionality.")
                return {"status": "error", "message": "Failed to load Kubernetes config.", "initialized": False}
        
        # Initialize storage backend based on Kubernetes availability
        if v1:
            storage_backend = KubernetesPvcBackend(k8s_core_v1=v1)  # Pass the k8s client
        else:
            raise RuntimeError("Kubernetes storage backend is required but Kubernetes API is not available.")

        # Initialize artifact manager (now in correct scope)
        artifact_manager = ArtifactManager(base_dir=current_dir, storage_backend=storage_backend)

        # Initialize workflow orchestrator
        workflow_orchestrator = WorkflowOrchestrator(
            mlops_workflow=mlops_workflow,
            k8s_debugger=k8s_debugger,
            artifact_manager=artifact_manager
        )

        # Create FastMCP server instance
        mcp = FastMCP("kubernetes-debugger")

        # Register tools
        @mcp.tool()
        def get_pods(namespace: str = "default", label_selector: str = "", field_selector: str = ""):
            return k8s_debugger.get_pods({"namespace": namespace, "label_selector": label_selector, "field_selector": field_selector})

        @mcp.tool()
        def describe_pod(name: str, namespace: str = "default"):
            return k8s_debugger.describe_pod({"name": name, "namespace": namespace})

        @mcp.tool()
        def get_pod_logs(name: str, namespace: str = "default", container: str = None, 
                         tail_lines: int = 100, previous: bool = False, analyze: bool = False):
            return k8s_debugger.get_pod_logs({"name": name, "namespace": namespace, "container": container, "tail_lines": tail_lines, "previous": previous, "analyze": analyze})

        @mcp.tool()
        def get_nodes(label_selector: str = "", field_selector: str = ""):
            return k8s_debugger.get_nodes({"label_selector": label_selector, "field_selector": field_selector})

        @mcp.tool()
        def describe_node(name: str):
            return k8s_debugger.describe_node({"name": name})

        @mcp.tool()
        def check_node_pressure(name: str):
            return k8s_debugger.check_node_pressure({"name": name})

        @mcp.tool()
        def get_deployments(namespace: str = "default", all_namespaces: bool = False, label_selector: str = ""):
            return k8s_debugger.get_deployments({"namespace": namespace, "all_namespaces": all_namespaces, "label_selector": label_selector})

        @mcp.tool()
        def describe_deployment(name: str, namespace: str = "default"):
            return k8s_debugger.describe_deployment({"name": name, "namespace": namespace})

        @mcp.tool()
        def scale_deployment(name: str, replicas: int, namespace: str = "default"):
            return k8s_debugger.scale_deployment({"name": name, "replicas": replicas, "namespace": namespace})

        @mcp.tool()
        def diagnose_cluster():
            return k8s_debugger.diagnose_cluster({})

        @mcp.tool()
        def analyze_resource(resource_type: str, resource_data: dict, events: list = None):
            return k8s_debugger.analyze_resource({"resource_type": resource_type, "resource_data": resource_data, "events": events or []})

        @mcp.tool()
        def recommend_action(resource_type: str, resource_data: dict, events: list = None, 
                           logs: dict = None, issue: str = None):
            return k8s_debugger.recommend_action({"resource_type": resource_type, "resource_data": resource_data, "events": events or [], "logs": logs or {}, "issue": issue})

        @mcp.tool()
        def apply_yaml(yaml_content: str, dry_run: bool = False, server_side: bool = False, force: bool = False):
            return k8s_debugger.apply_yaml({"yaml": yaml_content, "dry_run": dry_run, "server_side": server_side, "force": force})

        @mcp.tool()
        def delete_k8s_resource(kind: str, name: str, namespace: str = "default"):
            return k8s_debugger.delete_resource({"kind": kind, "name": name, "namespace": namespace})

        @mcp.tool()
        def apply_k8s_yaml_from_frontend(yaml_content: str, dry_run: bool = False, server_side: bool = False, force: bool = False):
            logger.info(f"Received YAML from frontend for application: {yaml_content[:100]}...")
            return k8s_debugger.apply_yaml({"yaml": yaml_content, "dry_run": dry_run, "server_side": server_side, "force": force})

        # --- DVC tools ---
        @mcp.tool()
        def dvc_init():
            return dvc_manager.init()

        @mcp.tool()
        def dvc_add(path: str):
            return dvc_manager.add(path)

        @mcp.tool()
        def dvc_push():
            return dvc_manager.push()

        @mcp.tool()
        def dvc_pull():
            return dvc_manager.pull()

        @mcp.tool()
        def dvc_status():
            return dvc_manager.status()

        @mcp.tool()
        def dvc_repro():
            return dvc_manager.repro()

        # --- High-level MLOps workflow tool ---
        @mcp.tool()
        def mlops_generate_code(prompt: str, job_name: str = None):
            job_dir = mlops_workflow.generate_job_dir(job_name)
            return mlops_workflow.generate_code_and_configs(prompt, job_dir)

        @mcp.tool()
        def mlops_list_jobs():
            return mlops_workflow.list_jobs()

        @mcp.tool()
        def mlops_get_job_files(job_id: str):
            return mlops_workflow.get_job_files(job_id)

        @mcp.tool()
        def mlops_list_experiments():
            return mlops_workflow.list_experiments()

        # --- Model Registry Tools ---
        @mcp.tool()
        def mlops_register_model(model_id: str, job_id: str, model_path: str, metrics: Dict[str, Any] = None, hyperparameters: Dict[str, Any] = None, version: str = None, metadata: Dict[str, Any] = None):
            return mlops_workflow.model_registry.register_model(
                model_id=model_id,
                job_id=job_id,
                model_path=model_path,
                metrics=metrics if metrics is not None else {},
                hyperparameters=hyperparameters if hyperparameters is not None else {},
                version=version,
                metadata=metadata if metadata is not None else {}
            )

        @mcp.tool()
        def mlops_list_registered_models():
            return mlops_workflow.model_registry.list_models()

        @mcp.tool()
        def mlops_get_model_details(model_id: str):
            return mlops_workflow.model_registry.get_model_details(model_id)

        @mcp.tool()
        def mlops_update_model_status(model_id: str, new_status: str):
            return mlops_workflow.model_registry.update_model_status(model_id, new_status)

        @mcp.tool()
        def mlops_build_image(job_id: str, dockerfile: str, image_tag: str):
            job_dir = os.path.join(current_dir, "mlops_jobs", job_id)
            return mlops_workflow.build_image(job_dir, dockerfile, image_tag)

        @mcp.tool()
        def mlops_push_image(image_tag: str, job_id: str = None):
            job_dir = os.path.join(current_dir, "mlops_jobs", job_id) if job_id else None
            return mlops_workflow.push_image(image_tag, job_dir=job_dir)

        @mcp.tool()
        def mlops_manage_images(job_id: str, training_tag: str = None, inference_tag: str = None):
            return mlops_workflow.manage_images(job_id, training_tag, inference_tag)

        # --- Artifact Management Tools ---
        @mcp.tool()
        def mlops_extract_model(job_id: str, pod_name: str, namespace: str = "default",
                              container: str = None, model_path: str = "/app/model"):
            return artifact_manager.extract_model_from_pod(
                job_id=job_id,
                pod_name=pod_name,
                namespace=namespace,
                container=container,
                model_path=model_path
            )

        @mcp.tool()
        def mlops_copy_model(job_id: str, pod_name: str, namespace: str = "default",
                           container: str = None, model_path: str = "/app/model"):
            return artifact_manager.copy_model_to_pod(
                job_id=job_id,
                pod_name=pod_name,
                namespace=namespace,
                container=container,
                model_path=model_path
            )

        @mcp.tool()
        def mlops_create_model_storage(job_id: str, namespace: str = "default",
                                     storage_class: str = "standard", size: str = "1Gi"):
            return artifact_manager.create_model_pvc(
                job_id=job_id,
                namespace=namespace,
                storage_class=storage_class,
                size=size
            )

        @mcp.tool()
        def mlops_cleanup_artifacts(job_id: str):
            return artifact_manager.cleanup_artifacts(job_id)

        # --- Inference Service Management Tools ---
        @mcp.tool()
        def mlops_deploy_inference(job_id: str, image_tag: str, namespace: str = "default",
                                 replicas: int = 1, resource_requests: dict = None,
                                 resource_limits: dict = None):
            return mlops_workflow.deploy_inference(
                job_id=job_id,
                image_tag=image_tag,
                namespace=namespace,
                replicas=replicas,
                resource_requests=resource_requests,
                resource_limits=resource_limits
            )

        @mcp.tool()
        def mlops_update_inference(job_id: str, namespace: str = "default",
                                 image_tag: str = None, replicas: int = None):
            return mlops_workflow.update_inference(
                job_id=job_id,
                namespace=namespace,
                image_tag=image_tag,
                replicas=replicas
            )

        @mcp.tool()
        def mlops_get_inference_status(job_id: str, namespace: str = "default"):
            return mlops_workflow.get_inference_status(
                job_id=job_id,
                namespace=namespace
            )

        @mcp.tool()
        def mlops_get_inference_metrics(job_id: str, namespace: str = "default"):
            return k8s_debugger.get_inference_metrics({"job_id": job_id, "namespace": namespace})

        @mcp.tool()
        def mlops_check_model_health(job_id: str, namespace: str = "default"):
            metrics_result = k8s_debugger.get_inference_metrics({"job_id": job_id, "namespace": namespace})
            if metrics_result["status"] != "success":
                return metrics_result # Propagate error
            metrics = metrics_result["metrics"]
            drift_detected = metrics_result.get("drift_detected", False)
            health_status = "Healthy"
            recommendations = []
            if drift_detected:
                health_status = "Needs Attention"
                recommendations.append("Investigate data/concept drift. Consider retraining the model with new data.")
            if metrics.get("error_rate", 0) > 0.05:
                health_status = "Needs Attention"
                recommendations.append(f"High error rate ({metrics['error_rate']*100:.2f}%). Check inference logs for errors.")
            if metrics.get("prediction_latency_ms_avg", 0) > 100:
                health_status = "Warning"
                recommendations.append(f"High prediction latency ({metrics['prediction_latency_ms_avg']}ms). Consider optimizing the model or scaling resources.")
            return {"status": "success", "job_id": job_id, "namespace": namespace, "health_status": health_status, "metrics": metrics, "drift_detected": drift_detected, "recommendations": recommendations, "message": f"Model health check for {job_id} completed."}

        # --- High-level Workflow Tools ---
        @mcp.tool()
        def mlops_execute_workflow(workflow_params: Dict[str, Any]):
            return workflow_orchestrator.execute_workflow(workflow_params)

        @mcp.tool()
        def mlops_run_inference(job_id: str, data: Any):
            return workflow_orchestrator.run_inference(job_id, data)

        # --- LLM and Recommendation Model Tools ---
        @mcp.tool()
        def mlops_create_recommendation_model(task_description: str, data_format: str, features: List[str]):
            return mlops_workflow.create_recommendation_model(
                task_description=task_description,
                data_format=data_format,
                features=features
            )

        @mcp.tool()
        def mlops_finetune_llm(task_description: str, data_format: str, framework: str = "pytorch"):
            return mlops_workflow.finetune_llm(
                task_description=task_description,
                data_format=data_format,
                framework=framework
            )

        @mcp.tool()
        def mlops_deploy_registered_model(model_id: str, namespace: str = "default", replicas: int = 1):
            model_details_result = mlops_workflow.model_registry.get_model_details(model_id)
            if model_details_result["status"] != "success":
                return model_details_result
            model_details = model_details_result["model_details"]
            job_id = model_details.get("job_id")
            model_path = model_details.get("model_path")
            model_name = model_details.get("model_name", "GenericModel")
            model_class_code = model_details.get("model_class_code", """class GenericModel(torch.nn.Module):\n    def __init__(self):\n        super(GenericModel, self).__init__()\n        self.linear = torch.nn.Linear(784, 10)\n    def forward(self, x):\n        return torch.nn.functional.log_softmax(x.view(x.size(0), -1), dim=1)""")
            model_file_name = os.path.basename(model_path) if model_path else "mnist_cnn.pt"
            model_load_logic = model_details.get("model_load_logic", f"model = {model_name}()\nmodel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))")
            transform_code = model_details.get("transform_code", "transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])")
            inference_code = workflow_orchestrator._generate_inference_script(
                model_name=model_name,
                model_class_code=model_class_code,
                model_load_logic=model_load_logic,
                transform_code=transform_code,
                model_file_name=model_file_name
            )
            if not job_id:
                job_id = f"inference-{model_id}-{int(time.time())}"
                logger.warning(f"Job ID not found for model {model_id}. Generating a new one: {job_id}")
            return mlops_workflow.deploy_inference_service_with_code(
                job_id=job_id,
                model_id=model_id,
                inference_code=inference_code,
                namespace=namespace,
                replicas=replicas
            )

        @mcp.tool()
        def mlops_deploy_ml_stack(params: Dict[str, Any]):
            return k8s_debugger.deploy_ml_stack(params)

        @mcp.tool()
        def mlops_cleanup_job(job_id: str, namespace: str = "default"):
            return k8s_debugger.cleanup_ml_job({"job_id": job_id, "namespace": namespace})

        @mcp.tool()
        def create_pvc(name: str, size: str = "1Gi", namespace: str = "default", storage_class: str = None):
            if not storage_class:
                sc_result = k8s_debugger.get_storage_classes({})
                if sc_result["status"] == "success" and sc_result["storage_classes"]:
                    if "standard" in sc_result["storage_classes"]:
                        storage_class = "standard"
                    elif sc_result["storage_classes"]:
                        storage_class = sc_result["storage_classes"][0]
                    else:
                        return {"status": "error", "message": "No StorageClasses found in the cluster."}
                else:
                    return {"status": "error", "message": "Failed to retrieve StorageClasses."}
            pvc_yaml = f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {name}
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: {storage_class}
  resources:
    requests:
      storage: {size}
"""
            return k8s_debugger.apply_yaml({"yaml": pvc_yaml})

        @mcp.tool()
        def fix_mnist_deployment():
            pvc_yaml = """
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mnist-model-storage
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
"""
            
            train_code = """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

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

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)

    model = MNISTNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    for epoch in range(1, 2):
        train(model, device, train_loader, optimizer, epoch)

    os.makedirs('/mnt/model', exist_ok=True)
    torch.save(model.state_dict(), '/mnt/model/mnist_cnn.pt')

if __name__ == '__main__':
    main()
"""
            inference_code = textwrap.dedent("""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os

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

def load_model(model_path):
    model = MNISTNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image_bytes):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()

if __name__ == '__main__':
    model_path = '/mnt/model/mnist_cnn.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit(1)

    model = load_model(model_path)
    print("Model loaded successfully.")

    from flask import Flask, request, jsonify
    import base64
    import numpy as np

    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'model': 'MNIST CNN'})

    @app.route('/predict', methods=['POST'])
    def predict_route():
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'No image provided in JSON body'}), 400

            img_data = base64.b64decode(data['image'])
            prediction = predict(model, img_data)

            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    print('Starting MNIST inference server...')
    print('Model architecture:', model)
    print('Server ready at http://0.0.0.0:8080')
    app.run(host='0.0.0.0', port=8080, debug=False)
""") # Closing parenthesis for textwrap.dedent
            config_map_train = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: mnist-train-code
  namespace: default
data:
  train.py: |
{train_code}
"""
            config_map_inference = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: mnist-inference-code
  namespace: default
data:
  inference.py: |
{inference_code}
"""
            k8s_debugger.apply_yaml({"yaml": pvc_yaml})
            k8s_debugger.apply_yaml({"yaml": config_map_train})
            k8s_debugger.apply_yaml({"yaml": config_map_inference})

        mcp_instance = mcp # Assign the created MCP instance to the global variable
        logger.info("MCP server initialized successfully.")
        return {"status": "success", "message": "MCP server initialized successfully.", "initialized": True}

    except Exception as e:
        logger.error(f"Failed to initialize MCP server: {e}")
        return {"status": "error", "message": f"Failed to initialize MCP server: {str(e)}", "initialized": False}

# Expose the MCP server instance for import
# mcp = create_mcp_server() # This will now be called dynamically

# --- Flask API Endpoints (for direct frontend interaction) ---
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
import base64

flask_app = Flask(__name__)
CORS(flask_app) # Enable CORS for all origins on all routes

@flask_app.route('/health', methods=['GET'])
def health():
    # This health endpoint can be used by the frontend to check if the backend is alive
    # It should reflect the state of the deployed server, not local Kubernetes connection
    # For now, it will report healthy if the Flask app itself is running
    return jsonify({'status': 'healthy', 'model': 'MNIST CNN'})

@flask_app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided in JSON body'}), 400

        img_data = base64.b64decode(data['image'])
        # In a real deployed scenario, this would trigger an inference call
        # to the actual deployed model, possibly via mlops_run_inference tool
        # For now, we simulate a prediction.
        
        # This part assumes a model is available locally, which won't be the case in K8s pod
        # We should remove this direct model loading/prediction if server runs in K8s
        # For now, let's return a dummy prediction to allow frontend testing.
        dummy_prediction = 7 # Just a dummy prediction
        dummy_confidence = 0.95 # Dummy confidence
        return jsonify({'prediction': dummy_prediction, 'confidence': dummy_confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/connect-kubeconfig', methods=['POST'])
def connect_kubeconfig():
    global mcp_instance
    if mcp_instance:
        return jsonify({'status': 'success', 'message': 'MCP server already connected.'}), 200

    try:
        data = request.get_json()
        kubeconfig_content = data.get('kubeconfig')
        
        if not kubeconfig_content:
            return jsonify({'error': 'No kubeconfig content provided'}), 400

        result = initialize_mcp_server(kubeconfig_content=kubeconfig_content)
        if result["status"] == "success":
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error in /api/connect-kubeconfig: {e}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/mcp-status', methods=['GET'])
def get_mcp_status():
    global mcp_instance, k8s_client_initialized
    status = {
        "mcp_initialized": mcp_instance is not None,
        "k8s_client_initialized": k8s_client_initialized
    }
    return jsonify(status), 200

# Helper function to check if MCP is initialized before calling tools
def require_mcp_initialized(func):
    def wrapper(*args, **kwargs):
        if mcp_instance is None:
            return jsonify({'error': 'MCP server not initialized. Please connect kubeconfig first.'}), 400
        return func(*args, **kwargs)
    return wrapper

@flask_app.route('/api/kubernetes/pods', methods=['GET'])
@require_mcp_initialized
def get_kubernetes_pods():
    try:
        namespace = request.args.get('namespace', 'default')
        label_selector = request.args.get('label_selector', '')
        field_selector = request.args.get('field_selector', '')
        result = mcp_instance.get_pods(namespace=namespace, label_selector=label_selector, field_selector=field_selector)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/kubernetes/pods: {e}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/mlops/jobs', methods=['GET'])
@require_mcp_initialized
def list_mlops_jobs():
    try:
        result = mcp_instance.mlops_list_jobs()
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/mlops/jobs: {e}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/mlops/models', methods=['GET'])
@require_mcp_initialized
def list_mlops_models():
    try:
        result = mcp_instance.mlops_list_registered_models()
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/mlops/models: {e}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/mlops/model/<model_id>', methods=['GET'])
@require_mcp_initialized
def get_mlops_model_details(model_id):
    try:
        result = mcp_instance.mlops_get_model_details(model_id=model_id)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/mlops/model/{model_id}: {e}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/mlops/deploy-model', methods=['POST'])
@require_mcp_initialized
def deploy_mlops_model():
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        namespace = data.get('namespace', 'default')
        replicas = data.get('replicas', 1)

        if not model_id:
            return jsonify({'error': 'model_id is required'}), 400

        result = mcp_instance.mlops_deploy_registered_model(model_id=model_id, namespace=namespace, replicas=replicas)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/mlops/deploy-model: {e}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/mlops/run-inference', methods=['POST'])
@require_mcp_initialized
def run_mlops_inference():
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        inference_data = data.get('data')

        if not job_id or not inference_data:
            return jsonify({'error': 'job_id and data are required'}), 400

        result = mcp_instance.mlops_run_inference(job_id=job_id, data=inference_data)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/mlops/run-inference: {e}")
        return jsonify({'error': str(e)}), 500

@flask_app.route('/api/call-tool', methods=['POST'])
@require_mcp_initialized
def call_mcp_tool():
    try:
        data = request.get_json()
        tool_name = data.get('tool_name')
        tool_args = data.get('tool_args', {})

        if not tool_name:
            return jsonify({'error': 'tool_name is required'}), 400
        
        # Check if the tool exists and is callable via MCP
        if not hasattr(mcp_instance, tool_name) or not callable(getattr(mcp_instance, tool_name)):
            return jsonify({'error': f'Tool "{tool_name}" not found or not callable.'}), 404

        # Call the MCP tool dynamically
        tool_function = getattr(mcp_instance, tool_name)
        result = tool_function(**tool_args)
        
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error calling MCP tool '{tool_name}': {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # This part remains for local testing of the Flask app if needed.
    # In a real K8s deployment, the FastMCP server handles routing to Flask app.
    import uvicorn # Added uvicorn import
    uvicorn.run(flask_app, host="0.0.0.0", port=8080, reload=False)
