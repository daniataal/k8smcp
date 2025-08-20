import os
import logging
import uuid
import time
import subprocess
import json
from typing import Optional, Dict, List, Any, Union, Tuple
from claude_analyzer import ClaudeAnalyzer
from artifact_manager import ArtifactManager
from inference_manager import InferenceManager
from kubernetes import client
from llm_trainer import LLMTrainer
from model_registry import ModelRegistry # Import ModelRegistry
from model_definitions import MODEL_CONFIGURATIONS # Import model configurations
import yaml # Added for deploy_training_job
from storage_backends import KubernetesPvcBackend # Import KubernetesPvcBackend
import io
import base64
import numpy as np # Import numpy
import time # Import time for latency calculation
from prometheus_client import start_http_server, Counter, Histogram # Import Prometheus client
import collections # Import collections for deque

logger = logging.getLogger(__name__)

class MLOpsWorkflow:
    def __init__(self, claude_analyzer: ClaudeAnalyzer, base_dir: str):
        self.claude_analyzer = claude_analyzer
        self.base_dir = os.path.join(base_dir, "mlops_jobs")
        os.makedirs(self.base_dir, exist_ok=True)
        self.registry = os.environ.get("CONTAINER_REGISTRY", "localhost:5000")
        self.podman_cmd = "podman"  # Can be configured via env var if needed
        
        # Initialize Kubernetes API clients for storage backend
        k8s_core_v1_api = client.CoreV1Api()

        self.artifact_manager = ArtifactManager(base_dir, storage_backend=KubernetesPvcBackend(k8s_core_v1=k8s_core_v1_api))
        self.inference_manager = InferenceManager(
            k8s_apps_v1=client.AppsV1Api(),
            k8s_core_v1=client.CoreV1Api()
        )
        self.llm_trainer = LLMTrainer(api_key=claude_analyzer.api_key)
        self.model_registry = ModelRegistry(base_dir) # Initialize ModelRegistry

    def generate_job_dir(self, job_name: str = None) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        job_id = job_name or f"job-{timestamp}-{unique_id}"
        job_dir = os.path.join(self.base_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    def generate_code_and_configs(self, model_type: str, job_dir: str) -> dict:
        """
        Generate ML training/inference code, Dockerfiles, and K8s YAMLs for a given model type.
        """
        if model_type not in MODEL_CONFIGURATIONS:
            return {"status": "error", "message": f"Unknown model type: {model_type}"}

        model_config = MODEL_CONFIGURATIONS[model_type]

        try:
            # Generate training code
            train_code = model_config["train_script_template"]

            # Generate inference code using the new _generate_inference_script function
            inference_code = self._generate_inference_script(
                model_name=model_config["model_name"],
                model_class_code=model_config["model_class_code"],
                model_load_logic=model_config["model_load_logic"],
                transform_code=model_config["transform_code"]
            )

            # Generate Dockerfiles
            dockerfile_train = model_config["dockerfile_train_template"]
            dockerfile_infer = model_config["dockerfile_infer_template"]
            
            # Generate DVC pipeline (dvc.yaml)
            dvc_yaml_content = self._generate_dvc_yaml(
                model_name=model_config["model_name"],
                data_path=model_config.get("data_path", "data"), # Default data path
                model_save_path=model_config.get("model_save_path", "/mnt/model/mnist_cnn.pt")
            )

            # Save files to job directory
            with open(os.path.join(job_dir, "train.py"), 'w') as f:
                f.write(train_code)
            with open(os.path.join(job_dir, "inference.py"), 'w') as f:
                f.write(inference_code)
            with open(os.path.join(job_dir, "Dockerfile.train"), 'w') as f:
                f.write(dockerfile_train)
            with open(os.path.join(job_dir, "Dockerfile.infer"), 'w') as f:
                f.write(dockerfile_infer)
            with open(os.path.join(job_dir, "dvc.yaml"), 'w') as f:
                f.write(dvc_yaml_content)

            logger.info(f"Generated code and configurations for job in {job_dir}")

            return {
                "status": "success",
                "message": f"Code and configurations generated successfully for {model_type}.",
                "job_dir": job_dir,
                "files": ["train.py", "inference.py", "Dockerfile.train", "Dockerfile.infer", "dvc.yaml"]
            }
        except Exception as e:
            logger.error(f"Error generating code for {model_type}: {e}")
            return {"status": "error", "message": f"Failed to generate code: {str(e)}"}

    def _generate_dvc_yaml(self, model_name: str, data_path: str, model_save_path: str) -> str:
        """
        Generates a basic dvc.yaml for a training pipeline.
        This assumes: 
        - A 'data' stage for pulling data (if data is DVC-tracked)
        - A 'train' stage for running the training script and tracking the model.
        """
        dvc_yaml_template = f"""
stages:
  train:
    cmd: python train.py
    deps:
      - {data_path} # Assuming data is DVC-tracked at this path
      - train.py
    outs:
      - {model_save_path} # DVC will track the model artifact
"""
        return dvc_yaml_template

    def _generate_inference_script(self, model_name: str, model_class_code: str, model_load_logic: str, transform_code: str, model_file_name: str = "model.pt") -> str:
        # Basic inference script template
        # This will be refined as we generalize more.
        inference_script_template = """
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
import time
from prometheus_client import start_http_server, Counter, Histogram
import collections

app = Flask(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'HTTP Request Latency', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total Predictions', ['model_name'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction Latency', ['model_name'])

# For simulated drift detection
PREDICTION_HISTORY = collections.deque(maxlen=100)

# Define model class
{model_class_code}

# Load the trained model
def load_model(model_path):
    {model_load_logic}
    return model

model_path = '/app/{model_file_name}'
if not os.path.exists(model_path):
    print(f"Error: Model not found at {{model_path}}")
    exit(1)

model = load_model(model_path)
print("Model loaded successfully.")

# Define transformation
transform = {transform_code}

@app.route('/health', methods=['GET'])
def health():
    REQUEST_COUNT.labels(method='GET', endpoint='/health').inc()
    return jsonify({{'status': 'healthy', 'model': '{model_name}'}})

@app.route('/predict', methods=['POST'])
def predict_route():
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({{'error': 'No image provided in JSON body'}}), 400

        img_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_data)).convert('L')
        img_tensor = transform(img).unsqueeze(0)

        prediction_start_time = time.time()
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True)
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities.max().item()
        prediction_latency = time.time() - prediction_start_time
        PREDICTION_LATENCY.labels(model_name='{model_name}').observe(prediction_latency)
        PREDICTION_COUNT.labels(model_name='{model_name}').inc()

        # Simulate data/concept drift detection
        PREDICTION_HISTORY.append({'confidence': confidence, 'prediction': pred.item()})
        if len(PREDICTION_HISTORY) == PREDICTION_HISTORY.maxlen:
            # Simple drift detection: check if average confidence drops below a threshold
            current_avg_confidence = sum([p['confidence'] for p in PREDICTION_HISTORY]) / len(PREDICTION_HISTORY)
            if current_avg_confidence < 0.6: # Example threshold
                print(f"[DRIFT_DETECTED] Average confidence dropped to {current_avg_confidence:.2f}")
                # Call the simulated alert function
                # This requires passing self.mlops down, or finding another way to access it
                # For now, just print and assume an external system would pick this up from logs.
                # In a real scenario, you would trigger an actual alert here.
                # Example: self.mlops._send_alert("drift_detection", "Model confidence drift detected", {"model_name": model_name, "avg_confidence": current_avg_confidence})
            # More sophisticated drift detection would compare distributions of inputs/outputs over time
        
        response = jsonify({{'prediction': pred.item(), 
            'confidence': round(confidence, 4),
            'probabilities': probabilities.squeeze().tolist()}})
        REQUEST_LATENCY.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)
        return response
    except Exception as e:
        REQUEST_LATENCY.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)
        return jsonify({{'error': str(e)}}), 500

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    return generate_latest(), 200, {{ 'Content-Type': 'text/plain' }}

if __name__ == '__main__':
    start_http_server(8000) # Start Prometheus metrics server on a different port
    app.run(host='0.0.0.0', port=8080, debug=False)
"""
        return inference_script_template.format(
            model_name=model_name,
            model_class_code=model_class_code,
            model_load_logic=model_load_logic,
            transform_code=transform_code,
            model_file_name=model_file_name
        )

    def _run_podman(self, args: List[str], cwd: Optional[str] = None, capture_output: bool = True) -> Dict:
        """Run a podman command and return the result"""
        cmd = [self.podman_cmd] + args
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                capture_output=capture_output,
                text=True
            )
            return {
                "status": "success",
                "stdout": proc.stdout if capture_output else None,
                "stderr": proc.stderr if capture_output else None
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": f"Podman command failed: {e.stderr}",
                "stderr": e.stderr,
                "stdout": e.stdout
            }

    def build_image(self, job_dir: str, dockerfile: str, image_name: str) -> Dict:
        """
        Build a container image using Podman.
        
        Args:
            job_dir: Directory containing the Dockerfile
            dockerfile: Name of the Dockerfile
            image_name: Name/tag for the image
            
        Returns:
            Dict with build status and details
        """
        if not os.path.exists(os.path.join(job_dir, dockerfile)):
            return {"status": "error", "message": f"Dockerfile {dockerfile} not found in {job_dir}"}

        # Add registry prefix if not already present
        if self.registry and ":" not in image_name:
            full_tag = f"{self.registry}/{image_name}"
        else:
            full_tag = image_name

        # Create build log file
        build_log = os.path.join(job_dir, "build.log")
        
        try:
            # Build the image
            with open(build_log, "w") as log:
                result = self._run_podman([
                    "build",
                    "-f", dockerfile,
                    "-t", full_tag,
                    "."
                ], cwd=job_dir)
                
                log.write(f"Build command output:\n{result.get('stdout', '')}\n")
                log.write(f"Build command errors:\n{result.get('stderr', '')}\n")

            if result["status"] == "success":
                # Get image details
                inspect_result = self._run_podman(["inspect", full_tag])
                if inspect_result["status"] == "success":
                    image_info = json.loads(inspect_result["stdout"])[0]
                else:
                    image_info = {}

                return {
                    "status": "success",
                    "image": full_tag,
                    "build_log": build_log,
                    "image_info": image_info
                }
            else:
                return {
                    "status": "error",
                    "message": "Build failed",
                    "build_log": build_log,
                    "details": result
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Build failed: {str(e)}",
                "build_log": build_log
            }

    def push_image(self, image_tag: str, job_dir: Optional[str] = None) -> Dict:
        """
        Push a container image to registry using Podman.
        
        Args:
            image_tag: Image tag to push
            job_dir: Optional directory for storing push logs
            
        Returns:
            Dict with push status and details
        """
        # Add registry prefix if not already present
        if self.registry and ":" not in image_tag:
            full_tag = f"{self.registry}/{image_tag}"
        else:
            full_tag = image_tag

        # Create push log file if job_dir provided
        push_log = None
        if job_dir:
            push_log = os.path.join(job_dir, "push.log")

        try:
            # Push the image
            result = self._run_podman(["push", full_tag])
            
            if push_log:
                with open(push_log, "w") as log:
                    log.write(f"Push command output:\n{result.get('stdout', '')}\n")
                    log.write(f"Push command errors:\n{result.get('stderr', '')}\n")

            if result["status"] == "success":
                response = {
                    "status": "success",
                    "image": full_tag,
                }
                if push_log:
                    response["push_log"] = push_log
                return response
            else:
                return {
                    "status": "error",
                    "message": "Push failed",
                    "push_log": push_log if push_log else None,
                    "details": result
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Push failed: {str(e)}",
                "push_log": push_log if push_log else None
            }

    def manage_images(self, job_id: str, training_tag: str = None, inference_tag: str = None) -> Dict:
        """
        Build and push both training and inference images for a job.
        
        Args:
            job_id: Job ID
            training_tag: Tag for training image (optional)
            inference_tag: Tag for inference image (optional)
            
        Returns:
            Dict with status and details of both images
        """
        job_dir = os.path.join(self.base_dir, job_id)
        if not os.path.exists(job_dir):
            return {"status": "error", "message": f"Job directory {job_dir} not found"}

        results = {
            "status": "success",
            "training": None,
            "inference": None
        }

        # Build and push training image if specified
        if training_tag:
            train_result = self.build_image(job_dir, "Dockerfile.train", training_tag)
            if train_result["status"] == "success":
                push_result = self.push_image(training_tag, job_dir)
                results["training"] = {
                    "build": train_result,
                    "push": push_result
                }
            else:
                results["status"] = "error"
                results["training"] = {"build": train_result}

        # Build and push inference image if specified
        if inference_tag:
            infer_result = self.build_image(job_dir, "Dockerfile.infer", inference_tag)
            if infer_result["status"] == "success":
                push_result = self.push_image(inference_tag, job_dir)
                results["inference"] = {
                    "build": infer_result,
                    "push": push_result
                }
            else:
                results["status"] = "error"
                results["inference"] = {"build": infer_result}

        return results

    def list_jobs(self) -> dict:
        """List all job directories and their files."""
        jobs = []
        for job_id in sorted(os.listdir(self.base_dir)):
            job_path = os.path.join(self.base_dir, job_id)
            if os.path.isdir(job_path):
                files = os.listdir(job_path)
                jobs.append({"job_id": job_id, "files": files})
        return {"status": "success", "jobs": jobs}

    def get_job_files(self, job_id: str) -> dict:
        """Get all files and their contents for a given job."""
        job_path = os.path.join(self.base_dir, job_id)
        if not os.path.isdir(job_path):
            return {"status": "error", "message": f"Job {job_id} not found"}
        files = {}
        for fname in os.listdir(job_path):
            fpath = os.path.join(job_path, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    files[fname] = f.read()
        return {"status": "success", "job_id": job_id, "files": files}

    def _log_experiment_result(self, job_id: str, status: str, workflow_params: Dict[str, Any], 
                                details: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None, 
                                hyperparameters: Optional[Dict[str, Any]] = None, 
                                model_id: Optional[str] = None) -> None:
        """
        Logs the result of an MLOps experiment to a JSON file within the job directory.
        This function is intended for internal use to keep a structured record of workflow executions.
        """
        log_file_path = os.path.join(self.base_dir, "mlops_jobs", job_id, "experiment_log.json")
        
        # Ensure the log file exists and is a valid JSON array
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                json.dump([], f)

        try:
            with open(log_file_path, "r+") as f:
                logs = json.load(f)
                
                experiment_entry = {
                    "timestamp": time.time(),
                    "job_id": job_id,
                    "model_id": model_id, # Include model_id
                    "status": status,
                    "workflow_params": workflow_params, # Log the full workflow parameters
                    "details": details,
                    "metrics": metrics if metrics is not None else {},
                    "hyperparameters": hyperparameters if hyperparameters is not None else {}
                }
                logs.append(experiment_entry)
                
                f.seek(0)
                json.dump(logs, f, indent=4)
                f.truncate()
            logger.info(f"Experiment result logged for job {job_id} with status {status}.")
        except Exception as e:
            logger.error(f"Error logging experiment result for job {job_id}: {e}")

    def _send_alert(self, alert_type: str, message: str, details: Dict[str, Any] = None) -> None:
        """
        Simulates sending an alert to an external system.
        In a real system, this would integrate with PagerDuty, Slack, email, etc.
        """
        alert_payload = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": time.time(),
            "details": details if details is not None else {}
        }
        logger.critical(f"[ALERT_SIMULATION] Sending alert: {alert_payload}")
        # Here you would typically integrate with an alerting API

    def deploy_training_job(self, job_id: str, image_tag: str, namespace: str = "default",
                            resource_requests: Dict[str, str] = None,
                            resource_limits: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Deploys a Kubernetes Job for training and monitors its completion.
        Includes a simulated retry mechanism for initial deployment attempts.
        """
        if not resource_requests:
            resource_requests = {"cpu": "1", "memory": "2Gi"}
        if not resource_limits:
            resource_limits = {"cpu": "4", "memory": "8Gi"}

        # Ensure PVC exists for model storage
        storage_result = self.artifact_manager.create_model_pvc(job_id, namespace)
        if storage_result["status"] != "success":
            return storage_result

        job_name = f"train-{job_id}"
        job_yaml = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "labels": {"job-id": job_id, "type": "training"}
            },
            "spec": {
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "training",
                            "image": image_tag,
                            "command": ["dvc"],
                            "args": ["repro"],
                            "resources": {
                                "requests": resource_requests,
                                "limits": resource_limits
                            },
                            "volumeMounts": [{
                                "name": "model-storage",
                                "mountPath": "/mnt/model"
                            }]
                        }],
                        "volumes": [{
                            "name": "model-storage",
                            "persistentVolumeClaim": {
                                "claimName": f"model-{job_id}"
                            }
                        }]
                    }
                },
                "backoffLimit": 4 # Retry up to 4 times on failure
            }
        }

        # Simulated retry logic for deployment (conceptual, for demonstration)
        max_retries = 3
        retry_delay = 5 # seconds
        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Deploying training job {job_name}...")
            deploy_result = self.k8s.apply_yaml({"yaml": yaml.dump(job_yaml)})
            
            if deploy_result["status"] == "success":
                logger.info(f"Deployment of {job_name} successful.")
                break # Exit retry loop on success
            else:
                logger.warning(f"Deployment of {job_name} failed on attempt {attempt + 1}: {deploy_result.get('message', 'Unknown error')}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt)) # Exponential backoff
                else:
                    return {"status": "error", "message": f"Failed to deploy training job after {max_retries} attempts: {deploy_result.get('message', 'Unknown error')}"}

        # If deployment failed after all retries, return error
        if deploy_result["status"] != "success":
            return deploy_result

        logger.info(f"Monitoring training job {job_name}...")
        max_wait_time = 900  # 15 minutes
        start_time = time.time()

        while True:
            try:
                job_status = self.k8s.batch_v1.read_namespaced_job_status(name=job_name, namespace=namespace)
                if job_status.status.succeeded is not None and job_status.status.succeeded > 0:
                    logger.info("Training job succeeded.")
                    return {"status": "success", "message": f"Training job {job_name} completed successfully.", "job_name": job_name}
                elif job_status.status.failed is not None and job_status.status.failed > 0:
                    # Attempt to get logs for debugging failed jobs
                    pod_selector = f"job-name={job_name}"
                    pods_result = self.k8s.get_pods({"namespace": namespace, "label_selector": pod_selector})
                    error_logs = "No logs retrieved."
                    if pods_result["status"] == "success" and pods_result["pods"]:
                        # Get logs from the first pod associated with the job
                        first_pod_name = pods_result["pods"][0]["name"]
                        logs_result = self.k8s.get_pod_logs({"name": first_pod_name, "namespace": namespace})
                        if logs_result["status"] == "success":
                            error_logs = logs_result["logs"]
                        else:
                            error_logs = f"Failed to retrieve logs: {logs_result.get('message','Unknown error')}"

                    return {"status": "error", "message": f"Training job {job_name} failed. Logs: {error_logs}"}
                
                if time.time() - start_time > max_wait_time:
                    return {"status": "error", "message": f"Training job {job_name} timed out after {max_wait_time} seconds."}

            except client.ApiException as e:
                if e.status == 404:
                    logger.info(f"Training job {job_name} not found yet...")
                else:
                    logger.error(f"Error checking training job status for {job_name}: {e}")
                    return {"status": "error", "message": f"Error checking training job status: {str(e)}"}
            except Exception as e:
                logger.error(f"Unexpected error while monitoring training job {job_name}: {e}")
                return {"status": "error", "message": f"Unexpected error monitoring training job: {str(e)}"}

            time.sleep(10) # Wait for 10 seconds before polling again

    def schedule_workflow(self, workflow_params: Dict[str, Any], schedule_interval: str) -> Dict[str, Any]:
        """
        Simulates scheduling an MLOps workflow for periodic execution.
        In a real platform, this would integrate with a Kubernetes-native scheduler (e.g., Argo Workflows, KubeFlow Pipelines) 
        or external cron-like services.
        """
        logger.info(f"[SCHEDULE_SIMULATION] Workflow for model type '{workflow_params.get('model_type', 'unknown')}' scheduled to run {schedule_interval}.")
        logger.info(f"[SCHEDULE_SIMULATION] Workflow parameters: {workflow_params}")
        return {"status": "success", "message": f"Workflow scheduled successfully for {schedule_interval}. (Simulated)"}

    def trigger_workflow_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates an event-driven trigger for an MLOps workflow.
        In a real system, this would be an actual event listener (e.g., Kafka consumer, webhook receiver)
        that processes events and initiates workflows.
        """
        logger.info(f"[EVENT_TRIGGER_SIMULATION] Event of type '{event_type}' received.")
        logger.info(f"[EVENT_TRIGGER_SIMULATION] Event data: {event_data}")
        
        # In a real scenario, based on event_type and event_data, a specific workflow
        # would be looked up and executed (e.g., retraining on new data arrival).
        # For now, we just log the event.
        return {"status": "success", "message": f"Event of type '{event_type}' triggered workflow. (Simulated)", "event_data": event_data}

    def ingest_data(self, data_source_type: str, connection_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates ingesting data from various sources.
        In a real system, this would connect to databases, data lakes (S3, GCS), streaming platforms (Kafka), etc.
        """
        logger.info(f"[DATA_INGESTION_SIMULATION] Attempting to ingest data from {data_source_type}.")
        logger.info(f"[DATA_INGESTION_SIMULATION] Connection details: {connection_details}")
        
        # Simulate success/failure based on some conditions or just always succeed for now
        if "error" in connection_details.get("type", "").lower():
            return {"status": "error", "message": f"Simulated ingestion failure from {data_source_type}."}

        return {"status": "success", "message": f"Data ingestion from {data_source_type} simulated successfully.", "ingested_dat-info": {"source_type": data_source_type, "details": connection_details}}

    def get_features(self, feature_names: List[str], entity_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simulates retrieving features from a feature store.
        In a real system, this would query a dedicated feature store (e.g., Feast, Tecton) 
        to get features for training or inference.
        """
        logger.info(f"[FEATURE_STORE_SIMULATION] Retrieving features: {feature_names}")
        if entity_ids:
            logger.info(f"[FEATURE_STORE_SIMULATION] For entities: {entity_ids}")
        
        # Simulate returning dummy feature data
        dummy_features = {"feature_1": [0.1, 0.2], "feature_2": ['A', 'B']}
        return {"status": "success", "message": "Features retrieved successfully. (Simulated)", "features": dummy_features}

    def put_features(self, features_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates storing features into a feature store.
        In a real system, this would write processed features to a feature store.
        """
        logger.info(f"[FEATURE_STORE_SIMULATION] Storing features: {list(features_data.keys())}")
        logger.info(f"[FEATURE_STORE_SIMULATION] Data preview: {list(features_data.values())[0] if features_data else 'N/A'}")
        
        return {"status": "success", "message": "Features stored successfully. (Simulated)"}

    def list_experiments(self) -> Dict[str, Any]:
        """Lists all logged MLOps experiments across all job directories."""
        all_experiments = []
        for job_id in os.listdir(self.base_dir):
            job_dir = os.path.join(self.base_dir, job_id)
            log_file_path = os.path.join(job_dir, "experiment_log.json")
            
            if os.path.isdir(job_dir) and os.path.exists(log_file_path):
                try:
                    with open(log_file_path, "r") as f:
                        experiments_in_job = json.load(f)
                        all_experiments.extend(experiments_in_job)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted experiment log file found: {log_file_path}. Skipping.")
                except Exception as e:
                    logger.error(f"Error reading experiment log {log_file_path}: {e}")
        
        # Sort experiments by timestamp, most recent first
        all_experiments.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        return {"status": "success", "experiments": all_experiments}

    def manage_training_artifacts(self, job_id: str, pod_name: str,
                                namespace: str = "default") -> Dict[str, Any]:
        """
        Manage artifacts for a training job:
        1. Create PVC if needed
        2. Extract model from training pod
        3. Clean up temporary files
        """
        # Create storage
        storage_result = self.artifact_manager.create_model_pvc(job_id, namespace)
        if storage_result["status"] != "success":
            return storage_result

        # Extract model
        extract_result = self.artifact_manager.extract_model_from_pod(
            job_id, pod_name, namespace
        )
        if extract_result["status"] != "success":
            return extract_result

        return {
            "status": "success",
            "storage": storage_result,
            "artifacts": extract_result
        }

    def setup_inference_artifacts(self, job_id: str, pod_name: str,
                                namespace: str = "default") -> Dict[str, Any]:
        """
        Set up artifacts for inference:
        1. Copy model to inference pod
        2. Verify copy success
        """
        return self.artifact_manager.copy_model_to_pod(
            job_id, pod_name, namespace
        )

    def get_volume_config(self, job_id: str) -> Dict[str, Any]:
        """Get volume configuration for pods"""
        return self.artifact_manager.get_volume_mounts(job_id)

    def deploy_inference(self, job_id: str, image_tag: str, namespace: str = "default",
                        replicas: int = 1, resource_requests: Dict[str, str] = None,
                        resource_limits: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Deploy an inference service for a trained model.
        
        Args:
            job_id: Job ID
            image_tag: Inference image tag
            namespace: Kubernetes namespace
            replicas: Number of replicas
            resource_requests: Resource requests for the pod
            resource_limits: Resource limits for the pod
        """
        # First, ensure model artifacts are properly set up
        storage_result = self.artifact_manager.create_model_pvc(job_id, namespace)
        if storage_result["status"] != "success":
            return storage_result

        # Deploy the inference service
        return self.inference_manager.deploy_inference_service(
            job_id=job_id,
            image=image_tag,
            namespace=namespace,
            replicas=replicas,
            resource_requests=resource_requests,
            resource_limits=resource_limits
        )

    def update_inference(self, job_id: str, namespace: str = "default",
                        image_tag: Optional[str] = None,
                        replicas: Optional[int] = None) -> Dict[str, Any]:
        """Update an existing inference service."""
        return self.inference_manager.update_inference_service(
            job_id=job_id,
            namespace=namespace,
            image=image_tag,
            replicas=replicas
        )

    def get_inference_status(self, job_id: str, 
                           namespace: str = "default") -> Dict[str, Any]:
        """Get status of an inference service."""
        return self.inference_manager.get_inference_status(
            job_id=job_id,
            namespace=namespace
        )

    def deploy_inference_service_with_code(self, job_id: str, model_id: str, inference_code: str, 
                                         namespace: str = "default", replicas: int = 1,
                                         resource_requests: Dict[str, str] = None,
                                         resource_limits: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Deploys an inference service for a trained model, using dynamically provided inference code.
        This function creates a ConfigMap for the inference code before deploying.
        """
        try:
            # 1. Create a ConfigMap for the inference code
            config_map_name = f"{model_id}-inference-code"
            config_map_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: {config_map_name}
  namespace: {namespace}
data:
  inference.py: |
{inference_code}
"""
            logger.info(f"Creating ConfigMap {config_map_name} for model {model_id}.")
            config_map_result = self.k8s.apply_yaml({"yaml": config_map_yaml})
            if config_map_result["status"] != "success":
                return {"status": "error", "message": f"Failed to create ConfigMap for inference code: {config_map_result.get('message', 'Unknown error')}"}

            # 2. Ensure model artifacts are properly set up (e.g., PVC exists)
            storage_result = self.artifact_manager.create_model_pvc(job_id, namespace)
            if storage_result["status"] != "success":
                return storage_result

            # 3. Deploy the inference service, passing the config map name
            # The inference_manager will need to be updated to mount this ConfigMap
            image_tag = f"{job_id}-infer:latest"
            deploy_result = self.inference_manager.deploy_inference_service(
                job_id=job_id,
                image=image_tag,
                namespace=namespace,
                replicas=replicas,
                resource_requests=resource_requests,
                resource_limits=resource_limits,
                inference_config_map_name=config_map_name # Pass the ConfigMap name
            )

            if deploy_result["status"] == "success":
                self.model_registry.update_model_status(model_id, "deployed")
                return {
                    "status": "success",
                    "message": f"Successfully deployed model {model_id} with dynamic code.",
                    "deployment_details": deploy_result
                }
            else:
                return {"status": "error", "message": f"Failed to deploy model {model_id}: {deploy_result.get('message', 'Unknown deployment error')}"}
        except Exception as e:
            logger.error(f"Error in deploy_inference_service_with_code for model {model_id}: {e}")
            return {"status": "error", "message": f"An unexpected error occurred during deployment: {str(e)}"}

    def create_recommendation_model(self, task_description: str, data_format: str,
                                  features: List[str]) -> Dict[str, Any]:
        """
        Create a recommendation model from description.
        
        Args:
            task_description: Description of the recommendation task
            data_format: Description of input data format
            features: List of features to use
        """
        # Generate recommendation system code
        system = self.llm_trainer.generate_recommendation_system(
            task_type="recommendation",
            data_description=task_description,
            features=features
        )
        
        if "status" in system and system["status"] == "error":
            return system

        # Create job directory
        job_id = f"recsys-{int(time.time())}"
        job_dir = self.generate_job_dir(job_id)

        # Save generated files
        for filename, content in system.items():
            with open(os.path.join(job_dir, filename), 'w') as f:
                f.write(content)

        # Generate Dockerfile and K8s configs
        docker_result = self.generate_code_and_configs(
            f"Create a recommendation system with {task_description}",
            job_id
        )

        return {
            "status": "success",
            "job_id": job_id,
            "files": list(system.keys()),
            "docker_config": docker_result
        }

    def finetune_llm(self, task_description: str, data_format: str,
                     framework: str = "pytorch") -> Dict[str, Any]:
        """
        Generate and set up LLM fine-tuning pipeline.
        
        Args:
            task_description: Description of the fine-tuning task
            data_format: Description of training data format
            framework: ML framework to use
        """
        # Generate fine-tuning configuration
        config = self.llm_trainer.generate_finetuning_config(
            task_description,
            data_format
        )
        
        if "status" in config and config["status"] == "error":
            return config

        # Generate training code
        training_code = self.llm_trainer.generate_training_code(
            config,
            framework
        )
        
        if "status" in training_code and training_code["status"] == "error":
            return training_code

        # Create job directory
        job_id = f"finetune-{int(time.time())}"
        job_dir = self.generate_job_dir(job_id)

        # Save generated files
        for filename, content in training_code.items():
            with open(os.path.join(job_dir, filename), 'w') as f:
                f.write(content)

        # Generate Dockerfile and K8s configs
        docker_result = self.generate_code_and_configs(
            f"Fine-tune LLM for {task_description}",
            job_id
        )

        return {
            "status": "success",
            "job_id": job_id,
            "config": config,
            "files": list(training_code.keys()),
            "docker_config": docker_result
        }
