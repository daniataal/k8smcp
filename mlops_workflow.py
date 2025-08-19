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

logger = logging.getLogger(__name__)

class MLOpsWorkflow:
    def __init__(self, claude_analyzer: ClaudeAnalyzer, base_dir: str):
        self.claude_analyzer = claude_analyzer
        self.base_dir = os.path.join(base_dir, "mlops_jobs")
        os.makedirs(self.base_dir, exist_ok=True)
        self.registry = os.environ.get("CONTAINER_REGISTRY", "localhost:5000")
        self.podman_cmd = "podman"  # Can be configured via env var if needed
        self.artifact_manager = ArtifactManager(base_dir)
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

    def generate_code_and_configs(self, prompt: str, job_dir: str) -> dict:
        """
        Use LLM to generate training code, inference code, Dockerfiles, and K8s YAMLs.
        Save them in job_dir.
        """
        if not self.claude_analyzer.is_available():
            return {"status": "error", "message": "Claude LLM is not available."}

        # Compose a system prompt for the LLM
        llm_prompt =  [
            {
                "role": "user",
                "content": f'''
You are an expert MLOps engineer. Given the following user prompt, generate:
- Python training code (train.py) using PyTorch or TensorFlow/Keras for the task
- Python inference code (inference.py) exposing a REST API (Flask or FastAPI)
- Dockerfile for training (Dockerfile.train) that specifies the base image (e.g., pytorch/pytorch:latest or tensorflow/tensorflow:latest)
- Dockerfile for inference (Dockerfile.infer) that specifies the base image (e.g., pytorch/pytorch:latest or tensorflow/tensorflow:latest)
- Kubernetes YAML for training job (train_deploy.yaml) including resource requests/limits, and optionally GPU resource requests (e.g., nvidia.com/gpu: 1)
- Kubernetes YAML for inference deployment/service (infer_deploy.yaml) including resource requests/limits, readiness/liveness probes (HTTP probe for port 8080), and optionally GPU resource requests
- requirements.txt listing all Python dependencies
- (Optional) DVC pipeline YAML (dvc.yaml) if data versioning is needed

User prompt:
"""
{prompt}
"""

Return each file as a JSON object with keys: filename and content.
Format your response as a JSON list:
[
  {{"filename": "train.py", "content": "..."}},
  {{"filename": "inference.py", "content": "..."}},
  {{"filename": "Dockerfile.train", "content": "..."}},
  {{"filename": "Dockerfile.infer", "content": "..."}},
  {{"filename": "requirements.txt", "content": "..."}},
  {{"filename": "train_deploy.yaml", "content": "..."}},
  {{"filename": "infer_deploy.yaml", "content": "..."}}
]
'''
            }
        ]

        response = self.claude_analyzer._send_request(llm_prompt, max_tokens=8000)
        if not response:
            return {"status": "error", "message": "Claude LLM did not respond."}

        # Try to extract the JSON list from the response
        import re, json
        try:
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                files = json.loads(json_match.group(1))
            else:
                files = json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"status": "error", "message": f"Failed to parse LLM response: {e}", "raw": response}

        # Write files to job_dir
        written_files = []
        for file_obj in files:
            filename = file_obj.get("filename")
            content = file_obj.get("content")
            if filename and content:
                file_path = os.path.join(job_dir, filename)
                with open(file_path, "w") as f:
                    f.write(content)
                written_files.append(filename)

        return {
            "status": "success",
            "job_dir": job_dir,
            "files": written_files,
            "job_id": os.path.basename(job_dir)
        }

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

    def _log_experiment_result(self, job_id: str, prompt: str, status: str, details: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None, hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """Logs the result of an MLOps experiment to a JSON file."""
        job_dir = os.path.join(self.base_dir, job_id)
        os.makedirs(job_dir, exist_ok=True) # Ensure job directory exists
        log_file_path = os.path.join(job_dir, "experiment_log.json")

        log_entry = {
            "timestamp": time.time(),
            "job_id": job_id,
            "prompt": prompt,
            "status": status,
            "details": details,
            "metrics": metrics,
            "hyperparameters": hyperparameters
        }

        # Read existing logs if any, then append and write back
        logs = []
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, "r") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted experiment log file: {log_file_path}. Starting fresh.")

        logs.append(log_entry)

        with open(log_file_path, "w") as f:
            json.dump(logs, f, indent=4)
        logger.info(f"Experiment result logged for job {job_id} with status {status}.")

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

    def deploy_registered_model(self, model_id: str, namespace: str = "default", replicas: int = 1) -> Dict[str, Any]:
        """Deploys an inference service for a model registered in the model registry."""
        model_details_result = self.model_registry.get_model_details(model_id)
        if model_details_result["status"] != "success":
            return {"status": "error", "message": f"Failed to get details for model {model_id}: {model_details_result.get("message", "Unknown error")}"}
        
        model_details = model_details_result["model_details"]
        job_id = model_details["job_id"]
        model_path = model_details["model_path"]

        # Assuming the inference image is tagged with {job_id}-infer:latest
        image_tag = f"{job_id}-infer:latest"

        # Simulate model validation before deployment
        validation_result = self._simulate_model_validation(model_id, model_details)
        if validation_result["status"] != "success":
            return {"status": "error", "message": f"Model validation failed for {model_id}: {validation_result.get("message", "Unknown validation error")}"}

        logger.info(f"Deploying registered model {model_id} (job_id: {job_id}) using image {image_tag}")

        deploy_result = self.inference_manager.deploy_inference_service(
            job_id=job_id, # Reusing job_id for deployment naming convention
            image=image_tag,
            namespace=namespace,
            replicas=replicas,
            # resource_requests and limits would ideally come from model_details or a deployment config
        )
        
        if deploy_result["status"] == "success":
            self.model_registry.update_model_status(model_id, "deployed")
            return {
                "status": "success",
                "message": f"Successfully deployed model {model_id}.",
                "deployment_details": deploy_result
            }
        else:
            return {"status": "error", "message": f"Failed to deploy model {model_id}: {deploy_result.get("message", "Unknown deployment error")}"}

    def _simulate_model_validation(self, model_id: str, model_details: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates a model validation process before deployment."""
        logger.info(f"Simulating validation for model {model_id}...")
        # In a real scenario, this would involve running tests, comparing metrics, etc.
        # For now, a simple check or always success.
        if model_details.get("metrics", {}).get("accuracy", 0) < 0.7: # Example validation rule
            return {"status": "error", "message": "Simulated validation failed: Model accuracy too low."}
        return {"status": "success", "message": "Simulated model validation passed."}

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
