import os
import logging
import uuid
import time
import subprocess
import json
from typing import Optional, Dict, List
from claude_analyzer import ClaudeAnalyzer

logger = logging.getLogger(__name__)

class MLOpsWorkflow:
    def __init__(self, claude_analyzer: ClaudeAnalyzer, base_dir: str):
        self.claude_analyzer = claude_analyzer
        self.base_dir = os.path.join(base_dir, "mlops_jobs")
        os.makedirs(self.base_dir, exist_ok=True)
        self.registry = os.environ.get("CONTAINER_REGISTRY", "localhost:5000")
        self.podman_cmd = "podman"  # Can be configured via env var if needed

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
        llm_prompt = [
            {
                "role": "user",
                "content": f"""
You are an expert MLOps engineer. Given the following user prompt, generate:
- Python training code (train.py) using PyTorch or TensorFlow/Keras for the task
- Python inference code (inference.py) exposing a REST API (Flask or FastAPI)
- Dockerfile for training (Dockerfile.train)
- Dockerfile for inference (Dockerfile.infer)
- Kubernetes YAML for training job (train_deploy.yaml)
- Kubernetes YAML for inference deployment/service (infer_deploy.yaml)
- (Optional) DVC pipeline YAML (dvc.yaml) if data versioning is needed

User prompt:
\"\"\"
{prompt}
\"\"\"

Return each file as a JSON object with keys: filename and content.
Format your response as a JSON list:
[
  {{"filename": "train.py", "content": "..."}},
  {{"filename": "inference.py", "content": "..."}},
  ...
]
"""
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
