import logging
import os
import subprocess
from typing import Optional, Dict, Any
from kubernetes import client, config

logger = logging.getLogger(__name__)

class ArtifactManager:
    """Manages ML model artifacts transfer between pods and storage."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.artifacts_dir = os.path.join(base_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def _run_kubectl_cp(self, src: str, dst: str) -> Dict[str, Any]:
        """Run kubectl cp command."""
        try:
            cmd = ["kubectl", "cp", src, dst]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {"status": "success", "output": result.stdout}
        except subprocess.CalledProcessError as e:
            logger.error(f"kubectl cp failed: {e.stderr}")
            return {"status": "error", "message": e.stderr}

    def extract_model_from_pod(self, job_id: str, pod_name: str, 
                             namespace: str = "default",
                             container: Optional[str] = None,
                             model_path: str = "/app/model") -> Dict[str, Any]:
        """
        Extract model artifacts from a training pod.
        """
        # Create job-specific artifacts directory
        job_artifacts_dir = os.path.join(self.artifacts_dir, job_id)
        os.makedirs(job_artifacts_dir, exist_ok=True)
        
        # Construct source and destination paths
        container_suffix = f"-c {container}" if container else ""
        src_path = f"{namespace}/{pod_name}{container_suffix}:{model_path}"
        
        # Copy artifacts from pod
        result = self._run_kubectl_cp(src_path, job_artifacts_dir)
        if result["status"] == "success":
            result["artifacts_dir"] = job_artifacts_dir
        return result

    def copy_model_to_pod(self, job_id: str, pod_name: str,
                         namespace: str = "default",
                         container: Optional[str] = None,
                         model_path: str = "/app/model") -> Dict[str, Any]:
        """
        Copy model artifacts to an inference pod.
        """
        # Get job artifacts directory
        job_artifacts_dir = os.path.join(self.artifacts_dir, job_id)
        if not os.path.exists(job_artifacts_dir):
            return {
                "status": "error",
                "message": f"No artifacts found for job {job_id}"
            }
        
        # Construct source and destination paths
        container_suffix = f"-c {container}" if container else ""
        dst_path = f"{namespace}/{pod_name}{container_suffix}:{model_path}"
        
        # Copy artifacts to pod
        return self._run_kubectl_cp(job_artifacts_dir, dst_path)

    def create_model_pvc(self, job_id: str, namespace: str = "default",
                        storage_class: str = "standard",
                        size: str = "1Gi") -> Dict[str, Any]:
        """
        Create a PVC for storing model artifacts.
        """
        try:
            pvc_name = f"model-{job_id}"
            pvc_manifest = {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": pvc_name},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "storageClassName": storage_class,
                    "resources": {"requests": {"storage": size}}
                }
            }
            
            api = client.CoreV1Api()
            api.create_namespaced_persistent_volume_claim(
                namespace=namespace,
                body=pvc_manifest
            )
            
            return {
                "status": "success",
                "pvc_name": pvc_name,
                "namespace": namespace
            }
        except Exception as e:
            logger.error(f"Failed to create PVC: {e}")
            return {"status": "error", "message": str(e)}

    def get_volume_mounts(self, job_id: str) -> Dict[str, Any]:
        """
        Get volume and volumeMount configurations for pods.
        """
        pvc_name = f"model-{job_id}"
        
        volume = {
            "name": "model-storage",
            "persistentVolumeClaim": {
                "claimName": pvc_name
            }
        }
        
        volume_mount = {
            "name": "model-storage",
            "mountPath": "/app/model"
        }
        
        return {
            "volume": volume,
            "volumeMount": volume_mount
        }

    def cleanup_artifacts(self, job_id: str) -> Dict[str, Any]:
        """
        Clean up artifacts for a job.
        """
        try:
            job_artifacts_dir = os.path.join(self.artifacts_dir, job_id)
            if os.path.exists(job_artifacts_dir):
                import shutil
                shutil.rmtree(job_artifacts_dir)
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Failed to cleanup artifacts: {e}")
            return {"status": "error", "message": str(e)}
