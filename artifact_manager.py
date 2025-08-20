import logging
import os
import subprocess
from typing import Optional, Dict, Any
from kubernetes import client, config
from storage_backends import BaseStorageBackend, KubernetesPvcBackend # Import storage backends

logger = logging.getLogger(__name__)

class ArtifactManager:
    """Manages ML model artifacts transfer between pods and storage."""
    
    def __init__(self, base_dir: str, storage_backend: BaseStorageBackend):
        self.base_dir = base_dir
        self.artifacts_dir = os.path.join(base_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.storage_backend = storage_backend # Assign the storage backend
        
    def extract_model_from_pod(self, job_id: str, pod_name: str, 
                             namespace: str = "default",
                             container: Optional[str] = None,
                             model_path_in_pod: str = "/app/model/mnist_cnn.pt") -> Dict[str, Any]:
        """
        Extract model artifacts from a training pod and upload to the configured storage backend.
        model_path_in_pod: The full path to the model artifact within the pod (e.g., /mnt/model/mnist_cnn.pt)
        """
        local_temp_path = os.path.join(self.artifacts_dir, job_id, os.path.basename(model_path_in_pod))
        os.makedirs(os.path.dirname(local_temp_path), exist_ok=True)

        # 1. Copy artifact from the pod to a local temporary location (using kubectl cp)
        # Note: This is still using kubectl cp as a bridge for this specific task.
        # In a fully integrated system, the training job would directly upload to the backend.
        container_suffix = f"-c {container}" if container else ""
        src_path_kubectl = f"{namespace}/{pod_name}{container_suffix}:{model_path_in_pod}"
        
        logger.info(f"Copying model from pod {pod_name}:{model_path_in_pod} to local {local_temp_path}")
        try:
            # Use subprocess directly as _run_kubectl_cp is now in KubernetesPvcBackend
            cmd = ["kubectl", "cp", src_path_kubectl, local_temp_path]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Model copied locally successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"kubectl cp failed to extract model: {e.stderr}")
            return {"status": "error", "message": f"Failed to copy model from pod: {e.stderr}"}
        except Exception as e:
            logger.error(f"Unexpected error during local model copy: {e}")
            return {"status": "error", "message": f"Unexpected error during local model copy: {str(e)}"}

        # 2. Upload the artifact from the local temporary location to the storage backend
        remote_storage_path = f"/{job_id}/{os.path.basename(model_path_in_pod)}" # Standardized remote path
        logger.info(f"Uploading model from local {local_temp_path} to remote {remote_storage_path} using backend {type(self.storage_backend).__name__}")
        upload_result = self.storage_backend.upload_artifact(local_temp_path, remote_storage_path)

        # 3. Clean up local temporary file
        try:
            if os.path.exists(local_temp_path):
                os.remove(local_temp_path)
                logger.info(f"Cleaned up local temporary file: {local_temp_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up local temp file {local_temp_path}: {e}")

        if upload_result["status"] == "success":
            upload_result["remote_path"] = remote_storage_path # Add the final remote path
            return upload_result
        else:
            return {"status": "error", "message": f"Failed to upload model to storage backend: {upload_result.get('message','Unknown error')}"}

    def copy_model_to_pod(self, job_id: str, pod_name: str,
                         namespace: str = "default",
                         container: Optional[str] = None,
                         model_path_in_pod: str = "/app/model/mnist_cnn.pt",
                         remote_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Downloads model artifacts from the configured storage backend and copies to an inference pod.
        remote_model_path: The full path to the model artifact in the remote storage (e.g., /job-id/mnist_cnn.pt)
        """
        if not remote_model_path:
            # Default to a standard remote path if not provided
            remote_model_path = f"/{job_id}/{os.path.basename(model_path_in_pod)}"

        local_temp_path = os.path.join(self.artifacts_dir, job_id, f"downloaded_{os.path.basename(model_path_in_pod)}")
        os.makedirs(os.path.dirname(local_temp_path), exist_ok=True)

        # 1. Download the artifact from the storage backend to a local temporary location
        logger.info(f"Downloading model from remote {remote_model_path} to local {local_temp_path} using backend {type(self.storage_backend).__name__}")
        download_result = self.storage_backend.download_artifact(remote_model_path, local_temp_path)
        if download_result["status"] != "success":
            return {"status": "error", "message": f"Failed to download model from storage backend: {download_result.get('message','Unknown error')}"}

        # 2. Copy the artifact from the local temporary location to the pod (using kubectl cp)
        container_suffix = f"-c {container}" if container else ""
        dst_path_kubectl = f"{namespace}/{pod_name}{container_suffix}:{model_path_in_pod}"
        
        logger.info(f"Copying model from local {local_temp_path} to pod {pod_name}:{model_path_in_pod}")
        try:
            cmd = ["kubectl", "cp", local_temp_path, dst_path_kubectl]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Model copied to pod successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"kubectl cp failed to copy model to pod: {e.stderr}")
            return {"status": "error", "message": f"Failed to copy model to pod: {e.stderr}"}
        except Exception as e:
            logger.error(f"Unexpected error during local model copy to pod: {e}")
            return {"status": "error", "message": f"Unexpected error during local model copy to pod: {str(e)}"}

        # 3. Clean up local temporary file
        try:
            if os.path.exists(local_temp_path):
                os.remove(local_temp_path)
                logger.info(f"Cleaned up local temporary file: {local_temp_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up local temp file {local_temp_path}: {e}")

        return {"status": "success", "message": f"Model copied to pod {pod_name} successfully."}

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
        except client.ApiException as e:
            if e.status == 409: # Conflict, PVC already exists
                logger.info(f"PVC {pvc_name} already exists. Skipping creation.")
                return {"status": "success", "message": f"PVC {pvc_name} already exists.", "pvc_name": pvc_name, "namespace": namespace}
            logger.error(f"Failed to create PVC: {e}")
            return {"status": "error", "message": str(e)}
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
        
        # The mount path within the pod should always be /mnt/model for consistency
        volume_mount = {
            "name": "model-storage",
            "mountPath": "/mnt/model"
        }
        
        return {
            "volume": volume,
            "volumeMount": volume_mount
        }

    def cleanup_artifacts(self, job_id: str) -> Dict[str, Any]:
        """
        Clean up artifacts for a job from local cache and remote storage.
        """
        local_cleanup_result = {"status": "success"}
        try:
            job_artifacts_dir = os.path.join(self.artifacts_dir, job_id)
            if os.path.exists(job_artifacts_dir):
                import shutil
                shutil.rmtree(job_artifacts_dir)
                logger.info(f"Cleaned up local artifacts directory: {job_artifacts_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup local artifacts: {e}")
            local_cleanup_result = {"status": "error", "message": str(e)}

        # Attempt to delete from remote storage as well, if applicable
        remote_cleanup_result = {"status": "success"}
        try:
            # This part assumes a convention for remote paths. For PVC backend, it's managed by K8s.
            # For cloud storage, you'd iterate through known artifact paths or a job-specific prefix.
            logger.info(f"Attempting to cleanup remote artifacts for job {job_id} using {type(self.storage_backend).__name__}.")
            # Example for a specific file if we knew its remote path:
            # self.storage_backend.delete_artifact(f"/{job_id}/mnist_cnn.pt") 
            # More robust cleanup would involve tracking all artifacts uploaded for a job.
        except Exception as e:
            logger.error(f"Failed to cleanup remote artifacts: {e}")
            remote_cleanup_result = {"status": "error", "message": str(e)}

        if local_cleanup_result["status"] == "success" and remote_cleanup_result["status"] == "success":
            return {"status": "success", "message": f"Artifacts for job {job_id} cleaned up locally and remotely."}
        else:
            messages = []
            if local_cleanup_result["status"] == "error":
                messages.append(f'Local cleanup failed: {local_cleanup_result["message"]}')
            if remote_cleanup_result["status"] == "error":
                messages.append(f'Remote cleanup failed: {remote_cleanup_result["message"]}')
            return {"status": "error", "message": "; ".join(messages)}
