# storage_backends.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import os
import subprocess
import logging
from kubernetes import client

logger = logging.getLogger(__name__)

class BaseStorageBackend(ABC):
    @abstractmethod
    def upload_artifact(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def download_artifact(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def delete_artifact(self, remote_path: str) -> Dict[str, Any]:
        pass

class KubernetesPvcBackend(BaseStorageBackend):
    def __init__(self, k8s_core_v1: client.CoreV1Api, namespace: str = "default"):
        self.k8s_core_v1 = k8s_core_v1
        self.namespace = namespace

    def _run_kubectl_cp(self, src: str, dst: str) -> Dict[str, Any]:
        """Run kubectl cp command."""
        try:
            cmd = ["kubectl", "cp", src, dst]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {"status": "success", "output": result.stdout}
        except subprocess.CalledProcessError as e:
            logger.error(f"kubectl cp failed: {e.stderr}")
            return {"status": "error", "message": e.stderr}

    def upload_artifact(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """
        Uploads an artifact from the local filesystem to a PVC via a temporary pod.
        This simulates direct PVC interaction for now. In a real scenario, the training/inference pods 
        themselves would write/read to the mounted PVC.
        remote_path here refers to the path within the mounted PVC in a dummy pod.
        """
        logger.warning("KubernetesPvcBackend: upload_artifact is a simulated operation using kubectl cp. In a real setup, applications directly write to mounted PVCs.")
        # For PVC, we simulate by copying from local to a dummy pod's mounted PVC
        # This requires a dummy pod with the PVC mounted
        # This is a simplification and in a real scenario, the training job would save directly to PVC

        # Create a temporary pod to act as an intermediary for kubectl cp
        pod_name = f"pvc-uploader-{os.path.basename(local_path).replace('.', '-')}-{os.urandom(4).hex()}"
        pvc_name = remote_path.split("/")[2] # Extract PVC name from remote_path format: /pvc/{pvc_name}/{file}
        mount_path = "/mnt/data"
        
        dummy_pod_yaml = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "volumes": [{
                    "name": "pvc-storage",
                    "persistentVolumeClaim": {"claimName": pvc_name}
                }],
                "containers": [{
                    "name": "dummy-container",
                    "image": "busybox",
                    "command": ["sh", "-c", "sleep 3600"], # Keep pod running
                    "volumeMounts": [{
                        "name": "pvc-storage",
                        "mountPath": mount_path
                    }]
                }],
                "restartPolicy": "Never"
            }
        }

        try:
            logger.info(f"Creating dummy pod {pod_name} to upload to PVC {pvc_name}")
            self.k8s_core_v1.create_namespaced_pod(body=dummy_pod_yaml, namespace=self.namespace)
            
            # Wait for pod to be running
            while True:
                pod_status = self.k8s_core_v1.read_namespaced_pod_status(name=pod_name, namespace=self.namespace)
                if pod_status.status.phase == "Running":
                    break
                logger.info(f"Waiting for dummy pod {pod_name} to be Running... current phase: {pod_status.status.phase}")
                import time
                time.sleep(2)

            # Copy artifact to the dummy pod's mounted PVC
            # remote_path is like /pvc/{pvc_name}/path/in/pvc/file.pt
            # We need to copy to /mnt/data/path/in/pvc/file.pt
            target_path_in_pod = os.path.join(mount_path, *remote_path.split("/")[3:])
            
            # Ensure the directory structure exists in the pod
            dir_in_pod = os.path.dirname(target_path_in_pod)
            self._run_kubectl_exec(pod_name, f"mkdir -p {dir_in_pod}")

            kubectl_dst = f"{self.namespace}/{pod_name}:{target_path_in_pod}"
            upload_result = self._run_kubectl_cp(local_path, kubectl_dst)
            return upload_result
        except Exception as e:
            logger.error(f"Failed to upload artifact to PVC via dummy pod: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            # Clean up the dummy pod
            try:
                logger.info(f"Deleting dummy pod {pod_name}")
                self.k8s_core_v1.delete_namespaced_pod(name=pod_name, namespace=self.namespace, body=client.V1DeleteOptions())
            except Exception as e:
                logger.warning(f"Failed to delete dummy pod {pod_name}: {e}")

    def download_artifact(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """
        Downloads an artifact from a PVC via a temporary pod to the local filesystem.
        remote_path here refers to the path within the mounted PVC in a dummy pod.
        """
        logger.warning("KubernetesPvcBackend: download_artifact is a simulated operation using kubectl cp. In a real setup, applications directly read from mounted PVCs.")

        pod_name = f"pvc-downloader-{os.path.basename(local_path).replace('.', '-')}-{os.urandom(4).hex()}"
        pvc_name = remote_path.split("/")[2] # Extract PVC name
        mount_path = "/mnt/data"

        dummy_pod_yaml = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "volumes": [{
                    "name": "pvc-storage",
                    "persistentVolumeClaim": {"claimName": pvc_name}
                }],
                "containers": [{
                    "name": "dummy-container",
                    "image": "busybox",
                    "command": ["sh", "-c", "sleep 3600"], 
                    "volumeMounts": [{
                        "name": "pvc-storage",
                        "mountPath": mount_path
                    }]
                }],
                "restartPolicy": "Never"
            }
        }

        try:
            logger.info(f"Creating dummy pod {pod_name} to download from PVC {pvc_name}")
            self.k8s_core_v1.create_namespaced_pod(body=dummy_pod_yaml, namespace=self.namespace)
            
            while True:
                pod_status = self.k8s_core_v1.read_namespaced_pod_status(name=pod_name, namespace=self.namespace)
                if pod_status.status.phase == "Running":
                    break
                logger.info(f"Waiting for dummy pod {pod_name} to be Running... current phase: {pod_status.status.phase}")
                import time
                time.sleep(2)

            source_path_in_pod = os.path.join(mount_path, *remote_path.split("/")[3:])
            kubectl_src = f"{self.namespace}/{pod_name}:{source_path_in_pod}"
            download_result = self._run_kubectl_cp(kubectl_src, local_path)
            return download_result
        except Exception as e:
            logger.error(f"Failed to download artifact from PVC via dummy pod: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            try:
                logger.info(f"Deleting dummy pod {pod_name}")
                self.k8s_core_v1.delete_namespaced_pod(name=pod_name, namespace=self.namespace, body=client.V1DeleteOptions())
            except Exception as e:
                logger.warning(f"Failed to delete dummy pod {pod_name}: {e}")

    def delete_artifact(self, remote_path: str) -> Dict[str, Any]:
        """
        Deletes an artifact from a PVC via a temporary pod.
        remote_path here refers to the path within the mounted PVC in a dummy pod.
        """
        logger.warning("KubernetesPvcBackend: delete_artifact is a simulated operation. In a real setup, managing individual files on PVCs might be less common.")
        # This is a highly simplified deletion. In a real scenario, you might delete the PVC itself or rely on app-level cleanup.
        # For simulation, we can run an exec command to delete the file inside a dummy pod.
        
        pod_name = f"pvc-deleter-{os.urandom(4).hex()}"
        pvc_name = remote_path.split("/")[2] # Extract PVC name
        mount_path = "/mnt/data"

        dummy_pod_yaml = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "volumes": [{
                    "name": "pvc-storage",
                    "persistentVolumeClaim": {"claimName": pvc_name}
                }],
                "containers": [{
                    "name": "dummy-container",
                    "image": "busybox",
                    "command": ["sh", "-c", "sleep 3600"], 
                    "volumeMounts": [{
                        "name": "pvc-storage",
                        "mountPath": mount_path
                    }]
                }],
                "restartPolicy": "Never"
            }
        }

        try:
            logger.info(f"Creating dummy pod {pod_name} to delete from PVC {pvc_name}")
            self.k8s_core_v1.create_namespaced_pod(body=dummy_pod_yaml, namespace=self.namespace)
            
            while True:
                pod_status = self.k8s_core_v1.read_namespaced_pod_status(name=pod_name, namespace=self.namespace)
                if pod_status.status.phase == "Running":
                    break
                logger.info(f"Waiting for dummy pod {pod_name} to be Running... current phase: {pod_status.status.phase}")
                import time
                time.sleep(2)

            target_path_in_pod = os.path.join(mount_path, *remote_path.split("/")[3:])
            exec_command = ['rm', '-rf', target_path_in_pod]
            
            # Execute command in the pod
            api_client = client.CoreV1Api()
            resp = client.stream(api_client.connect_get_namespaced_pod_exec,
                                 name=pod_name,
                                 namespace=self.namespace,
                                 command=exec_command,
                                 stderr=True, stdin=False,
                                 stdout=True, tty=False)
            logger.info(f"kubectl exec output: {resp}")
            return {"status": "success", "message": f"Attempted to delete {remote_path}. Output: {resp}"}
        except Exception as e:
            logger.error(f"Failed to delete artifact from PVC via dummy pod: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            try:
                logger.info(f"Deleting dummy pod {pod_name}")
                self.k8s_core_v1.delete_namespaced_pod(name=pod_name, namespace=self.namespace, body=client.V1DeleteOptions())
            except Exception as e:
                logger.warning(f"Failed to delete dummy pod {pod_name}: {e}")

    def _run_kubectl_exec(self, pod_name: str, command: str) -> Dict[str, Any]:
        """
        Executes a command inside a specified pod.
        """
        try:
            exec_command = ["kubectl", "exec", "-n", self.namespace, pod_name, "--", "sh", "-c", command]
            result = subprocess.run(exec_command, capture_output=True, text=True, check=True)
            return {"status": "success", "output": result.stdout}
        except subprocess.CalledProcessError as e:
            logger.error(f"kubectl exec failed: {e.stderr}")
            return {"status": "error", "message": e.stderr}
