import logging
import yaml
from typing import Dict, Any, Optional
from kubernetes import client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

class InferenceManager:
    """Manages ML model inference services in Kubernetes."""
    
    def __init__(self, k8s_apps_v1: client.AppsV1Api, k8s_core_v1: client.CoreV1Api):
        self.apps_v1 = k8s_apps_v1
        self.core_v1 = k8s_core_v1

    def generate_inference_yaml(self, job_id: str, image: str, model_path: str = "/app/model",
                              replicas: int = 1, port: int = 8080,
                              resource_requests: Dict[str, str] = None,
                              resource_limits: Dict[str, str] = None) -> Dict[str, Any]:
        """Generate Kubernetes YAML for inference deployment and service."""
        
        if not resource_requests:
            resource_requests = {"cpu": "500m", "memory": "1Gi"}
        if not resource_limits:
            resource_limits = {"cpu": "2", "memory": "4Gi"}

        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"inference-{job_id}",
                "labels": {
                    "app": f"inference-{job_id}",
                    "job-id": job_id,
                    "type": "inference"
                }
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": f"inference-{job_id}"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": f"inference-{job_id}",
                            "job-id": job_id,
                            "type": "inference"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "inference",
                            "image": image,
                            "ports": [{
                                "containerPort": port
                            }],
                            "resources": {
                                "requests": resource_requests,
                                "limits": resource_limits
                            },
                            "volumeMounts": [{
                                "name": "model-storage",
                                "mountPath": model_path
                            }]
                        }],
                        "volumes": [{
                            "name": "model-storage",
                            "persistentVolumeClaim": {
                                "claimName": f"model-{job_id}"
                            }
                        }]
                    }
                }
            }
        }

        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"inference-{job_id}",
                "labels": {
                    "app": f"inference-{job_id}",
                    "job-id": job_id,
                    "type": "inference"
                }
            },
            "spec": {
                "selector": {
                    "app": f"inference-{job_id}"
                },
                "ports": [{
                    "port": port,
                    "targetPort": port,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }

        return {
            "deployment": deployment,
            "service": service
        }

    def deploy_inference_service(self, job_id: str, image: str,
                               namespace: str = "default", **kwargs) -> Dict[str, Any]:
        """
        Deploy an inference service for a trained model.
        
        Args:
            job_id: Unique identifier for the job/model
            image: Docker image for inference
            namespace: Kubernetes namespace
            **kwargs: Additional args for generate_inference_yaml
        """
        try:
            # Generate YAML
            manifests = self.generate_inference_yaml(job_id, image, **kwargs)
            
            # Create deployment
            deployment = self.apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=manifests["deployment"]
            )
            
            # Create service
            service = self.core_v1.create_namespaced_service(
                namespace=namespace,
                body=manifests["service"]
            )
            
            return {
                "status": "success",
                "deployment": {
                    "name": deployment.metadata.name,
                    "replicas": deployment.spec.replicas
                },
                "service": {
                    "name": service.metadata.name,
                    "cluster_ip": service.spec.cluster_ip,
                    "ports": [{"port": p.port, "target_port": p.target_port} 
                             for p in service.spec.ports]
                }
            }
        except ApiException as e:
            logger.error(f"Failed to deploy inference service: {e}")
            return {"status": "error", "message": str(e)}

    def update_inference_service(self, job_id: str, namespace: str = "default",
                               image: Optional[str] = None,
                               replicas: Optional[int] = None) -> Dict[str, Any]:
        """Update an existing inference service."""
        try:
            deployment_name = f"inference-{job_id}"
            
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update image if provided
            if image:
                deployment.spec.template.spec.containers[0].image = image
            
            # Update replicas if provided
            if replicas is not None:
                deployment.spec.replicas = replicas
            
            # Apply updates
            updated = self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            return {
                "status": "success",
                "deployment": {
                    "name": updated.metadata.name,
                    "replicas": updated.spec.replicas,
                    "image": updated.spec.template.spec.containers[0].image
                }
            }
        except ApiException as e:
            logger.error(f"Failed to update inference service: {e}")
            return {"status": "error", "message": str(e)}

    def delete_inference_service(self, job_id: str, 
                               namespace: str = "default") -> Dict[str, Any]:
        """Delete an inference service and its resources."""
        try:
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=f"inference-{job_id}",
                namespace=namespace
            )
            
            # Delete service
            self.core_v1.delete_namespaced_service(
                name=f"inference-{job_id}",
                namespace=namespace
            )
            
            return {
                "status": "success",
                "message": f"Deleted inference service for job {job_id}"
            }
        except ApiException as e:
            logger.error(f"Failed to delete inference service: {e}")
            return {"status": "error", "message": str(e)}

    def get_inference_status(self, job_id: str, 
                           namespace: str = "default") -> Dict[str, Any]:
        """Get status of an inference service."""
        try:
            # Get deployment status
            deployment = self.apps_v1.read_namespaced_deployment(
                name=f"inference-{job_id}",
                namespace=namespace
            )
            
            # Get service status
            service = self.core_v1.read_namespaced_service(
                name=f"inference-{job_id}",
                namespace=namespace
            )
            
            return {
                "status": "success",
                "deployment": {
                    "name": deployment.metadata.name,
                    "available_replicas": deployment.status.available_replicas,
                    "ready_replicas": deployment.status.ready_replicas,
                    "image": deployment.spec.template.spec.containers[0].image
                },
                "service": {
                    "name": service.metadata.name,
                    "cluster_ip": service.spec.cluster_ip,
                    "ports": [{"port": p.port, "target_port": p.target_port} 
                             for p in service.spec.ports]
                }
            }
        except ApiException as e:
            logger.error(f"Failed to get inference service status: {e}")
            return {"status": "error", "message": str(e)}
