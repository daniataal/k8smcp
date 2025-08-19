import logging
import json
import yaml
from typing import Dict, Any, List, Optional, Tuple
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream, portforward
from command_router import CommandRouter  # Removed relative import
from dvc_manager import DVCManager
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class KubernetesDebugger:
    def __init__(self, claude_analyzer, k8s_available=True):
        self.k8s_available = k8s_available
        if (k8s_available):
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            self.batch_v1 = client.BatchV1Api()
            self.custom_api = client.CustomObjectsApi()
        else:
            self.v1 = None
            self.apps_v1 = None
            self.networking_v1 = None
            self.batch_v1 = None
            self.custom_api = None
        self.claude_analyzer = claude_analyzer
        self.router = CommandRouter(self)
        self.dvc_manager = DVCManager()
        self._register_commands()

    def _register_commands(self):
        """Register all command handlers"""
        # Pod operations
        self.router.register("get pods", self.get_pods)
        self.router.register("describe pod", self.describe_pod)
        self.router.register("logs", self.get_pod_logs)
        self.router.register("exec", self.execute_command_in_pod)
        
        # Node operations
        self.router.register("get nodes", self.get_nodes)
        self.router.register("describe node", self.describe_node)
        self.router.register("node pressure", self.check_node_pressure)
        
        # Deployment operations
        self.router.register("get deployments", self.get_deployments)
        self.router.register("describe deployment", self.describe_deployment)
        self.router.register("scale", self.scale_deployment)
        self.router.register("rollout", self.manage_rollout)
        self.router.register("delete deployment", self.delete_deployment)
        self.router.register("delete all deployments", self.delete_all_deployments)
        
        # Service operations
        self.router.register("get services", self.get_services)
        self.router.register("describe service", self.describe_service)
        
        # ConfigMap and Secret operations
        self.router.register("get configmaps", self.get_configmaps)
        self.router.register("get secrets", self.get_secrets)
        
        # Context operations
        self.router.register("get contexts", self.get_contexts)
        self.router.register("use context", self.use_context)
        
        # Create/Apply/Delete operations
        self.router.register("apply", self.apply_yaml)
        self.router.register("delete", self.delete_resource)
        self.router.register("create", self.create_resource)
        
        # Port forwarding
        self.router.register("port-forward", self.port_forward)
        
        # Monitoring integrations
        self.router.register("prometheus", self.query_prometheus)
        self.router.register("grafana", self.query_grafana)
        self.router.register("elasticsearch", self.query_elasticsearch)
        
        # Debug operations
        self.router.register("diagnose", self.diagnose_cluster)
        self.router.register("analyze", self.analyze_resource)
        self.router.register("recommend", self.recommend_action)

        # DVC operations
        self.router.register("dvc init", lambda params: self.dvc_manager.init())
        self.router.register("dvc add", lambda params: self.dvc_manager.add(params.get("path", "")))
        self.router.register("dvc push", lambda params: self.dvc_manager.push())
        self.router.register("dvc pull", lambda params: self.dvc_manager.pull())
        self.router.register("dvc status", lambda params: self.dvc_manager.status())
        self.router.register("dvc repro", lambda params: self.dvc_manager.repro())

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for handling commands"""
        try:
            return self.router.route(command, params)
        except Exception as e:
            logger.exception(f"Error handling command {command}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _require_k8s(self):
        if not self.k8s_available:
            return {
                "status": "error",
                "message": "Kubernetes API is not available. Check your kubeconfig or cluster environment."
            }
        return None

    # --- Stub implementations for all registered handlers ---
    def get_pods(self, params):
        err = self._require_k8s()
        if err:
            return err
        namespace = params.get("namespace", "default")
        label_selector = params.get("label_selector", "")
        field_selector = params.get("field_selector", "")
        try:
            pods = self.v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
                field_selector=field_selector
            )
            pod_list = []
            for pod in pods.items:
                pod_list.append({
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "containers": [c.name for c in pod.spec.containers]
                })
            return {"status": "success", "pods": pod_list}
        except Exception as e:
            logger.error(f"Error fetching pods: {e}")
            return {"status": "error", "message": str(e)}

    def describe_pod(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        name = params.get("name")
        namespace = params.get("namespace", "default")
        
        if not name:
            return {"status": "error", "message": "Pod name is required"}
            
        try:
            pod = self.v1.read_namespaced_pod(name=name, namespace=namespace)
            events = self.v1.list_namespaced_event(
                namespace=namespace,
                field_selector=f'involvedObject.name={name}'
            )
            
            description = {
                "metadata": {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "uid": pod.metadata.uid,
                    "labels": pod.metadata.labels,
                    "annotations": pod.metadata.annotations,
                    "creation_timestamp": str(pod.metadata.creation_timestamp)
                },
                "spec": {
                    "node_name": pod.spec.node_name,
                    "service_account": pod.spec.service_account,
                    "containers": [{
                        "name": c.name,
                        "image": c.image,
                        "ports": [{"container_port": p.container_port, "protocol": p.protocol} for p in (c.ports or [])],
                        "resources": c.resources.to_dict() if c.resources else {},
                        "volume_mounts": [{"name": v.name, "mount_path": v.mount_path} for v in (c.volume_mounts or [])]
                    } for c in pod.spec.containers],
                    "volumes": [{
                        "name": v.name,
                        "type": next(iter([k for k in v.to_dict().keys() if k != 'name']), None)
                    } for v in (pod.spec.volumes or [])]
                },
                "status": {
                    "phase": pod.status.phase,
                    "host_ip": pod.status.host_ip,
                    "pod_ip": pod.status.pod_ip,
                    "start_time": str(pod.status.start_time) if pod.status.start_time else None,
                    "conditions": [{
                        "type": c.type,
                        "status": c.status,
                        "last_transition_time": str(c.last_transition_time)
                    } for c in (pod.status.conditions or [])],
                    "container_statuses": [{
                        "name": cs.name,
                        "ready": cs.ready,
                        "restart_count": cs.restart_count,
                        "state": next(iter(cs.state.to_dict().items()))[0] if cs.state else None
                    } for cs in (pod.status.container_statuses or [])]
                },
                "events": [{
                    "type": e.type,
                    "reason": e.reason,
                    "message": e.message,
                    "count": e.count,
                    "first_timestamp": str(e.first_timestamp) if e.first_timestamp else None,
                    "last_timestamp": str(e.last_timestamp) if e.last_timestamp else None
                } for e in events.items]
            }
            
            return {"status": "success", "description": description}
            
        except ApiException as e:
            logger.error(f"Error describing pod {name}: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error describing pod {name}: {e}")
            return {"status": "error", "message": str(e)}

    def get_pod_logs(self, params):
        err = self._require_k8s()
        if err:
            return err

        name = params.get("name")
        namespace = params.get("namespace", "default")
        container = params.get("container")
        tail_lines = params.get("tail_lines")
        since_seconds = params.get("since_seconds")
        previous = params.get("previous", False)

        if not name:
            return {"status": "error", "message": "Pod name is required"}

        try:
            logs = self.v1.read_namespaced_pod_log(
                name=name,
                namespace=namespace,
                container=container,
                tail_lines=tail_lines,
                since_seconds=since_seconds,
                previous=previous
            )
            return {
                "status": "success",
                "logs": logs,
                "pod": name,
                "container": container,
                "namespace": namespace
            }
        except ApiException as e:
            logger.error(f"Error getting logs for pod {name}: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error getting logs for pod {name}: {e}")
            return {"status": "error", "message": str(e)}

    def execute_command_in_pod(self, params):
        """
        Execute a command in a Kubernetes pod.
        
        Args:
            params: Dictionary containing:
                - name: Pod name
                - namespace: Pod namespace
                - container: Container name (optional)
                - command: Command to execute (list of strings)
        
        Returns:
            Dictionary containing the command output or error message.
        """
        err = self._require_k8s()
        if err:
            return err
        
        pod_name = params.get("name")
        namespace = params.get("namespace", "default")
        container = params.get("container")
        command = params.get("command", [])
        
        if not pod_name or not command:
            return {
                "status": "error",
                "message": "Pod name and command are required."
            }
        
        try:
            logger.info(f"Executing command {command} in pod {pod_name} (namespace: {namespace}, container: {container})")
            response = stream(
                self.v1.connect_get_namespaced_pod_exec,
                name=pod_name,
                namespace=namespace,
                container=container,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False
            )
            return {"status": "success", "output": response}
        except ApiException as e:
            logger.error(f"Failed to execute command in pod {pod_name}: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error while executing command in pod {pod_name}: {e}")
            return {"status": "error", "message": str(e)}

    def ensure_delete_deployment_rbac(self, namespace="default"):
        """
        Ensure the default service account has permission to delete deployments in the given namespace.
        """
        rbac_yaml = f"""
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: deployment-deleter
  namespace: {namespace}
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["delete", "get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: deployment-deleter-binding
  namespace: {namespace}
subjects:
- kind: ServiceAccount
  name: default
  namespace: {namespace}
roleRef:
  kind: Role
  name: deployment-deleter
  apiGroup: rbac.authorization.k8s.io
"""
        return self.apply_yaml({"yaml": rbac_yaml})

    def get_nodes(self, params):
        err = self._require_k8s()
        if err:
            return err
        try:
            nodes = self.v1.list_node(label_selector=params.get("label_selector", ""), field_selector=params.get("field_selector", ""))
            node_list = []
            for node in nodes.items:
                node_list.append({
                    "name": node.metadata.name,
                    "labels": node.metadata.labels,
                    "status": node.status.conditions[-1].type if node.status.conditions else "Unknown"
                })
            return {"status": "success", "nodes": node_list}
        except Exception as e:
            logger.error(f"Error fetching nodes: {e}")
            return {"status": "error", "message": str(e)}

    def describe_node(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        name = params.get("name")
        if not name:
            return {"status": "error", "message": "Node name is required"}
        
        try:
            node = self.v1.read_node(name=name)
            description = {
                "metadata": {
                    "name": node.metadata.name,
                    "labels": node.metadata.labels,
                    "annotations": node.metadata.annotations,
                },
                "status": {
                    "conditions": [{
                        "type": c.type,
                        "status": c.status,
                        "message": c.message
                    } for c in node.status.conditions],
                    "capacity": node.status.capacity,
                    "allocatable": node.status.allocatable,
                    "node_info": {
                        "architecture": node.status.node_info.architecture,
                        "container_runtime_version": node.status.node_info.container_runtime_version,
                        "kernel_version": node.status.node_info.kernel_version,
                        "os_image": node.status.node_info.os_image,
                        "kubelet_version": node.status.node_info.kubelet_version
                    }
                }
            }
            return {"status": "success", "description": description}
        except ApiException as e:
            logger.error(f"Error describing node {name}: {e}")
            return {"status": "error", "message": str(e)}

    def check_node_pressure(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        name = params.get("name")
        if not name:
            return {"status": "error", "message": "Node name is required"}
        
        try:
            node = self.v1.read_node(name=name)
            pressures = {}
            for condition in node.status.conditions:
                if condition.type in ["MemoryPressure", "DiskPressure", "PIDPressure"]:
                    pressures[condition.type] = {
                        "status": condition.status,
                        "message": condition.message,
                        "last_transition": str(condition.last_transition_time)
                    }
            return {"status": "success", "pressures": pressures}
        except ApiException as e:
            logger.error(f"Error checking node pressure for {name}: {e}")
            return {"status": "error", "message": str(e)}

    def get_deployments(self, params):
        err = self._require_k8s()
        if err:
            return err
        namespace = params.get("namespace", "default")
        all_namespaces = params.get("all_namespaces", False)
        label_selector = params.get("label_selector", "")
        try:
            if all_namespaces:
                deployments = self.apps_v1.list_deployment_for_all_namespaces(label_selector=label_selector)
            else:
                deployments = self.apps_v1.list_namespaced_deployment(namespace=namespace, label_selector=label_selector)
            deployment_list = []
            for dep in deployments.items:
                deployment_list.append({
                    "name": dep.metadata.name,
                    "namespace": dep.metadata.namespace,
                    "replicas": dep.spec.replicas,
                    "available_replicas": dep.status.available_replicas,
                    "ready_replicas": dep.status.ready_replicas,
                    "updated_replicas": dep.status.updated_replicas,
                    "labels": dep.metadata.labels,
                    "status": dep.status.conditions[-1].type if dep.status.conditions else "Unknown"
                })
            return {"status": "success", "deployments": deployment_list}
        except Exception as e:
            logger.error(f"Error fetching deployments: {e}")
            return {"status": "error", "message": str(e)}

    def describe_deployment(self, params):
        err = self._require_k8s()
        if err:
            return err
        name = params.get("name")
        namespace = params.get("namespace", "default")
        if not name:
            return {"status": "error", "message": "Deployment name is required."}
        try:
            dep = self.apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
            description = {
                "name": dep.metadata.name,
                "namespace": dep.metadata.namespace,
                "replicas": dep.spec.replicas,
                "available_replicas": dep.status.available_replicas,
                "ready_replicas": dep.status.ready_replicas,
                "updated_replicas": dep.status.updated_replicas,
                "labels": dep.metadata.labels,
                "selector": dep.spec.selector.match_labels,
                "strategy": dep.spec.strategy.type,
                "conditions": [c.type + ": " + c.status for c in dep.status.conditions] if dep.status.conditions else [],
                "containers": [
                    {
                        "name": c.name,
                        "image": c.image,
                        "resources": c.resources.to_dict() if c.resources else {}
                    } for c in dep.spec.template.spec.containers
                ]
            }
            return {"status": "success", "description": description}
        except Exception as e:
            logger.error(f"Error describing deployment: {e}")
            return {"status": "error", "message": str(e)}

    def scale_deployment(self, params):
        err = self._require_k8s()
        if err:
            return err
        name = params.get("name")
        namespace = params.get("namespace", "default")
        replicas = params.get("replicas")
        if not name or replicas is None:
            return {"status": "error", "message": "Deployment name and replicas are required."}
        try:
            body = {"spec": {"replicas": replicas}}
            self.apps_v1.patch_namespaced_deployment_scale(name, namespace, body)
            return {"status": "success", "result": f"Scaled deployment '{name}' to {replicas} replicas."}
        except Exception as e:
            logger.error(f"Error scaling deployment: {e}")
            return {"status": "error", "message": str(e)}

    def manage_rollout(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        name = params.get("name")
        namespace = params.get("namespace", "default")
        action = params.get("action", "status")  # status, pause, resume, restart, undo
        
        if not name:
            return {"status": "error", "message": "Deployment name is required"}
            
        try:
            if action == "restart":
                body = {
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {
                                    "kubectl.kubernetes.io/restartedAt": datetime.now().isoformat()
                                }
                            }
                        }
                    }
                }
                self.apps_v1.patch_namespaced_deployment(name, namespace, body)
            elif action in ["pause", "resume"]:
                body = {"spec": {"paused": action == "pause"}}
                self.apps_v1.patch_namespaced_deployment(name, namespace, body)
            
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            return {
                "status": "success",
                "rollout": {
                    "name": name,
                    "namespace": namespace,
                    "action": action,
                    "paused": deployment.spec.paused,
                    "replicas": deployment.spec.replicas,
                    "updated_replicas": deployment.status.updated_replicas,
                    "ready_replicas": deployment.status.ready_replicas,
                    "available_replicas": deployment.status.available_replicas
                }
            }
        except ApiException as e:
            logger.error(f"Error managing rollout for {name}: {e}")
            return {"status": "error", "message": str(e)}

    def delete_deployment(self, params):
        """
        Delete a Kubernetes deployment by name and namespace.
        Args:
            params: dict with 'name' and optional 'namespace'
        Returns:
            dict with status and message
        """
        err = self._require_k8s()
        if err:
            return err
        name = params.get("name")
        namespace = params.get("namespace", "default")
        if not name:
            return {"status": "error", "message": "Deployment name is required."}
        try:
            self.apps_v1.delete_namespaced_deployment(name=name, namespace=namespace)
            return {
                "status": "success",
                "message": f"Deleted deployment '{name}' in namespace '{namespace}'."
            }
        except Exception as e:
            logger.error(f"Error deleting deployment {name}: {e}")
            return {"status": "error", "message": str(e)}

    def delete_all_deployments(self, params):
        """Delete all deployments in a namespace using direct API calls."""
        err = self._require_k8s()
        if err:
            return err

        namespace = params.get("namespace", "default")
        force = params.get("force", False)
        timeout = params.get("timeout_seconds", 60)

        try:
            # List deployments
            deployments = self.apps_v1.list_namespaced_deployment(namespace=namespace)

            if not deployments.items:
                return {
                    "status": "success",
                    "message": f"No deployments found in namespace {namespace}"
                }

            results = []
            delete_options = client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=0 if force else timeout
            )

            # Delete each deployment using direct API call
            for deployment in deployments.items:
                name = deployment.metadata.name
                try:
                    self.apps_v1.delete_namespaced_deployment(
                        name=name,
                        namespace=namespace,
                        body=delete_options
                    )
                    results.append({
                        "name": name,
                        "status": "deleted"
                    })
                except ApiException as e:
                    results.append({
                        "name": name,
                        "status": "error",
                        "error": str(e)
                    })

            successes = sum(1 for r in results if r["status"] == "deleted")
            failures = sum(1 for r in results if r["status"] == "error")

            return {
                "status": "success" if failures == 0 else "partial_success",
                "summary": f"Deleted {successes} deployments, {failures} failures",
                "namespace": namespace,
                "results": results
            }

        except ApiException as e:
            logger.error(f"Failed to delete deployments in namespace {namespace}: {e}")
            return {
                "status": "error",
                "message": f"API error: {str(e)}",
                "code": e.status
            }
        except Exception as e:
            logger.error(f"Unexpected error deleting deployments: {e}")
            return {
                "status": "error", 
                "message": f"Unexpected error: {str(e)}"
            }

    def get_services(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        namespace = params.get("namespace", "default")
        label_selector = params.get("label_selector", "")
        
        try:
            services = self.v1.list_namespaced_service(namespace=namespace, label_selector=label_selector)
            service_list = []
            for svc in services.items:
                service_list.append({
                    "name": svc.metadata.name,
                    "namespace": svc.metadata.namespace,
                    "type": svc.spec.type,
                    "cluster_ip": svc.spec.cluster_ip,
                    "external_ip": svc.spec.external_i_ps if hasattr(svc.spec, 'external_i_ps') else None,
                    "ports": [{
                        "port": p.port,
                        "target_port": p.target_port,
                        "protocol": p.protocol,
                        "node_port": p.node_port if hasattr(p, 'node_port') else None
                    } for p in svc.spec.ports],
                    "selector": svc.spec.selector
                })
            return {"status": "success", "services": service_list}
        except ApiException as e:
            logger.error(f"Error getting services: {e}")
            return {"status": "error", "message": str(e)}

    def describe_service(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        name = params.get("name")
        namespace = params.get("namespace", "default")
        
        if not name:
            return {"status": "error", "message": "Service name is required"}
            
        try:
            svc = self.v1.read_namespaced_service(name=name, namespace=namespace)
            endpoints = self.v1.read_namespaced_endpoints(name=name, namespace=namespace)
            
            description = {
                "metadata": {
                    "name": svc.metadata.name,
                    "namespace": svc.metadata.namespace,
                    "labels": svc.metadata.labels,
                    "annotations": svc.metadata.annotations
                },
                "spec": {
                    "type": svc.spec.type,
                    "cluster_ip": svc.spec.cluster_ip,
                    "external_ips": svc.spec.external_i_ps if hasattr(svc.spec, 'external_i_ps') else None,
                    "ports": [{
                        "port": p.port,
                        "target_port": p.target_port,
                        "protocol": p.protocol,
                        "node_port": p.node_port if hasattr(p, 'node_port') else None
                    } for p in svc.spec.ports],
                    "selector": svc.spec.selector
                },
                "endpoints": [{
                    "addresses": [addr.ip for addr in subset.addresses] if subset.addresses else [],
                    "ports": [{
                        "port": p.port,
                        "protocol": p.protocol
                    } for p in subset.ports]
                } for subset in endpoints.subsets] if endpoints.subsets else []
            }
            return {"status": "success", "description": description}
        except ApiException as e:
            logger.error(f"Error describing service {name}: {e}")
            return {"status": "error", "message": str(e)}

    def get_configmaps(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        namespace = params.get("namespace", "default")
        label_selector = params.get("label_selector", "")
        
        try:
            configmaps = self.v1.list_namespaced_config_map(namespace=namespace, label_selector=label_selector)
            configmap_list = []
            for cm in configmaps.items:
                configmap_list.append({
                    "name": cm.metadata.name,
                    "namespace": cm.metadata.namespace,
                    "data_keys": list(cm.data.keys()) if cm.data else [],
                    "binary_data_keys": list(cm.binary_data.keys()) if cm.binary_data else []
                })
            return {"status": "success", "configmaps": configmap_list}
        except ApiException as e:
            logger.error(f"Error getting configmaps: {e}")
            return {"status": "error", "message": str(e)}

    def get_secrets(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        namespace = params.get("namespace", "default")
        label_selector = params.get("label_selector", "")
        
        try:
            secrets = self.v1.list_namespaced_secret(namespace=namespace, label_selector=label_selector)
            secret_list = []
            for secret in secrets.items:
                secret_list.append({
                    "name": secret.metadata.name,
                    "namespace": secret.metadata.namespace,
                    "type": secret.type,
                    "data_keys": list(secret.data.keys()) if secret.data else []
                })
            return {"status": "success", "secrets": secret_list}
        except ApiException as e:
            logger.error(f"Error getting secrets: {e}")
            return {"status": "error", "message": str(e)}

    def get_storage_classes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get all available StorageClasses in the cluster."""
        err = self._require_k8s()
        if err:
            return err
        
        try:
            api_instance = client.StorageV1Api()
            storage_classes = api_instance.list_storage_class()
            sc_names = [sc.metadata.name for sc in storage_classes.items]
            return {"status": "success", "storage_classes": sc_names}
        except ApiException as e:
            logger.error(f"Error getting StorageClasses: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error getting StorageClasses: {e}")
            return {"status": "error", "message": str(e)}

    def get_contexts(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        try:
            contexts = config.list_kube_config_contexts()
            current = config.current_context
            return {
                "status": "success",
                "contexts": [ctx["name"] for ctx in contexts[0]],
                "current_context": current
            }
        except config.config_exception.ConfigException as e:
            logger.error(f"Error getting contexts: {e}")
            return {"status": "error", "message": str(e)}

    def use_context(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        context = params.get("context")
        if not context:
            return {"status": "error", "message": "Context name is required"}
            
        try:
            config.load_kube_config(context=context)
            return {"status": "success", "message": f"Switched to context {context}"}
        except config.config_exception.ConfigException as e:
            logger.error(f"Error switching context: {e}")
            return {"status": "error", "message": str(e)}

    def apply_yaml(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        yaml_content = params.get("yaml")
        if not yaml_content:
            return {"status": "error", "message": "YAML content is required"}
            
        try:
            resources = list(yaml.safe_load_all(yaml_content))
            results = []
            
            for resource in resources:
                kind = resource["kind"]
                name = resource["metadata"]["name"]
                namespace = resource["metadata"].get("namespace", "default")
                
                try:
                    # Create/update the resource using the appropriate API
                    if kind.lower() == "persistentvolumeclaim":
                        self.v1.replace_namespaced_persistent_volume_claim(
                            name=name,
                            namespace=namespace,
                            body=resource
                        )
                    elif kind.lower() == "configmap":
                        self.v1.replace_namespaced_config_map(
                            name=name,
                            namespace=namespace,
                            body=resource
                        )
                    elif kind.lower() == "job":
                        self.batch_v1.replace_namespaced_job(
                            name=name,
                            namespace=namespace,
                            body=resource
                        )
                    elif kind.lower() == "deployment":
                        self.apps_v1.replace_namespaced_deployment(
                            name=name,
                            namespace=namespace,
                            body=resource
                        )
                    elif kind.lower() == "service":
                        self.v1.replace_namespaced_service(
                            name=name,
                            namespace=namespace,
                            body=resource
                        )
                    
                    results.append({
                        "kind": kind,
                        "name": name,
                        "namespace": namespace,
                        "status": "applied"
                    })
                except ApiException as e:
                    if e.status == 404:  # Resource doesn't exist, create it
                        if kind.lower() == "persistentvolumeclaim":
                            self.v1.create_namespaced_persistent_volume_claim(
                                namespace=namespace,
                                body=resource
                            )
                        elif kind.lower() == "configmap":
                            self.v1.create_namespaced_config_map(
                                namespace=namespace,
                                body=resource
                            )
                        elif kind.lower() == "job":
                            self.batch_v1.create_namespaced_job(
                                namespace=namespace,
                                body=resource
                            )
                        elif kind.lower() == "deployment":
                            self.apps_v1.create_namespaced_deployment(
                                namespace=namespace,
                                body=resource
                            )
                        elif kind.lower() == "service":
                            self.v1.create_namespaced_service(
                                namespace=namespace,
                                body=resource
                            )
                        
                        results.append({
                            "kind": kind,
                            "name": name,
                            "namespace": namespace,
                            "status": "created"
                        })
                    else:
                        raise
            
            return {"status": "success", "results": results}
        except Exception as e:
            logger.error(f"Error applying YAML: {e}")
            return {"status": "error", "message": str(e)}

    def delete_resource(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        kind = params.get("kind", "").lower()
        name = params.get("name")
        namespace = params.get("namespace", "default")
        
        if not kind or not name:
            return {"status": "error", "message": "Resource kind and name are required"}
            
        try:
            if kind == "deployment":
                self.apps_v1.delete_namespaced_deployment(name=name, namespace=namespace)
            elif kind == "service":
                self.v1.delete_namespaced_service(name=name, namespace=namespace)
            elif kind == "pod":
                self.v1.delete_namespaced_pod(name=name, namespace=namespace)
            elif kind == "configmap":
                self.v1.delete_namespaced_config_map(name=name, namespace=namespace)
            elif kind == "secret":
                self.v1.delete_namespaced_secret(name=name, namespace=namespace)
            elif kind == "job":
                self.batch_v1.delete_namespaced_job(name=name, namespace=namespace)
            else:
                return {"status": "error", "message": f"Unsupported resource kind: {kind}"}
                
            return {
                "status": "success",
                "message": f"Deleted {kind} {name} in namespace {namespace}"
            }
        except ApiException as e:
            logger.error(f"Error deleting resource: {e}")
            return {"status": "error", "message": str(e)}

    def create_resource(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        kind = params.get("kind", "").lower()
        name = params.get("name")
        namespace = params.get("namespace", "default")
        spec = params.get("spec", {})
        
        if not kind or not name:
            return {"status": "error", "message": "Resource kind and name are required"}
            
        try:
            body = {
                "apiVersion": "v1",
                "kind": kind.capitalize(),
                "metadata": {
                    "name": name,
                    "namespace": namespace
                },
                "spec": spec
            }
            
            if kind == "deployment":
                body["apiVersion"] = "apps/v1"
                result = self.apps_v1.create_namespaced_deployment(namespace=namespace, body=body)
            elif kind == "service":
                result = self.v1.create_namespaced_service(namespace=namespace, body=body)
            elif kind == "pod":
                result = self.v1.create_namespaced_pod(namespace=namespace, body=body)
            elif kind == "configmap":
                result = self.v1.create_namespaced_config_map(namespace=namespace, body=body)
            elif kind == "secret":
                result = self.v1.create_namespaced_secret(namespace=namespace, body=body)
            else:
                return {"status": "error", "message": f"Unsupported resource kind: {kind}"}
                
            return {
                "status": "success",
                "message": f"Created {kind} {name} in namespace {namespace}"
            }
        except ApiException as e:
            logger.error(f"Error creating resource: {e}")
            return {"status": "error", "message": str(e)}

    def port_forward(self, params):
        err = self._require_k8s()
        if err:
            return err
        
        pod_name = params.get("pod")
        namespace = params.get("namespace", "default")
        local_port = params.get("local_port")
        remote_port = params.get("remote_port")
        
        if not pod_name or not local_port or not remote_port:
            return {
                "status": "error",
                "message": "Pod name, local port, and remote port are required"
            }
            
        try:
            pf = portforward(
                self.v1.connect_get_namespaced_pod_portforward,
                pod_name,
                namespace,
                ports=[f"{local_port}:{remote_port}"]
            )
            return {
                "status": "success",
                "message": f"Port forwarding established from localhost:{local_port} to {pod_name}:{remote_port}",
                "pod": pod_name,
                "namespace": namespace,
                "local_port": local_port,
                "remote_port": remote_port
            }
        except Exception as e:
            logger.error(f"Error setting up port forward: {e}")
            return {"status": "error", "message": str(e)}

    def query_prometheus(self, params):
        """
        Query Prometheus for metrics.
        
        Args:
            params: Dictionary containing query parameters.
        
        Returns:
            Dictionary containing the query results or an error message.
        """
        err = self._require_k8s()
        if err:
            return err
        
        query = params.get("query", "")
        if not query:
            return {"status": "error", "message": "Query parameter is required."}
        
        try:
            # Simulate a Prometheus query (replace with actual implementation)
            logger.info(f"Querying Prometheus with query: {query}")
            result = {"data": f"Results for query: {query}"}
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return {"status": "error", "message": str(e)}

    def query_grafana(self, params):
        """
        Query Grafana for dashboard data.
        
        Args:
            params: Dictionary containing query parameters.
        
        Returns:
            Dictionary containing the query results or an error message.
        """
        err = self._require_k8s()
        if err:
            return err
        
        dashboard_id = params.get("dashboard_id", "")
        if not dashboard_id:
            return {"status": "error", "message": "Dashboard ID is required."}
        
        try:
            # Simulate a Grafana query (replace with actual implementation)
            logger.info(f"Querying Grafana for dashboard ID: {dashboard_id}")
            result = {"data": f"Dashboard data for ID: {dashboard_id}"}
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Failed to query Grafana: {e}")
            return {"status": "error", "message": str(e)}

    def query_elasticsearch(self, params):
        """
        Query Elasticsearch for logs or metrics.
        
        Args:
            params: Dictionary containing query parameters.
        
        Returns:
            Dictionary containing the query results or an error message.
        """
        err = self._require_k8s()
        if err:
            return err
        
        index = params.get("index", "")
        query = params.get("query", "")
        if not index or not query:
            return {"status": "error", "message": "Index and query parameters are required."}
        
        try:
            # Simulate an Elasticsearch query (replace with actual implementation)
            logger.info(f"Querying Elasticsearch index: {index} with query: {query}")
            result = {"data": f"Results for index: {index}, query: {query}"}
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Failed to query Elasticsearch: {e}")
            return {"status": "error", "message": str(e)}

    def diagnose_cluster(self, params):
        """
        Diagnose the overall health of the Kubernetes cluster.
        
        Args:
            params: Dictionary containing optional parameters for diagnosis.
        
        Returns:
            Dictionary containing the diagnosis results or an error message.
        """
        err = self._require_k8s()
        if err:
            return err
        
        try:
            # Simulate a cluster diagnosis (replace with actual implementation)
            logger.info("Diagnosing cluster health...")
            result = {
                "status": "success",
                "diagnosis": {
                    "summary": "Cluster is healthy with minor warnings.",
                    "issues": [
                        {"severity": "LOW", "issue": "Some pods are in Pending state."},
                        {"severity": "MEDIUM", "issue": "Node resource usage is high on node-1."}
                    ],
                    "recommendations": [
                        {"action": "Scale up the cluster", "reason": "To handle increased workload."},
                        {"action": "Investigate Pending pods", "reason": "To ensure they are scheduled properly."}
                    ]
                }
            }
            return result
        except Exception as e:
            logger.error(f"Failed to diagnose cluster: {e}")
            return {"status": "error", "message": str(e)}

    def analyze_resource(self, params):
        """
        Analyze a specific Kubernetes resource.
        
        Args:
            params: Dictionary containing:
                - resource_type: Type of resource (e.g., pod, deployment)
                - resource_data: Resource data
                - events: Events related to the resource (optional)
        
        Returns:
            Dictionary containing analysis results or an error message.
        """
        err = self._require_k8s()
        if err:
            return err
        
        resource_type = params.get("resource_type", "unknown")
        resource_data = params.get("resource_data", {})
        events = params.get("events", [])
        
        try:
            logger.info(f"Analyzing resource of type {resource_type}")
            # Simulate resource analysis (replace with actual implementation)
            result = {
                "status": "success",
                "analysis": {
                    "summary": f"Resource {resource_type} is healthy.",
                    "issues": [],
                    "recommendations": []
                }
            }
            
            # Example: Add simulated issues and recommendations
            if resource_type == "pod" and "status" in resource_data and resource_data["status"] != "Running":
                result["analysis"]["issues"].append({
                    "severity": "HIGH",
                    "issue": "Pod is not in Running state.",
                    "details": resource_data.get("status")
                })
                result["analysis"]["recommendations"].append({
                    "action": "Check pod logs",
                    "command": f"kubectl logs {resource_data.get('name', 'unknown')} -n {resource_data.get('namespace', 'default')}",
                    "reason": "To identify why the pod is not running."
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to analyze resource: {e}")
            return {"status": "error", "message": str(e)}

    def recommend_action(self, params):
        """
        Recommend actions to resolve issues with a Kubernetes resource.
        
        Args:
            params: Dictionary containing:
                - resource_type: Type of resource (e.g., pod, deployment)
                - resource_data: Resource data
                - events: Events related to the resource (optional)
                - logs: Logs related to the resource (optional)
                - issue: Description of the issue (optional)
        
        Returns:
            Dictionary containing recommendations or an error message.
        """
        err = self._require_k8s()
        if err:
            return err
        
        resource_type = params.get("resource_type", "unknown")
        resource_data = params.get("resource_data", {})
        events = params.get("events", [])
        logs = params.get("logs", {})
        issue = params.get("issue", None)
        
        try:
            logger.info(f"Recommending actions for resource of type {resource_type}")
            # Simulate recommendations (replace with actual implementation)
            recommendations = [
                {
                    "action": "Restart the resource",
                    "command": f"kubectl rollout restart {resource_type}/{resource_data.get('name', 'unknown')} -n {resource_data.get('namespace', 'default')}",
                    "reason": "To resolve potential transient issues."
                },
                {
                    "action": "Check resource logs",
                    "command": f"kubectl logs {resource_data.get('name', 'unknown')} -n {resource_data.get('namespace', 'default')}",
                    "reason": "To identify potential root causes."
                }
            ]
            
            return {"status": "success", "recommendations": recommendations}
        except Exception as e:
            logger.error(f"Failed to recommend actions: {e}")
            return {"status": "error", "message": str(e)}

    def deploy_ml_stack(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a complete ML training and inference stack."""
        
        name = params.get("name", f"ml-{int(time.time())}")
        framework = params.get("framework", "pytorch")
        training_code = params.get("training_code", "")
        inference_code = params.get("inference_code", "")
        requirements_content = params.get("requirements_txt", "")
        storage_class = params.get("storage_class")

        # Dynamically determine storage class if not provided
        if not storage_class:
            sc_result = self.get_storage_classes({})
            if sc_result["status"] == "success" and sc_result["storage_classes"]:
                if "standard" in sc_result["storage_classes"]:
                    storage_class = "standard"
                else:
                    storage_class = sc_result["storage_classes"][0]
                logger.info(f"Using StorageClass: {storage_class}")
            else:
                return {"status": "error", "message": "No StorageClasses found in the cluster."}

        # Create ConfigMap for requirements
        requirements_yaml = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{name}-requirements"
            },
            "data": {
                "requirements.txt": requirements_content
            }
        }

        # Create ConfigMap for training code
        train_code_cm_yaml = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{name}-train-code"
            },
            "data": {
                "train.py": training_code
            }
        }

        # Create ConfigMap for inference code
        inference_code_cm_yaml = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{name}-inference-code"
            },
            "data": {
                "inference.py": inference_code
            }
        }

        # Create PVC
        pvc_yaml = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": f"{name}-models"},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": storage_class,
                "resources": {"requests": {"storage": "1Gi"}}
            }
        }

        # Create training job with init container for dependencies
        training_yaml = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": f"{name}-training"},
            "spec": {
                "template": {
                    "spec": {
                        "initContainers": [{
                            "name": "install-deps",
                            "image": "python:3.9",
                            "command": ["pip", "install", "-r", "/opt/app/requirements.txt"],
                            "volumeMounts": [{
                                "name": "requirements",
                                "mountPath": "/opt/app"
                            }]
                        }],
                        "containers": [{
                            "name": "trainer",
                            "image": "python:3.9",
                            "command": ["python", "/opt/app/train.py"],
                            "volumeMounts": [
                                {
                                    "name": "model-storage",
                                    "mountPath": "/models"
                                },
                                {
                                    "name": "training-code",
                                    "mountPath": "/opt/app"
                                },
                                {
                                    "name": "requirements",
                                    "mountPath": "/opt/app"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "2Gi"
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": "4Gi"
                                }
                            }
                        }],
                        "volumes": [
                            {
                                "name": "model-storage",
                                "persistentVolumeClaim": {
                                    "claimName": f"{name}-models"
                                }
                            },
                            {
                                "name": "requirements",
                                "configMap": {
                                    "name": f"{name}-requirements"
                                }
                            },
                            {
                                "name": "training-code",
                                "configMap": {
                                    "name": f"{name}-train-code"
                                }
                            }
                        ],
                        "restartPolicy": "Never"
                    }
                }
            }
        }

        # Create inference deployment with init container
        inference_yaml = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": f"{name}-inference"},
            "spec": {
                "replicas": params.get("replicas", 1),
                "selector": {"matchLabels": {"app": f"{name}-inference"}},
                "template": {
                    "metadata": {"labels": {"app": f"{name}-inference"}},
                    "spec": {
                        "initContainers": [{
                            "name": "install-deps",
                            "image": "python:3.9",
                            "command": ["pip", "install", "-r", "/opt/app/requirements.txt"],
                            "volumeMounts": [{
                                "name": "requirements",
                                "mountPath": "/opt/app"
                            }]
                        }],
                        "containers": [{
                            "name": "inference",
                            "image": "python:3.9",
                            "command": ["python", "/opt/app/inference.py"],
                            "volumeMounts": [
                                {
                                    "name": "model-storage",
                                    "mountPath": "/models"
                                },
                                {
                                    "name": "inference-code",
                                    "mountPath": "/opt/app"
                                },
                                {
                                    "name": "requirements",
                                    "mountPath": "/opt/app"
                                }
                            ],
                            "ports": [{"containerPort": 8080}],
                            "resources": {
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "1Gi"
                                },
                                "limits": {
                                    "cpu": "1",
                                    "memory": "2Gi"
                                }
                            }
                        }],
                        "volumes": [
                            {
                                "name": "model-storage",
                                "persistentVolumeClaim": {
                                    "claimName": f"{name}-models"
                                }
                            },
                            {
                                "name": "requirements",
                                "configMap": {
                                    "name": f"{name}-requirements"
                                }
                            },
                            {
                                "name": "inference-code",
                                "configMap": {
                                    "name": f"{name}-inference-code"
                                }
                            }
                        ]
                    }
                }
            }
        }

        # Create service for inference
        service_yaml = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{name}-inference"},
            "spec": {
                "selector": {"app": f"{name}-inference"},
                "ports": [{"port": 8080, "targetPort": 8080}],
                "type": "ClusterIP"
            }
        }

        try:
            # Apply resources in order
            resources = [
                ("Requirements ConfigMap", requirements_yaml),
                ("Training Code ConfigMap", train_code_cm_yaml),
                ("Inference Code ConfigMap", inference_code_cm_yaml),
                ("PVC", pvc_yaml),
                ("Training Job", training_yaml),
                ("Inference Deployment", inference_yaml),
                ("Service", service_yaml)
            ]

            results = []
            for resource_type, resource in resources:
                result = self.apply_yaml({"yaml": yaml.dump(resource)})
                if result["status"] != "success":
                    return {
                        "status": "error",
                        "message": f"Failed to create {resource_type}",
                        "details": result
                    }
                results.append(result)

            return {
                "status": "success",
                "name": name,
                "resources": results,
                "endpoints": {
                    "inference": f"http://{name}-inference:8080/predict"
                }
            }

        except Exception as e:
            logger.error(f"Failed to deploy ML stack: {e}")
            return {"status": "error", "message": str(e)}
