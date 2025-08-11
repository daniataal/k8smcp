import logging
import json
import yaml
from typing import Dict, Any, List, Optional, Tuple
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from command_router import CommandRouter  # Removed relative import

logger = logging.getLogger(__name__)

class KubernetesDebugger:
    def __init__(self, claude_analyzer, k8s_available=True):
        self.k8s_available = k8s_available
        if k8s_available:
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
        return {"status": "success", "logs": "Example log content"}

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
        return {"status": "success", "description": {"name": params.get("name", "example-node")}}

    def check_node_pressure(self, params):
        err = self._require_k8s()
        if err:
            return err
        return {"status": "success", "pressure": {"example-node": "Normal"}}

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
        return {"status": "success", "result": "rollout managed"}

    def get_services(self, params):
        err = self._require_k8s()
        if err:
            return err
        return {"status": "success", "services": [{"name": "example-service", "namespace": "default"}]}

    def describe_service(self, params):
        err = self._require_k8s()
        if err:
            return err
        return {"status": "success", "description": {"name": params.get("name", "example-service")}}

    def get_configmaps(self, params):
        err = self._require_k8s()
        if err:
            return err
        return {"status": "success", "configmaps": [{"name": "example-configmap", "namespace": "default"}]}

    def get_secrets(self, params):
        err = self._require_k8s()
        if err:
            return err
        return {"status": "success", "secrets": [{"name": "example-secret", "namespace": "default"}]}

    def get_contexts(self, params):
        err = self._require_k8s()
        if err:
            return err  # Return the error response if Kubernetes is not available
        return {"status": "success", "contexts": ["context1", "context2"]}

    def use_context(self, params):
        err = self._require_k8s()
        if err:
            return err  # Return the error response if Kubernetes is not available
        context = params.get("context", "default-context")
        return {"status": "success", "result": f"Switched to context {context}"}

    def apply_yaml(self, params):
        err = self._require_k8s()
        if err:
            return err  # Return the error response if Kubernetes is not available
        yaml_content = params.get("yaml", "")
        return {"status": "success", "result": "Applied YAML successfully"}

    def delete_resource(self, params):
        err = self._require_k8s()
        if err:
            return err  # Return the error response if Kubernetes is not available
        resource_name = params.get("name", "example-resource")
        return {"status": "success", "result": f"Deleted resource {resource_name}"}

    def create_resource(self, params):
        err = self._require_k8s()
        if err:
            return err  # Return the error response if Kubernetes is not available
        resource_name = params.get("name", "example-resource")
        return {"status": "success", "result": f"Created resource {resource_name}"}

    def port_forward(self, params):
        err = self._require_k8s()
        if err:
            return err  # Return the error response if Kubernetes is not available
        pod_name = params.get("pod", "example-pod")
        port = params.get("port", 8080)
        return {"status": "success", "result": f"Port-forwarded {pod_name} on port {port}"}

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
