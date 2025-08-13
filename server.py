#!/usr/bin/env python3

import os
import sys
import logging
from typing import Dict, Any, List, Optional  # Add this import line

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_kubernetes():
    """Setup Kubernetes configuration. Returns True if successful, False otherwise."""
    try:
        config.load_kube_config()
        logger.info("Loaded kubeconfig successfully")
        return True
    except Exception as e1:
        logger.warning(f"Failed to load kubeconfig: {e1}")
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster config successfully")
            return True
        except Exception as e2:
            logger.error(f"Failed to load Kubernetes config: {e2}")
            return False

def create_mcp_server():
    """Create and configure the MCP server"""
    try:
        # Set up Kubernetes
        k8s_available = setup_kubernetes()
        if not k8s_available:
            logger.warning("Kubernetes API is not available. MCP server will start with limited functionality.")

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

        # Initialize artifact manager
        artifact_manager = ArtifactManager(base_dir=current_dir)

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
            """Get pods from a Kubernetes namespace"""
            return k8s_debugger.get_pods({
                "namespace": namespace,
                "label_selector": label_selector,
                "field_selector": field_selector
            })

        @mcp.tool()
        def describe_pod(name: str, namespace: str = "default"):
            """Describe a specific Kubernetes pod"""
            return k8s_debugger.describe_pod({
                "name": name,
                "namespace": namespace
            })

        @mcp.tool()
        def get_pod_logs(name: str, namespace: str = "default", container: str = None, 
                         tail_lines: int = 100, previous: bool = False, analyze: bool = False):
            """Get logs from a Kubernetes pod"""
            return k8s_debugger.get_pod_logs({
                "name": name,
                "namespace": namespace,
                "container": container,
                "tail_lines": tail_lines,
                "previous": previous,
                "analyze": analyze
            })

        @mcp.tool()
        def get_nodes(label_selector: str = "", field_selector: str = ""):
            """Get Kubernetes nodes"""
            return k8s_debugger.get_nodes({
                "label_selector": label_selector,
                "field_selector": field_selector
            })

        @mcp.tool()
        def describe_node(name: str):
            """Describe a specific Kubernetes node"""
            return k8s_debugger.describe_node({"name": name})

        @mcp.tool()
        def check_node_pressure(name: str):
            """Check pressure conditions on a Kubernetes node"""
            return k8s_debugger.check_node_pressure({"name": name})

        @mcp.tool()
        def get_deployments(namespace: str = "default", all_namespaces: bool = False, label_selector: str = ""):
            """Get Kubernetes deployments"""
            return k8s_debugger.get_deployments({
                "namespace": namespace,
                "all_namespaces": all_namespaces,
                "label_selector": label_selector
            })

        @mcp.tool()
        def describe_deployment(name: str, namespace: str = "default"):
            """Describe a specific Kubernetes deployment"""
            return k8s_debugger.describe_deployment({
                "name": name,
                "namespace": namespace
            })

        @mcp.tool()
        def scale_deployment(name: str, replicas: int, namespace: str = "default"):
            """Scale a Kubernetes deployment"""
            return k8s_debugger.scale_deployment({
                "name": name,
                "replicas": replicas,
                "namespace": namespace
            })

        @mcp.tool()
        def diagnose_cluster():
            """Perform cluster health diagnosis"""
            return k8s_debugger.diagnose_cluster({})

        @mcp.tool()
        def analyze_resource(resource_type: str, resource_data: dict, events: list = None):
            """AI-powered resource analysis"""
            return k8s_debugger.analyze_resource({
                "resource_type": resource_type,
                "resource_data": resource_data,
                "events": events or []
            })

        @mcp.tool()
        def recommend_action(resource_type: str, resource_data: dict, events: list = None, 
                           logs: dict = None, issue: str = None):
            """Get AI recommendations for issues"""
            return k8s_debugger.recommend_action({
                "resource_type": resource_type,
                "resource_data": resource_data,
                "events": events or [],
                "logs": logs or {},
                "issue": issue
            })

        @mcp.tool()
        def apply_yaml(yaml_content: str, dry_run: bool = False, server_side: bool = False, force: bool = False):
            """Apply YAML content to the cluster"""
            return k8s_debugger.apply_yaml({
                "yaml": yaml_content,
                "dry_run": dry_run,
                "server_side": server_side,
                "force": force
            })

        # --- DVC tools ---
        @mcp.tool()
        def dvc_init():
            """Initialize DVC in the current repo"""
            return dvc_manager.init()

        @mcp.tool()
        def dvc_add(path: str):
            """Add a file or directory to DVC tracking"""
            return dvc_manager.add(path)

        @mcp.tool()
        def dvc_push():
            """Push DVC-tracked data to remote storage"""
            return dvc_manager.push()

        @mcp.tool()
        def dvc_pull():
            """Pull DVC-tracked data from remote storage"""
            return dvc_manager.pull()

        @mcp.tool()
        def dvc_status():
            """Get DVC status"""
            return dvc_manager.status()

        @mcp.tool()
        def dvc_repro():
            """Reproduce DVC pipeline"""
            return dvc_manager.repro()

        # --- High-level MLOps workflow tool ---
        @mcp.tool()
        def mlops_generate_code(prompt: str, job_name: str = None):
            """
            Generate ML training/inference code, Dockerfiles, and K8s YAMLs from a high-level prompt.
            """
            job_dir = mlops_workflow.generate_job_dir(job_name)
            return mlops_workflow.generate_code_and_configs(prompt, job_dir)

        @mcp.tool()
        def mlops_list_jobs():
            """List all generated MLOps job directories and their files."""
            return mlops_workflow.list_jobs()

        @mcp.tool()
        def mlops_get_job_files(job_id: str):
            """Get all files and their contents for a given job."""
            return mlops_workflow.get_job_files(job_id)

        @mcp.tool()
        def mlops_build_image(job_id: str, dockerfile: str, image_tag: str):
            """
            Build a container image using Podman for a given job and Dockerfile.
            """
            job_dir = os.path.join(current_dir, "mlops_jobs", job_id)
            return mlops_workflow.build_image(job_dir, dockerfile, image_tag)

        @mcp.tool()
        def mlops_push_image(image_tag: str, job_id: str = None):
            """
            Push a container image to a registry using Podman.
            """
            job_dir = os.path.join(current_dir, "mlops_jobs", job_id) if job_id else None
            return mlops_workflow.push_image(image_tag, job_dir=job_dir)

        @mcp.tool()
        def mlops_manage_images(job_id: str, training_tag: str = None, inference_tag: str = None):
            """
            Build and push training and/or inference images for a job using Podman.
            Args:
                job_id: Job ID
                training_tag: Tag for training image (optional)
                inference_tag: Tag for inference image (optional)
            """
            return mlops_workflow.manage_images(job_id, training_tag, inference_tag)

        # --- Artifact Management Tools ---
        @mcp.tool()
        def mlops_extract_model(job_id: str, pod_name: str, namespace: str = "default",
                              container: str = None, model_path: str = "/app/model"):
            """Extract model artifacts from a training pod"""
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
            """Copy model artifacts to an inference pod"""
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
            """Create persistent storage for model artifacts"""
            return artifact_manager.create_model_pvc(
                job_id=job_id,
                namespace=namespace,
                storage_class=storage_class,
                size=size
            )

        @mcp.tool()
        def mlops_cleanup_artifacts(job_id: str):
            """Clean up artifacts for a specific job"""
            return artifact_manager.cleanup_artifacts(job_id)

        # --- Inference Service Management Tools ---
        @mcp.tool()
        def mlops_deploy_inference(job_id: str, image_tag: str, namespace: str = "default",
                                 replicas: int = 1, resource_requests: dict = None,
                                 resource_limits: dict = None):
            """Deploy an inference service for a trained model"""
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
            """Update an existing inference service"""
            return mlops_workflow.update_inference(
                job_id=job_id,
                namespace=namespace,
                image_tag=image_tag,
                replicas=replicas
            )

        @mcp.tool()
        def mlops_get_inference_status(job_id: str, namespace: str = "default"):
            """Get status of an inference service"""
            return mlops_workflow.get_inference_status(
                job_id=job_id,
                namespace=namespace
            )

        # --- High-level Workflow Tools ---
        @mcp.tool()
        def mlops_execute_workflow(prompt: str):
            """
            Execute a complete MLOps workflow from a natural language prompt.
            Examples:
            - "Train an MNIST classifier using PyTorch and deploy it"
            - "Build and deploy a sentiment analysis model"
            - "Create an object detection service using YOLOv8"
            """
            return workflow_orchestrator.execute_workflow(prompt)

        @mcp.tool()
        def mlops_run_inference(job_id: str, data: Any):
            """Run inference on a deployed model"""
            return workflow_orchestrator.run_inference(job_id, data)

        # --- LLM and Recommendation Model Tools ---
        @mcp.tool()
        def mlops_create_recommendation_model(task_description: str, data_format: str, features: List[str]):
            """
            Create a recommendation model from description.
            
            Examples:
            - "Product recommendation system based on user purchase history"
            - "Movie recommender using user ratings and genres"
            - "Content recommendation based on user browsing behavior"
            """
            return mlops_workflow.create_recommendation_model(
                task_description=task_description,
                data_format=data_format,
                features=features
            )

        @mcp.tool()
        def mlops_finetune_llm(task_description: str, data_format: str, framework: str = "pytorch"):
            """
            Generate and set up LLM fine-tuning pipeline.
            
            Examples:
            - "Fine-tune for sentiment analysis on product reviews"
            - "Adapt language model for medical text classification"
            - "Customize LLM for code completion in Python"
            """
            return mlops_workflow.finetune_llm(
                task_description=task_description,
                data_format=data_format,
                framework=framework
            )

        return mcp

    except Exception as e:
        logger.error(f"Failed to create MCP server: {e}")
        raise

# Expose the MCP server instance for import
mcp = create_mcp_server()

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run("server:mcp", host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        logger.error(f"Failed to start the server: {e}")
        sys.exit(1)
