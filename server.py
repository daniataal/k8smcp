#!/usr/bin/env python3

import os
import sys
import logging

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if (current_dir not in sys.path):
    sys.path.append(current_dir)

from fastmcp import FastMCP
from kubernetes import client, config
from k8s_debugger import KubernetesDebugger
from claude_analyzer import ClaudeAnalyzer

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
