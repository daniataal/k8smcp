# Kubernetes MCP Debugger

A comprehensive debugging tool for Kubernetes clusters that integrates with Claude AI for intelligent analysis and recommendations.

## Features

- 🔍 **Complete Cluster Visibility**: Debug all Kubernetes resources including pods, nodes, deployments, services, etc.
- 🤖 **AI-Powered Analysis**: Integrates with Claude AI to analyze issues and provide intelligent recommendations
- 📊 **Monitoring Integration**: Connect with Prometheus, Grafana, Elasticsearch, and other monitoring systems
- 🛠️ **Full kubectl Functionality**: Execute all kubectl commands including apply, delete, scale, and rollout
- 🔄 **Real-time Debugging**: Port forwarding, log streaming, and exec into containers

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your Claude API key:

```bash
export CLAUDE_API_KEY="your-api-key-here"
```

## Usage

### With Claude Desktop

If you want to use the `uv` package manager for faster startup, install it first:

```bash
pip install uv
```

Or follow instructions at https://github.com/astral-sh/uv

If `uv` is not installed or not found at `/usr/local/bin/uv`, use `python` directly in your Claude Desktop config:

```json
{
  "mcpServers": {
    "kubernetes-debugger": {
      "command": "python",
      "args": ["path/to/folder/k8smcp/main.py"],
      "env": {
        "CLAUDE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Add this configuration to your Claude Desktop config file:

**On macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**On Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "kubernetes-debugger": {
      "command": "/usr/local/bin/uv",
      "args": [
        "--directory",
        "/path/to/folder/k8smcp",
        "run",
        "main.py"
      ],
      "env": {
        "CLAUDE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Or if you prefer using Python directly:

```json
{
  "mcpServers": {
    "kubernetes-debugger": {
      "command": "python",
      "args": ["/path/to/folder/k8smcp/main.py"],
      "env": {
        "CLAUDE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

**Example Claude Desktop config with custom kubeconfig:**

```json
{
  "mcpServers": {
    "kubernetes-debugger": {
      "command": "python",
      "args": ["/path/to/folder/k8smcp/main.py"],
      "env": {
        "CLAUDE_API_KEY": "your-api-key-here",
        "KUBECONFIG": "/path/to/kubeconfigfile"
      }
    }
  }
}
```

### Standalone Usage

Start the MCP server using fastmcp:

```bash
export CLAUDE_API_KEY="your-api-key-here"
fastmcp run server.py:mcp
```

Or run it directly:

```bash
python server.py
```

Or for Claude Desktop integration:

```bash
python main.py
```

**Note:**  
- The `main.py` script calls `main()` directly and does not wrap it in another event loop.
- If you encounter issues, ensure that `fastmcp` is installed and properly configured.

### Example Commands in Claude

Once configured, you can ask Claude to help with Kubernetes debugging:

```
"Can you show me all pods in the kube-system namespace?"

"Check if there are any nodes with pressure conditions"

"Analyze the frontend deployment in the default namespace"

"Get logs from the database pod and analyze any errors"

"What's the health status of my cluster?"
```

## Available Tools

The MCP server provides these tools that Claude can use:

- `get_pods` - List pods in a namespace
- `describe_pod` - Get detailed pod information
- `get_pod_logs` - Retrieve and optionally analyze pod logs
- `get_nodes` - List cluster nodes
- `describe_node` - Get detailed node information
- `check_node_pressure` - Check node pressure conditions
- `get_deployments` - List deployments
- `describe_deployment` - Get detailed deployment information
- `scale_deployment` - Scale a deployment
- `diagnose_cluster` - Perform cluster health diagnosis
- `analyze_resource` - AI-powered resource analysis
- `recommend_action` - Get AI recommendations for issues

## Configuration

Environment variables:
- `CLAUDE_API_KEY`: Your Claude API key for AI analysis features
- `KUBECONFIG`: Path to kubeconfig file (optional, uses default location)

## Requirements

- Python 3.8+
- kubectl configured with cluster access
- Claude API key (optional, for AI analysis features)
- uv package manager (recommended) or pip

## Architecture

The system consists of the following components:

1. **FastMCP Server**: Handles communication with MCP clients
2. **Kubernetes Debugger**: Interfaces with the Kubernetes API
3. **Command Router**: Routes commands to appropriate handler functions
4. **Claude Analyzer**: Provides AI-powered analysis and recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
