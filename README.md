# Kubernetes Microservice Control Plane (K8sMCP)

This project provides a Kubernetes Microservice Control Plane (K8sMCP) with an integrated UI, allowing for streamlined management and interaction with Kubernetes clusters and MLOps workflows. The backend exposes various functionalities via a RESTful API, consumed by the Next.js frontend.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Kubernetes Debugging & Management:** Interact with Kubernetes resources (pods, nodes, deployments) through a high-level API.
- **MLOps Workflow Automation:** Generate, build, and deploy ML models and finetune LLMs.
- **Model Registry:** Register and manage trained models with metadata.
- **Artifact Management:** Handle model artifacts with persistent storage.
- **Dynamic Tool Execution:** A generic API endpoint to call any registered MCP tool.
- **Interactive UI:** A user-friendly Next.js application for interacting with the backend.

## Architecture

The application consists of two main components:

1.  **Backend (Python Flask & FastMCP):**
    -   `server.py`: The core backend application, exposing a RESTful API using Flask.
    -   `FastMCP`: An underlying framework that provides a Microservice Control Plane, registering various tools for Kubernetes interaction and MLOps workflows.
    -   `Kubernetes Client`: Interacts with the Kubernetes API.
    -   `Claude Analyzer`: Integrates with Claude API for AI-powered analysis.
    -   `DVC Manager`: Manages data versioning with DVC.
    -   `MLOpsWorkflow`: Orchestrates MLOps training, inference, and model management.

2.  **Frontend (Next.js/React):**
    -   A modern web interface built with Next.js, providing a dashboard for Kubernetes and MLOps operations.
    -   Communicates with the backend API to trigger actions and display status.

## Prerequisites

Before you begin, ensure you have the following installed:

-   Python 3.8+
-   Node.js (LTS version) & npm or yarn
-   Docker (Docker Desktop) or Podman (for building and running containers)
-   Kubernetes cluster (Minikube, Kind, or a cloud-based cluster) with `kubectl` configured
-   (Optional) Claude API Key: Set as an environment variable `CLAUDE_API_KEY` for AI analysis features.

## Setup and Installation

### Local Development Setup (without Docker)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd k8smcp
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Python dependencies (for backend and database):**
    ```bash
    pip install -r requirements.txt
    ```

4.  **PostgreSQL Setup (for local development):**
    -   Ensure you have a PostgreSQL database running and accessible.
    -   Set the `DATABASE_URL` environment variable. Example:
        ```bash
        export DATABASE_URL="postgresql://user:password@localhost:5432/k8smcp_db"
        ```
        Replace `user`, `password`, `localhost:5432`, and `k8smcp_db` with your PostgreSQL credentials and database information.

5.  **Kubernetes Configuration:**
    Ensure your `kubectl` is configured to connect to your Kubernetes cluster. The backend will attempt to load `kubeconfig` or in-cluster configuration.

### Docker Compose Setup (Recommended for Full Stack)

This setup uses Docker Compose to run the backend, frontend, and a PostgreSQL database as interconnected services.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd k8smcp
    ```

2.  **Ensure Docker is running.**

3.  **Build and run the services:**
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker images for the backend and frontend, and then start all services, including PostgreSQL. The first build might take some time.

4.  **Set Claude API Key (if needed):**
    If you want to enable Claude AI features, ensure your `CLAUDE_API_KEY` environment variable is set in your host machine's environment before running `docker-compose up`.

## Running the Application

### Running Locally (without Docker Compose)

#### Running the Backend

Make sure your virtual environment is activated.

```bash
python3 server.py
```

The backend server will start on `http://0.0.0.0:8080`.

#### Running the Frontend

Open a new terminal, navigate to the `frontend` directory, and run:

```bash
cd frontend
npm run dev # or yarn dev
```

The frontend application will be accessible at `http://localhost:3000` (or another port if configured).

### Running with Docker Compose (Recommended)

Once `docker-compose up --build` is successfully run, the services will be accessible:

-   **Backend API:** `http://localhost:8080`
-   **Frontend UI:** `http://localhost:3000`

To stop the services:

```bash
docker-compose down
```

### Connecting Kubernetes `kubeconfig`

After launching both the backend and frontend (either locally or via Docker Compose), you will connect your Kubernetes `kubeconfig` through the frontend UI. This action will trigger the backend to initialize the FastMCP server and enable all Kubernetes and MLOps-related functionalities.

### Data Migration (Optional)

If you have existing model data in `model_registry/registered_models.json` from a previous setup, you can migrate it to the PostgreSQL database using the provided migration script. **Ensure your PostgreSQL database is running and `DATABASE_URL` is configured (either directly or via Docker Compose) before running this.**

```bash
python3 migrate_data.py
```

After successful migration, you can optionally delete the `model_registry/registered_models.json` file.

## API Endpoints

-   `GET /health`: Checks the health of the Flask application.
-   `GET /api/mcp-status`: Checks if the FastMCP server is initialized and Kubernetes client is configured.
-   `POST /api/connect-kubeconfig`: Accepts Kubernetes `kubeconfig` content to initialize the FastMCP server.
    -   **Request Body Example:**
        ```json
        {
            "kubeconfig": "<your-kubeconfig-content-as-string>"
        }
        ```
-   `POST /predict`: (Dummy) Endpoint for model prediction. In a real scenario, this would trigger inference on a deployed model.
-   `POST /api/apply-yaml`: Applies Kubernetes YAML content to the cluster. (Requires MCP initialized)
-   `GET /api/kubernetes/pods`: Lists Kubernetes pods. (Requires MCP initialized)
-   `GET /api/mlops/jobs`: Lists MLOps job directories. (Requires MCP initialized)
-   `GET /api/mlops/models`: Lists registered MLOps models. (Requires MCP initialized)
-   `GET /api/mlops/model/<model_id>`: Retrieves details for a specific registered model. (Requires MCP initialized)
-   `POST /api/mlops/deploy-model`: Deploys an inference service for a registered model. (Requires MCP initialized)
-   `POST /api/mlops/run-inference`: Runs inference on a deployed model. (Requires MCP initialized)
-   `POST /api/call-tool`: Generic endpoint to call any FastMCP tool by name. (Requires MCP initialized)
    -   **Request Body Example:**
        ```json
        {
            "tool_name": "get_deployments",
            "tool_args": {
                "namespace": "my-namespace",
                "label_selector": "app=my-app"
            }
        }
        ```

## Project Structure

-   `k8smcp/`:
    -   `server.py`: Backend application.
    -   `database.py`: SQLAlchemy database setup and models.
    -   `model_registry.py`: Logic for interacting with the model registry (now uses PostgreSQL).
    -   `migrate_data.py`: Script for migrating old JSON data to PostgreSQL.
    -   `Dockerfile.backend`: Dockerfile for the backend service.
    -   `docker-compose.yml`: Docker Compose file for orchestrating services.
    -   `artifact_manager.py`, `claude_analyzer.py`, `dvc_manager.py`, `k8s_debugger.py`, `mlops_workflow.py`, `workflow_orchestrator.py`: Core MCP tool implementations.
    -   `requirements.txt`: Python dependencies.
-   `frontend/`:
    -   `app/`: Next.js application pages.
    -   `components/`: React components (UI elements, dashboard sections).
    -   `public/`: Static assets.
    -   `package.json`: Node.js dependencies.
    -   `Dockerfile.frontend`: Dockerfile for the frontend service.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
