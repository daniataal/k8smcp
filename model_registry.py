import os
import json
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, base_dir: str):
        self.registry_dir = os.path.join(base_dir, "model_registry")
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_file = os.path.join(self.registry_dir, "registered_models.json")
        self._initialize_registry_file()

    def _initialize_registry_file(self):
        """Ensures the registered_models.json file exists and is a valid JSON array."""
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w") as f:
                json.dump([], f)
        else:
            # Validate if it's a valid JSON array
            try:
                with open(self.registry_file, "r") as f:
                    content = json.load(f)
                    if not isinstance(content, list):
                        raise ValueError("Registry file is not a JSON array.")
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Corrupted or invalid model registry file: {self.registry_file}. Reinitializing.")
                with open(self.registry_file, "w") as f:
                    json.dump([], f)

    def register_model(self, model_id: str, job_id: str, model_path: str, metrics: Dict[str, Any], hyperparameters: Dict[str, Any], version: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registers a new model with its metadata.
        Model_id should be unique.
        """
        # Check if model_id already exists
        if self.get_model_details(model_id)["status"] == "success":
            return {"status": "error", "message": f"Model ID '{model_id}' already exists in the registry. Use update_model if you want to modify."}

        model_entry = {
            "model_id": model_id,
            "job_id": job_id,
            "model_path": model_path,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "version": version if version else f"v{int(time.time())}",
            "registered_at": time.time(),
            "metadata": metadata if metadata else {},
            "status": "registered" # e.g., registered, staging, production, archived
        }

        try:
            with open(self.registry_file, "r+") as f:
                models = json.load(f)
                models.append(model_entry)
                f.seek(0)
                json.dump(models, f, indent=4)
                f.truncate()
            logger.info(f"Model '{model_id}' registered successfully.")
            return {"status": "success", "message": f"Model '{model_id}' registered.", "model_details": model_entry}
        except Exception as e:
            logger.error(f"Error registering model '{model_id}': {e}")
            return {"status": "error", "message": f"Failed to register model: {str(e)}"}

    def list_models(self) -> Dict[str, Any]:
        """Lists all registered models with their basic metadata."""
        try:
            with open(self.registry_file, "r") as f:
                models = json.load(f)
            return {"status": "success", "models": models}
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"status": "error", "message": f"Failed to list models: {str(e)}"}

    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """Retrieves detailed information for a specific registered model."""
        try:
            with open(self.registry_file, "r") as f:
                models = json.load(f)
            for model in models:
                if model.get("model_id") == model_id:
                    return {"status": "success", "model_details": model}
            return {"status": "error", "message": f"Model ID '{model_id}' not found in registry."}
        except Exception as e:
            logger.error(f"Error getting details for model '{model_id}': {e}")
            return {"status": "error", "message": f"Failed to get model details: {str(e)}"}

    def update_model_status(self, model_id: str, new_status: str) -> Dict[str, Any]:
        """Updates the status of a registered model."""
        try:
            with open(self.registry_file, "r+") as f:
                models = json.load(f)
                found = False
                for model in models:
                    if model.get("model_id") == model_id:
                        model["status"] = new_status
                        found = True
                        break
                if not found:
                    return {"status": "error", "message": f"Model ID '{model_id}' not found in registry."}
                
                f.seek(0)
                json.dump(models, f, indent=4)
                f.truncate()
            logger.info(f"Model '{model_id}' status updated to '{new_status}'.")
            return {"status": "success", "message": f"Model '{model_id}' status updated to '{new_status}'."}
        except Exception as e:
            logger.error(f"Error updating status for model '{model_id}': {e}")
            return {"status": "error", "message": f"Failed to update model status: {str(e)}"}

    def promote_model(self, model_id: str, new_status: str, new_version: Optional[str] = None, 
                      metadata_update: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Promotes a model to a new status (e.g., staging, production).
        Optionally creates a new version entry and updates metadata.
        """
        result = self.get_model_details(model_id)
        if result["status"] != "success":
            return result # Model not found

        current_model = result["model_details"]
        old_status = current_model["status"]

        # Update existing model's status
        update_status_result = self.update_model_status(model_id, new_status)
        if update_status_result["status"] != "success":
            return update_status_result

        # If a new version is provided, create a new entry for the promoted version
        if new_version:
            # Create a copy of the current model's details for the new version
            new_model_entry = current_model.copy()
            new_model_entry["model_id"] = f'{current_model["model_id"]}-{new_version}' # New model_id for the version
            new_model_entry["version"] = new_version
            new_model_entry["registered_at"] = time.time()
            new_model_entry["status"] = new_status
            if metadata_update:
                new_model_entry["metadata"].update(metadata_update)

            # Register the new version as a separate entry
            register_result = self.register_model(
                model_id=new_model_entry["model_id"],
                job_id=new_model_entry["job_id"],
                model_path=new_model_entry["model_path"],
                metrics=new_model_entry["metrics"],
                hyperparameters=new_model_entry["hyperparameters"],
                version=new_model_entry["version"],
                metadata=new_model_entry["metadata"]
            )
            if register_result["status"] != "success":
                # If registration of new version fails, revert the status of the original model
                self.update_model_status(model_id, old_status) 
                return {"status": "error", "message": f"Failed to register new version {new_version} for model {model_id}: {register_result['message']}"}
            
            logger.info(f"Model '{model_id}' promoted to '{new_status}' with new version '{new_version}'.")
            return {"status": "success", "message": f"Model '{model_id}' promoted to '{new_status}' with new version '{new_version}'.", "new_model_details": register_result["model_details"]}
        
        logger.info(f"Model '{model_id}' status updated from '{old_status}' to '{new_status}'.")
        return {"status": "success", "message": f"Model '{model_id}' status updated to '{new_status}'."}

    def rollback_model(self, current_model_id: str, target_model_id: str) -> Dict[str, Any]:
        """
        Rolls back the active model to a previous version specified by target_model_id.
        Sets current_model_id to 'archived' and target_model_id to 'production'.
        """
        current_model_result = self.get_model_details(current_model_id)
        if current_model_result["status"] != "success":
            return current_model_result # Current model not found

        target_model_result = self.get_model_details(target_model_id)
        if target_model_result["status"] != "success":
            return target_model_result # Target model not found

        # Mark current model as archived
        archive_result = self.update_model_status(current_model_id, "archived")
        if archive_result["status"] != "success":
            return archive_result

        # Promote target model to production
        promote_result = self.update_model_status(target_model_id, "production")
        if promote_result["status"] != "success":
            # If promotion fails, try to revert the archive status of the current model
            self.update_model_status(current_model_id, "production") # Assuming it was in production
            return promote_result

        logger.info(f"Model '{current_model_id}' rolled back to '{target_model_id}'.")
        return {"status": "success", "message": f"Model '{current_model_id}' rolled back to '{target_model_id}'."}
