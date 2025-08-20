import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from database import SessionLocal, Model, create_db_and_tables # Import from new database.py
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, base_dir: str):
        # Ensure database tables are created when the registry is initialized
        create_db_and_tables()
        logger.info("Database tables checked/created.")

    def register_model(self, model_id: str, job_id: str, model_path: str, metrics: Dict[str, Any], hyperparameters: Dict[str, Any], version: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            # Check if model_id already exists
            existing_model = db.query(Model).filter(Model.model_id == model_id).first()
            if existing_model:
                return {"status": "error", "message": f"Model ID '{model_id}' already exists in the registry. Use update_model if you want to modify."}

            model_entry = Model(
                model_id=model_id,
                job_id=job_id,
                model_path=model_path,
                metrics=json.dumps(metrics),
                hyperparameters=json.dumps(hyperparameters),
                version=version if version else f"v{int(time.time())}",
                registered_at=time.time(),
                metadata_json=json.dumps(metadata) if metadata else json.dumps({}),
                status="registered"
            )

            db.add(model_entry)
            db.commit()
            db.refresh(model_entry)
            logger.info(f"Model '{model_id}' registered successfully in DB.")
            return {"status": "success", "message": f"Model '{model_id}' registered.", "model_details": model_entry.__dict__}
        except IntegrityError:
            db.rollback()
            return {"status": "error", "message": f"Model ID '{model_id}' already exists. This is an integrity constraint violation."}
        except Exception as e:
            db.rollback()
            logger.error(f"Error registering model '{model_id}': {e}")
            return {"status": "error", "message": f"Failed to register model: {str(e)}"}
        finally:
            db.close()

    def list_models(self) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            models = db.query(Model).all()
            # Convert SQLAlchemy objects to dictionaries for consistent output
            model_list = []
            for model in models:
                model_dict = {
                    "model_id": model.model_id,
                    "job_id": model.job_id,
                    "model_path": model.model_path,
                    "metrics": json.loads(model.metrics) if model.metrics else {},
                    "hyperparameters": json.loads(model.hyperparameters) if model.hyperparameters else {},
                    "version": model.version,
                    "registered_at": model.registered_at,
                    "metadata": json.loads(model.metadata_json) if model.metadata_json else {},
                    "status": model.status
                }
                model_list.append(model_dict)
            return {"status": "success", "models": model_list}
        except Exception as e:
            logger.error(f"Error listing models from DB: {e}")
            return {"status": "error", "message": f"Failed to list models: {str(e)}"}
        finally:
            db.close()

    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            model = db.query(Model).filter(Model.model_id == model_id).first()
            if model:
                model_dict = {
                    "model_id": model.model_id,
                    "job_id": model.job_id,
                    "model_path": model.model_path,
                    "metrics": json.loads(model.metrics) if model.metrics else {},
                    "hyperparameters": json.loads(model.hyperparameters) if model.hyperparameters else {},
                    "version": model.version,
                    "registered_at": model.registered_at,
                    "metadata": json.loads(model.metadata_json) if model.metadata_json else {},
                    "status": model.status
                }
                return {"status": "success", "model_details": model_dict}
            return {"status": "error", "message": f"Model ID '{model_id}' not found in registry."}
        except Exception as e:
            logger.error(f"Error getting details for model '{model_id}' from DB: {e}")
            return {"status": "error", "message": f"Failed to get model details: {str(e)}"}
        finally:
            db.close()

    def update_model_status(self, model_id: str, new_status: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            model = db.query(Model).filter(Model.model_id == model_id).first()
            if not model:
                return {"status": "error", "message": f"Model ID '{model_id}' not found in registry."}
            
            model.status = new_status
            db.commit()
            logger.info(f"Model '{model_id}' status updated to '{new_status}' in DB.")
            return {"status": "success", "message": f"Model '{model_id}' status updated to '{new_status}'."}
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating status for model '{model_id}' in DB: {e}")
            return {"status": "error", "message": f"Failed to update model status: {str(e)}"}
        finally:
            db.close()

    def promote_model(self, model_id: str, new_status: str, new_version: Optional[str] = None, 
                      metadata_update: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            new_model_entry_data = current_model.copy() # Use a copy of the dictionary
            new_model_entry_data["model_id"] = f'{current_model["model_id"]}-{new_version}' # New model_id for the version
            new_model_entry_data["version"] = new_version
            new_model_entry_data["registered_at"] = time.time()
            new_model_entry_data["status"] = new_status
            if metadata_update:
                new_model_entry_data["metadata"].update(metadata_update)

            # Register the new version as a separate entry
            register_result = self.register_model(
                model_id=new_model_entry_data["model_id"],
                job_id=new_model_entry_data["job_id"],
                model_path=new_model_entry_data["model_path"],
                metrics=new_model_entry_data["metrics"],
                hyperparameters=new_model_entry_data["hyperparameters"],
                version=new_model_entry_data["version"],
                metadata=new_model_entry_data["metadata"]
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
