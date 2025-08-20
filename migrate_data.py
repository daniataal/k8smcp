import json
import os
import time
import logging
from database import SessionLocal, Model, create_db_and_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def migrate_json_to_postgres(json_file_path: str):
    if not os.path.exists(json_file_path):
        logger.info(f"No JSON registry file found at {json_file_path}. Skipping migration.")
        return {"status": "skipped", "message": "No JSON file to migrate."}

    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {json_file_path}: {e}")
        return {"status": "error", "message": f"Failed to decode JSON: {str(e)}"}

    if not isinstance(data, list):
        logger.error(f"JSON file {json_file_path} does not contain a list. Skipping migration.")
        return {"status": "error", "message": "JSON file is not a list."}

    db = SessionLocal()
    migrated_count = 0
    skipped_count = 0

    try:
        for entry in data:
            model_id = entry.get("model_id")
            if not model_id:
                logger.warning(f"Skipping entry with no model_id: {entry}")
                skipped_count += 1
                continue

            # Check if model already exists in DB
            existing_model = db.query(Model).filter(Model.model_id == model_id).first()
            if existing_model:
                logger.info(f"Model '{model_id}' already exists in the database. Skipping.")
                skipped_count += 1
                continue

            try:
                new_model = Model(
                    model_id=model_id,
                    job_id=entry.get("job_id"),
                    model_path=entry.get("model_path"),
                    metrics=json.dumps(entry.get("metrics", {})),
                    hyperparameters=json.dumps(entry.get("hyperparameters", {})),
                    version=entry.get("version", f"v{int(time.time())}"),
                    registered_at=entry.get("registered_at", time.time()),
                    metadata_json=json.dumps(entry.get("metadata", {})),
                    status=entry.get("status", "registered")
                )
                db.add(new_model)
                migrated_count += 1
            except Exception as e:
                logger.error(f"Error processing model {model_id}: {e}")
                skipped_count += 1
                # Continue to next entry even if one fails
        
        db.commit()
        logger.info(f"Migration complete. Migrated {migrated_count} models, skipped {skipped_count} models.")
        return {"status": "success", "message": f"Migration complete. Migrated {migrated_count} models, skipped {skipped_count} models."}

    except Exception as e:
        db.rollback()
        logger.error(f"An error occurred during migration: {e}")
        return {"status": "error", "message": f"Migration failed: {str(e)}"}
    finally:
        db.close()

if __name__ == "__main__":
    # Ensure database tables exist before migration
    create_db_and_tables()

    # Path to your old JSON registry file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_registry_file = os.path.join(current_dir, "model_registry", "registered_models.json")
    
    print(f"Starting data migration from {json_registry_file}...")
    migration_result = migrate_json_to_postgres(json_registry_file)
    print(f"Migration result: {migration_result}")

    if migration_result["status"] == "success":
        print("You can now consider deleting the old JSON file if migration was successful.")
        # Example of how to remove the old file after successful migration
        # os.remove(json_registry_file)
