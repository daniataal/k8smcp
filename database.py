from sqlalchemy import create_engine, Column, String, Float, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import json
import time

# Database connection string
# Get from environment variable or use a default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/k8smcp_db")

Engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
Base = declarative_base()

class Model(Base):
    __tablename__ = "models"

    model_id = Column(String, primary_key=True, index=True)
    job_id = Column(String, index=True)
    model_path = Column(String)
    metrics = Column(Text) # Stored as JSON string
    hyperparameters = Column(Text) # Stored as JSON string
    version = Column(String)
    registered_at = Column(Float)
    metadata_json = Column(Text) # Stored as JSON string to avoid conflict with 'metadata'
    status = Column(String)

    def __repr__(self):
        return f"<Model(model_id='{self.model_id}', status='{self.status}')>"

# Function to create all tables
def create_db_and_tables():
    Base.metadata.create_all(bind=Engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Example usage (for testing, will remove later)
if __name__ == "__main__":
    print("Creating database tables...")
    create_db_and_tables()
    print("Tables created (if they didn't exist).")

    # Example of adding a model
    db = SessionLocal()
    try:
        new_model = Model(
            model_id="test-model-123",
            job_id="job-abc",
            model_path="/path/to/model.pt",
            metrics=json.dumps({"accuracy": 0.95}),
            hyperparameters=json.dumps({"lr": 0.01}),
            version="v1.0",
            registered_at=time.time(),
            metadata_json=json.dumps({"author": "test"}),
            status="registered"
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        print(f"Added model: {new_model}")

        # Example of querying a model
        retrieved_model = db.query(Model).filter(Model.model_id == "test-model-123").first()
        if retrieved_model:
            print(f"Retrieved model: {retrieved_model.model_id}, Status: {retrieved_model.status}")

    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()
