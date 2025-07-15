"""
Database configuration and models.
"""

from sqlalchemy import create_engine, Column, Integer, String, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import json
from data.mock_data import MOCK_OUTLETS

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

SQLALCHEMY_DATABASE_URL = "sqlite:///./data/zus.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Outlet(Base):
    __tablename__ = "outlets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    address = Column(String)
    opening_time = Column(String)
    closing_time = Column(String)
    services = Column(String)  # Store as JSON string

    def get_services(self):
        """Convert JSON string to list"""
        return json.loads(self.services) if self.services else []

    def set_services(self, services_list):
        """Convert list to JSON string"""
        self.services = json.dumps(services_list) if services_list else None

# Create tables and populate with mock data
def init_db():
    Base.metadata.create_all(bind=engine)
    
    # Add mock data
    db = SessionLocal()
    try:
        # Check if we already have data
        if db.query(Outlet).first() is None:
            for outlet_data in MOCK_OUTLETS:
                outlet = Outlet(
                    name=outlet_data["name"],
                    address=outlet_data["address"],
                    opening_time=outlet_data["opening_time"],
                    closing_time=outlet_data["closing_time"]
                )
                outlet.set_services(outlet_data.get("services", []))
                db.add(outlet)
            db.commit()
    finally:
        db.close()

# Initialize database with mock data
init_db()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 