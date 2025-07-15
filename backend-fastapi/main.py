"""
FastAPI application with all endpoints.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from sqlalchemy.orm import Session
from sqlalchemy import text
import json

from utils.database import get_db, Outlet
from utils.vector_store import vector_store
from utils.text2sql import sql_generator

app = FastAPI(
    title="ZUS Coffee API",
    description="API for calculator, product search, and outlet queries"
)

# Calculator endpoint
class CalculationRequest(BaseModel):
    num1: float = Field(..., description="First number")
    operator: Literal['+', '-', '*', '/'] = Field(..., description="Arithmetic operator")
    num2: float = Field(..., description="Second number")

@app.post("/calculate")
async def calculate(request: CalculationRequest):
    """
    Performs a simple arithmetic calculation based on the provided numbers and operator.
    Handles division by zero error.
    """
    try:
        result = 0.0
        if request.operator == '+':
            result = request.num1 + request.num2
        elif request.operator == '-':
            result = request.num1 - request.num2
        elif request.operator == '*':
            result = request.num1 * request.num2
        elif request.operator == '/':
            if request.num2 == 0:
                raise HTTPException(status_code=400, detail="Division by zero is not allowed.")
            result = request.num1 / request.num2
        
        return {"result": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Products endpoint
class ProductQuery(BaseModel):
    query: str = Field(..., description="Search query for products")
    top_k: Optional[int] = Field(default=3, description="Number of results to return")

class ProductResponse(BaseModel):
    name: str
    description: str
    price: float
    colors: List[str]
    category: str

@app.post("/products")
async def search_products(query: ProductQuery):
    """
    Search for products using vector similarity search and generate AI summary
    """
    try:
        result = vector_store.search(query.query, k=query.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Outlets endpoint
class OutletQuery(BaseModel):
    query: str = Field(..., description="Natural language query for outlets")

class OutletResponse(BaseModel):
    name: str
    address: str
    opening_time: str
    closing_time: str
    services: List[str]

@app.post("/outlets")
async def query_outlets(query: OutletQuery, db: Session = Depends(get_db)):
    """
    Query outlets using natural language to SQL conversion
    """
    try:
        # Generate SQL from natural language query
        sql_query = sql_generator.generate_sql(query.query)
        
        # Execute the generated SQL
        results = db.execute(text(sql_query))
        outlets = results.all()
        
        if outlets:
            return {
                "results": [
                    {
                        "name": outlet.name,
                        "address": outlet.address,
                        "opening_time": outlet.opening_time,
                        "closing_time": outlet.closing_time,
                        "services": outlet.get_services() if hasattr(outlet, 'get_services') else json.loads(outlet.services) if outlet.services else []
                    }
                    for outlet in outlets
                ],
                "sql_query": sql_query
            }
        else:
            return {
                "results": [],
                "message": "No outlets found matching your query.",
                "sql_query": sql_query
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint - health check
    """
    return {
        "status": "ok",
        "message": "ZUS Coffee API is running"
    }