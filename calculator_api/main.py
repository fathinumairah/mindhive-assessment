# calculator_api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

# Initialize the FastAPI application
app = FastAPI(
    title="Simple Calculator API",
    description="A basic API for performing arithmetic operations."
)

# Define the request body model for our calculation endpoint
# This specifies what data the API expects for a calculation
class CalculationRequest(BaseModel):
    num1: float # The first number (can be integer or decimal)
    operator: Literal['+', '-', '*', '/'] # The arithmetic operator (must be one of these)
    num2: float # The second number

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
                # If division by zero, raise a specific HTTP error (400 Bad Request)
                raise HTTPException(status_code=400, detail="Division by zero is not allowed.")
            result = request.num1 / request.num2
        
        # Return the result as a JSON object
        return {"result": result}
    except HTTPException as e:
        # Re-raise HTTPException if it's already an HTTP error
        raise e
    except Exception as e:
        # Catch any other unexpected errors and return a 500 Internal Server Error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")