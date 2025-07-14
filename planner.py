# mindhive-chatbot/planner.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import re
import httpx # <--- ADDED: Necessary for making async HTTP requests

class Intent(Enum):
    CALCULATION = "calculation"
    OUTLET_INFO = "outlet_info"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"

class Action(Enum):
    ASK_FOR_INFO = "ask_for_info"       # When more details are needed from the user
    USE_CALCULATOR = "use_calculator"   # When a mathematical calculation is requested
    USE_OUTLET_DB = "use_outlet_db"     # When information about a specific outlet is requested
    RESPOND_DIRECTLY = "respond_directly" # For general conversational replies using the LLM
    
@dataclass
class PlanningResult:
    intent: Intent
    action: Action
    missing_info: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

class AgenticPlanner:
    def __init__(self):
        # Patterns for identifying calculation-related intents
        self.calculation_patterns = [
            r'(\d+)\s*([\+\-\*\/])\s*(\d+)',
            r'what is (\d+)\s*([\+\-\*\/])\s*(\d+)',
            r'\d+\s*(plus|minus|times|multiply|divide|substract|divided by)\s*\d+',
            r'sum of|difference of|product of|quotient of',
            r'calculate|math',
            r'what\'s|whats\s+[\w\s]*\d+',
        ]
        
        self.outlet_patterns = [
            r'ss\s*\d+',
            r'outlet|store|shop|location|branch',
            r'opening|closing|hours|time',
            r'damansara|petaling jaya|kuala lumpur|pj|kl',
        ]
        
        self.operator_map = {
            'plus': '+', 'add': '+',
            'minus': '-', 'subtract': '-',
            'times': '*', 'multiply': '*',
            'divide': '/', 'divided by': '/'
        }
    
    def analyze_intent(self, user_input: str) -> Intent:
        user_input_lower = user_input.lower()
        
        for pattern in self.calculation_patterns:
            if re.search(pattern, user_input_lower):
                return Intent.CALCULATION
        
        for pattern in self.outlet_patterns:
            if re.search(pattern, user_input_lower):
                return Intent.OUTLET_INFO
                
        return Intent.GENERAL_CHAT
    
    def extract_calculation_data(self, user_input: str) -> Optional[Dict[str, Any]]:
        user_input_lower = user_input.lower()
        
        math_pattern = r'(\d+)\s*([\+\-\*\/])\s*(\d+)'
        match = re.search(math_pattern, user_input)
        if match:
            try:
                # IMPORTANT: Cast to float, as the FastAPI expects floats
                return {
                    'num1': float(match.group(1)),
                    'operator': match.group(2),
                    'num2': float(match.group(3))
                }
            except ValueError:
                pass
        
        nl_math_pattern = r'(\d+)\s*(plus|minus|times|multiply|divide|substract|divided by)\s*(\d+)'
        match_nl = re.search(nl_math_pattern, user_input_lower)
        if match_nl:
            op_word = match_nl.group(2)
            operator_symbol = self.operator_map.get(op_word)
            if operator_symbol:
                try:
                    return {
                        'num1': float(match_nl.group(1)),
                        'operator': operator_symbol,
                        'num2': float(match_nl.group(3))
                    }
                except ValueError:
                    pass
        
        what_is_pattern = r'what is (\d+)\s*([\+\-\*\/])\s*(\d+)'
        match_what_is = re.search(what_is_pattern, user_input_lower)
        if match_what_is:
            try:
                return {
                    'num1': float(match_what_is.group(1)),
                    'operator': match_what_is.group(2),
                    'num2': float(match_what_is.group(3))
                }
            except ValueError:
                pass
        
        return None
    
    def extract_outlet_data(self, user_input: str) -> Optional[Dict[str, Any]]:
        user_input_lower = user_input.lower()
        
        location = None
        if 'ss2' in user_input_lower or 'ss 2' in user_input_lower:
            location = 'SS2'
        elif 'ss15' in user_input_lower or 'ss 15' in user_input_lower:
            location = 'SS15'
        elif 'damansara' in user_input_lower:
            location = 'Damansara'
        elif 'petaling jaya' in user_input_lower or 'pj' in user_input_lower:
            location = 'Petaling Jaya'
        elif 'kuala lumpur' in user_input_lower or 'kl' in user_input_lower:
            location = 'Kuala Lumpur'
        
        info_type = None
        if 'opening' in user_input_lower or 'open' in user_input_lower:
            info_type = 'opening_hours'
        elif 'closing' in user_input_lower or 'close' in user_input_lower:
            info_type = 'closing_hours'
        elif 'hours' in user_input_lower or 'time' in user_input_lower:
            info_type = 'hours'

        if location or info_type:
            return {'location': location, 'info_type': info_type}
        return None
    
    def plan_next_action(self, user_input: str) -> PlanningResult:
        intent = self.analyze_intent(user_input)
        
        extracted_data = None
        missing_info = None
        action = Action.RESPOND_DIRECTLY
        confidence = 0.5
        
        if intent == Intent.CALCULATION:
            extracted_data = self.extract_calculation_data(user_input)
            if extracted_data:
                action = Action.USE_CALCULATOR
                confidence = 0.9
            else:
                action = Action.ASK_FOR_INFO
                missing_info = "I can help with calculations! What numbers and operation do you need? (e.g., '5 + 3' or '10 times 5')"
                confidence = 0.8
                
        elif intent == Intent.OUTLET_INFO:
            extracted_data = self.extract_outlet_data(user_input)
            
            if extracted_data and extracted_data.get('location') and \
               extracted_data['location'] not in ['Petaling Jaya', 'Kuala Lumpur']:
                action = Action.USE_OUTLET_DB
                confidence = 0.9
            
            elif extracted_data and (extracted_data.get('location') in ['Petaling Jaya', 'Kuala Lumpur'] or not extracted_data.get('location')) \
                and extracted_data.get('info_type'):
                action = Action.ASK_FOR_INFO
                missing_info = f"Yes, we have outlets in Petaling Jaya! Which specific outlet are you referring to (e.g., SS2, SS15, Damansara) to check the {extracted_data['info_type'].replace('_', ' ')}?"
                confidence = 0.85
            
            elif extracted_data and extracted_data.get('location') in ['Petaling Jaya', 'Kuala Lumpur'] and not extracted_data.get('info_type'):
                action = Action.ASK_FOR_INFO
                missing_info = f"Yes, we have outlets in {extracted_data['location']}! Which specific outlet are you referring to?"
                confidence = 0.85
            
            else:
                action = Action.ASK_FOR_INFO
                missing_info = "Which outlet are you asking about? Please specify a location (e.g., SS2, SS15, Damansara) or what kind of information you're looking for."
                confidence = 0.7
        
        return PlanningResult(
            intent=intent,
            action=action,
            missing_info=missing_info,
            extracted_data=extracted_data,
            confidence=confidence
        )


# --- NEW: Asynchronous function to call the Calculator API ---
async def call_calculator_api(num1: float, operator: str, num2: float) -> str:
    """
    Calls the external Calculator FastAPI to perform arithmetic operations.
    Handles successful responses and basic error cases.
    """
    # !!! IMPORTANT: Ensure your FastAPI calculator API is running at this URL/port !!!
    calculator_api_url = "http://localhost:8001/calculate" 

    payload = {
        "num1": num1,
        "operator": operator,
        "num2": num2
    }
    
    try:
        # Use httpx for async HTTP requests
        async with httpx.AsyncClient() as client:
            response = await client.post(calculator_api_url, json=payload, timeout=5.0)
            response.raise_for_status() # Raises an HTTPStatusError for bad responses (4xx or 5xx)

            data = response.json()
            if "result" in data:
                # Format float results cleanly (e.g., 5.0 -> 5, 2.5 -> 2.5)
                result = data["result"]
                if result == int(result): # If it's a whole number, display as int
                    return str(int(result))
                return str(result)
            else:
                return "Error: Calculator API did not return a valid result."
    except httpx.HTTPStatusError as e:
        # Handle specific HTTP errors (e.g., 400 for division by zero, 500 for server errors)
        if e.response.status_code == 400:
            error_detail = e.response.json().get("detail", "Invalid input for calculation.")
            return f"Calculation Error: {error_detail}"
        else:
            return f"Calculator API Error: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        # Handle network or connection errors (e.g., API not running, or connection refused)
        return f"Could not connect to the calculator service. Please try again later. (Error: {e})"
    except Exception as e:
        # Catch any other unexpected errors
        return f"An unexpected error occurred while calling the calculator: {str(e)}"

def get_mock_outlet_info(location: Optional[str], info_type: Optional[str]) -> str:
    """
    Mocks retrieving specific information about coffee shop outlets.
    This is a mock implementation for Part 2. It will be replaced by
    actual Text2SQL API calls and RAG in Part 4.
    """
    location_map = {
        'SS2': {'opening_hours': '9:00 AM', 'closing_hours': '10:00 PM', 'general_info': 'a bustling spot in Petaling Jaya with good vibes.'},
        'SS15': {'opening_hours': '8:00 AM', 'closing_hours': '9:00 PM', 'general_info': 'a lively student hangout spot.'},
        'Damansara': {'opening_hours': '7:00 AM', 'closing_hours': '11:00 PM', 'general_info': 'a cozy spot for early birds in Damansara.'},
        'Petaling Jaya': {'general_info': 'several great outlets like SS2, SS15, and Damansara.'},
        'Kuala Lumpur': {'general_info': 'several great outlets like our flagship KLCC branch (details not available yet!).'}
    }

    if not location:
        return "I need a specific outlet (like SS2, SS15, or Damansara) to give you information."

    outlet_data = location_map.get(location)

    if not outlet_data:
        return f"I don't have detailed information for an outlet specifically called '{location}'. Did you mean SS2, SS15, or Damansara?"
    
    if location in ['Petaling Jaya', 'Kuala Lumpur']:
        if info_type:
            return f"We have several outlets in {location}. Which specific one (e.g., SS2, SS15, Damansara) are you interested in for its {info_type.replace('_', ' ')}?"
        return f"Yes, we have outlets in {location}, including {outlet_data['general_info']}. Which specific outlet would you like to know about?"

    if info_type == 'opening_hours':
        return f"The {location} outlet opens at {outlet_data['opening_hours']}."
    elif info_type == 'closing_hours':
        return f"The {location} outlet closes at {outlet_data['closing_hours']}."
    elif info_type == 'hours':
        return f"The {location} outlet opens at {outlet_data['opening_hours']} and closes at {outlet_data['closing_hours']}."
    else:
        return f"The {location} outlet is {outlet_data['general_info']} Would you like to know its opening or closing hours?"