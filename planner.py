# planner.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import re

class Intent(Enum):
    """
    Defines the different types of user intentions our chatbot can recognize.
    These intents guide the planner's decision-making process.
    """
    CALCULATION = "calculation"
    OUTLET_INFO = "outlet_info"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"

class Action(Enum):
    """
    Defines the specific actions the chatbot can take based on the determined intent.
    These actions drive the execution flow of the chatbot controller.
    """
    ASK_FOR_INFO = "ask_for_info"       # When more details are needed from the user
    USE_CALCULATOR = "use_calculator"   # When a mathematical calculation is requested
    USE_OUTLET_DB = "use_outlet_db"     # When information about a specific outlet is requested
    RESPOND_DIRECTLY = "respond_directly" # For general conversational replies using the LLM
    
@dataclass
class PlanningResult:
    """
    A data structure to hold the outcome of the planning process.
    It encapsulates the bot's decision: what the user wants, what to do,
    any missing information, extracted data for tools, and confidence.
    """
    intent: Intent
    action: Action
    missing_info: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

class AgenticPlanner:
    """
    The core decision-making component of the chatbot.
    It analyzes user input, determines the user's intent, extracts relevant data,
    and then decides the most appropriate action for the chatbot to take.
    """
    
    def __init__(self):
        """
        Initializes the planner with regular expression patterns for intent detection.
        These patterns help classify user inputs into predefined intents.
        """
        # Patterns for identifying calculation-related intents
        # Ordered by specificity/priority
        self.calculation_patterns = [
            r'(\d+)\s*([\+\-\*\/])\s*(\d+)',  # e.g., "5 + 3", "10 / 2" (most specific)
            r'what is (\d+)\s*([\+\-\*\/])\s*(\d+)', # "what is 5+3"
            r'\d+\s*(plus|minus|times|multiply|divide|substract|divided by)\s*\d+', # "10 plus 5"
            r'sum of|difference of|product of|quotient of', # natural language operations
            r'calculate|math', # broad keywords
            r'what\'s|whats\s+[\w\s]*\d+', # e.g., "what's 10 * 2", "what's the sum of 5 and 3"
        ]
        
        # Patterns for identifying outlet information-related intents
        # Ordered by specificity/priority
        self.outlet_patterns = [
            r'ss\s*\d+', # specific outlets like "ss2"
            r'outlet|store|shop|location|branch', # Keywords for places
            r'opening|closing|hours|time', # Keywords for time-related info
            r'damansara|petaling jaya|kuala lumpur|pj|kl', # Specific location keywords
        ]
        
        # Mapping natural language operators to symbols for calculation
        self.operator_map = {
            'plus': '+', 'add': '+',
            'minus': '-', 'subtract': '-',
            'times': '*', 'multiply': '*',
            'divide': '/', 'divided by': '/'
        }
    
    def analyze_intent(self, user_input: str) -> Intent:
        """
        Analyzes the user's input to determine their primary intent.
        Order of checks matters: more specific/tool-driven intents first.
        """
        user_input_lower = user_input.lower()
        
        # Check for calculation intent first, as it's often very specific
        for pattern in self.calculation_patterns:
            if re.search(pattern, user_input_lower):
                return Intent.CALCULATION
        
        # Check for outlet info intent
        for pattern in self.outlet_patterns:
            if re.search(pattern, user_input_lower):
                return Intent.OUTLET_INFO
                
        # Default to general chat if no specific intent is detected
        return Intent.GENERAL_CHAT
    
    def extract_calculation_data(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Extracts numbers and the operation from a user's calculation request.
        Supports basic arithmetic operations.
        """
        user_input_lower = user_input.lower()
        
        # Try direct symbol match (e.g., "5 + 3")
        math_pattern = r'(\d+)\s*([\+\-\*\/])\s*(\d+)'
        match = re.search(math_pattern, user_input)
        if match:
            try:
                return {
                    'num1': int(match.group(1)),
                    'operator': match.group(2),
                    'num2': int(match.group(3))
                }
            except ValueError:
                pass # Continue to next extraction method if conversion fails
        
        # Try natural language operator match (e.g., "10 plus 5")
        nl_math_pattern = r'(\d+)\s*(plus|minus|times|multiply|divide|substract|divided by)\s*(\d+)'
        match_nl = re.search(nl_math_pattern, user_input_lower)
        if match_nl:
            op_word = match_nl.group(2)
            operator_symbol = self.operator_map.get(op_word)
            if operator_symbol:
                try:
                    return {
                        'num1': int(match_nl.group(1)),
                        'operator': operator_symbol,
                        'num2': int(match_nl.group(3))
                    }
                except ValueError:
                    pass
        
        # Try "what is X op Y" pattern
        what_is_pattern = r'what is (\d+)\s*([\+\-\*\/])\s*(\d+)'
        match_what_is = re.search(what_is_pattern, user_input_lower)
        if match_what_is:
            try:
                return {
                    'num1': int(match_what_is.group(1)),
                    'operator': match_what_is.group(2),
                    'num2': int(match_what_is.group(3))
                }
            except ValueError:
                pass
        
        return None
    
    def extract_outlet_data(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Extracts outlet location and the type of information requested (e.g., opening hours).
        """
        user_input_lower = user_input.lower()
        
        # Extract location based on keywords
        location = None
        if 'ss2' in user_input_lower or 'ss 2' in user_input_lower:
            location = 'SS2'
        elif 'ss15' in user_input_lower or 'ss 15' in user_input_lower:
            location = 'SS15'
        elif 'damansara' in user_input_lower:
            location = 'Damansara'
        elif 'petaling jaya' in user_input_lower or 'pj' in user_input_lower:
            # If "Petaling Jaya" is mentioned generally, we still set the location
            # but the planner will likely ask for a specific outlet within PJ.
            location = 'Petaling Jaya'
        elif 'kuala lumpur' in user_input_lower or 'kl' in user_input_lower:
            location = 'Kuala Lumpur'
        
        # Extract type of information requested (e.g., opening, closing, general hours)
        info_type = None
        if 'opening' in user_input_lower or 'open' in user_input_lower:
            info_type = 'opening_hours'
        elif 'closing' in user_input_lower or 'close' in user_input_lower:
            info_type = 'closing_hours'
        elif 'hours' in user_input_lower or 'time' in user_input_lower:
            info_type = 'hours' # General hours query

        # Return extracted data if any relevant information was found
        if location or info_type:
            return {'location': location, 'info_type': info_type}
        return None
    
    def plan_next_action(self, user_input: str) -> PlanningResult:
        """
        The main planning method. This function serves as the chatbot's decision-making brain,
        determining the best course of action based on the user's input and extracted data.
        """
        # Step 1: Analyze user's intent
        intent = self.analyze_intent(user_input)
        
        # Initialize variables for planning result
        extracted_data = None
        missing_info = None
        action = Action.RESPOND_DIRECTLY # Default action
        confidence = 0.5 # Default confidence

        # Step 2 & 3: Decide action based on intent and data completeness
        if intent == Intent.CALCULATION:
            extracted_data = self.extract_calculation_data(user_input)
            if extracted_data:
                action = Action.USE_CALCULATOR
                confidence = 0.9
            else:
                # If intent is calculation but data is missing, ask for more info
                action = Action.ASK_FOR_INFO
                # Refined prompt for clarity
                missing_info = "I can help with calculations! What numbers and operation do you need? (e.g., '5 + 3' or '10 times 5')"
                confidence = 0.8
                
        elif intent == Intent.OUTLET_INFO:
            extracted_data = self.extract_outlet_data(user_input)
            
            # Scenario 1: Specific outlet found (e.g., SS2, SS15, Damansara)
            if extracted_data and extracted_data.get('location') and \
               extracted_data['location'] not in ['Petaling Jaya', 'Kuala Lumpur']:
                action = Action.USE_OUTLET_DB
                confidence = 0.9
            
            # Scenario 2: General location (Petaling Jaya, Kuala Lumpur) or only info_type found
            elif extracted_data and (extracted_data.get('location') in ['Petaling Jaya', 'Kuala Lumpur'] or not extracted_data.get('location')) \
                and extracted_data.get('info_type'):
                # Bot needs a specific outlet, but knows it's about hours
                action = Action.ASK_FOR_INFO
                # Refined prompt for clarity
                missing_info = f"Yes, we have outlets in Petaling Jaya! Which specific outlet are you referring to (e.g., SS2, SS15, Damansara) to check the {extracted_data['info_type'].replace('_', ' ')}?"
                confidence = 0.85
            
            # Scenario 3: Only a general query for outlets (e.g., "Is there an outlet in PJ?")
            elif extracted_data and extracted_data.get('location') in ['Petaling Jaya', 'Kuala Lumpur'] and not extracted_data.get('info_type'):
                action = Action.ASK_FOR_INFO
                missing_info = f"Yes, we have outlets in {extracted_data['location']}! Which specific outlet are you referring to?"
                confidence = 0.85
            
            # Scenario 4: Intent detected, but no location or info type
            else:
                action = Action.ASK_FOR_INFO
                # Refined prompt for clarity
                missing_info = "Which outlet are you asking about? Please specify a location (e.g., SS2, SS15, Damansara) or what kind of information you're looking for."
                confidence = 0.7
        
        return PlanningResult(
            intent=intent,
            action=action,
            missing_info=missing_info,
            extracted_data=extracted_data,
            confidence=confidence
        )


# --- Helper Functions for Tool Execution (Mock Implementations for Part 2) ---

def perform_simple_calculation(num1: int, operator: str, num2: int) -> str:
    """
    Performs a basic arithmetic calculation based on provided numbers and operator.
    This is a mock implementation for Part 2. It will be replaced by a proper API call in Part 3.
    """
    try:
        if operator == '+':
            return str(num1 + num2)
        elif operator == '-':
            return str(num1 - num2)
        elif operator == '*':
            return str(num1 * num2)
        elif operator == '/':
            if num2 == 0:
                return "Error: Division by zero is not allowed."
            return str(num1 / num2)
        else:
            return "Error: Invalid operator for calculation."
    except Exception as e:
        return f"An unexpected error occurred during calculation: {e}"

def get_mock_outlet_info(location: Optional[str], info_type: Optional[str]) -> str:
    """
    Mocks retrieving specific information about coffee shop outlets.
    This is a mock implementation for Part 2. It will be replaced by
    actual Text2SQL API calls and RAG in Part 4.
    """
    # Simple dictionary to simulate a database of outlet information
    location_map = {
        'SS2': {'opening_hours': '9:00 AM', 'closing_hours': '10:00 PM', 'general_info': 'a bustling spot in Petaling Jaya with good vibes.'},
        'SS15': {'opening_hours': '8:00 AM', 'closing_hours': '9:00 PM', 'general_info': 'a lively student hangout spot.'},
        'Damansara': {'opening_hours': '7:00 AM', 'closing_hours': '11:00 PM', 'general_info': 'a cozy spot for early birds in Damansara.'},
        'Petaling Jaya': {'general_info': 'several great outlets like SS2, SS15, and Damansara.'}, # Added general PJ info
        'Kuala Lumpur': {'general_info': 'several great outlets like our flagship KLCC branch (details not available yet!).'} # Added general KL info
    }

    if not location:
        return "I need a specific outlet (like SS2, SS15, or Damansara) to give you information."

    outlet_data = location_map.get(location)

    if not outlet_data:
        return f"I don't have detailed information for an outlet specifically called '{location}'. Did you mean SS2, SS15, or Damansara?"
    
    # Handle general location queries
    if location in ['Petaling Jaya', 'Kuala Lumpur']:
        if info_type: # If they asked for specific info for a general location
            return f"We have several outlets in {location}. Which specific one (e.g., SS2, SS15, Damansara) are you interested in for its {info_type.replace('_', ' ')}?"
        return f"Yes, we have outlets in {location}, including {outlet_data['general_info']}. Which specific outlet would you like to know about?"

    # Provide information based on the requested type for specific outlets
    if info_type == 'opening_hours':
        return f"The {location} outlet opens at {outlet_data['opening_hours']}."
    elif info_type == 'closing_hours':
        return f"The {location} outlet closes at {outlet_data['closing_hours']}."
    elif info_type == 'hours':
        return f"The {location} outlet opens at {outlet_data['opening_hours']} and closes at {outlet_data['closing_hours']}."
    else:
        # Default response if info_type is general or not specified after specific location is found
        return f"The {location} outlet is {outlet_data['general_info']} Would you like to know its opening or closing hours?"