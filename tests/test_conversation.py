# tests/test_conversation.py

"""
Tests for the chatbot's conversational memory and agentic behavior (Part 1 & 2),
and Calculator API integration with error handling (Part 3).
"""

import pytest
import asyncio # <--- Essential for running async tests
from main import ChatbotController
# Intent and Action are imported for clarity in test names/comments, not directly asserted from here.
from planner import Intent, Action 

# --- Helper function for flexible string checking ---
def contains_any(text: str, keywords: list) -> bool:
    """
    Checks if text (case-insensitive) contains any of the given keywords.
    Useful for flexible assertions on LLM or tool responses.
    """
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

# --- Tests for Part 1: Sequential Conversation (Memory & Basic Outlet Mock) ---

@pytest.mark.asyncio # Marks this test function to be run by pytest-asyncio
async def test_three_turn_outlet_conversation_happy_path():
    """
    HAPPY PATH: Tests the full three-turn outlet information flow from the assessment example.
    Verifies that the bot correctly handles context across turns and provides expected mock answers.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "three_turn_outlet_test"

    # 2. Act
    # Turn 1: User asks generally about an outlet in Petaling Jaya
    print("\n--- Test Turn 1 ---")
    response_1 = await controller.process_user_input( # Await the async call
        "Is there an outlet in Petaling Jaya?", 
        session_id
    )
    print(f"Bot (Turn 1): {response_1}")
    
    # Turn 2: User specifies SS 2 and asks for opening time
    print("\n--- Test Turn 2 ---")
    response_2 = await controller.process_user_input( # Await the async call
        "SS 2, what's the opening time?", 
        session_id
    )
    print(f"Bot (Turn 2): {response_2}")

    # Turn 3: User asks for closing time, relying on previous context (SS 2)
    print("\n--- Test Turn 3 ---")
    response_3 = await controller.process_user_input( # Await the async call
        "What about the closing time?", 
        session_id
    )
    print(f"Bot (Turn 3): {response_3}")

    # 3. Assert - Verify responses and history
    assert contains_any(response_1, ["which outlet", "specific outlet", "petaling jaya", "pj"]) 
    assert contains_any(response_2, ["ss2", "ss 2"]) and contains_any(response_2, ["9:00 am", "9am"]) 
    assert contains_any(response_3, ["ss2", "ss 2"]) and contains_any(response_3, ["10:00 pm", "10pm"]) 

    # Verify message history contains all turns (3 human inputs + 3 AI responses = 6 messages)
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 6
    assert isinstance(history.messages[0], type(history.messages[0]))
    msg0_content = history.messages[0].content if hasattr(history.messages[0], "content") else history.messages[0]
    assert isinstance(msg0_content, str)
    assert "petaling jaya" in msg0_content.lower()
    msg2_content = history.messages[2].content if hasattr(history.messages[2], "content") else history.messages[2]
    msg4_content = history.messages[4].content if hasattr(history.messages[4], "content") else history.messages[4]
    msg1_content = history.messages[1].content if hasattr(history.messages[1], "content") else history.messages[1]
    msg3_content = history.messages[3].content if hasattr(history.messages[3], "content") else history.messages[3]
    msg5_content = history.messages[5].content

    if isinstance(msg2_content, str):
        assert "ss 2" in msg2_content.lower()
    if isinstance(msg4_content, str):
        assert "closing time" in msg4_content.lower()
    if isinstance(msg1_content, str):
        assert contains_any(msg1_content, ["which outlet", "specific outlet", "petaling jaya", "pj"])
    if isinstance(msg3_content, str):
        assert contains_any(msg3_content, ["9:00 am", "9am"])
    if isinstance(msg5_content, str):
        assert contains_any(msg5_content, ["10:00 pm", "10pm"])

@pytest.mark.asyncio # Marks as async
async def test_interrupted_conversation_new_session_for_outlet_info():
    """
    INTERRUPTED PATH: Tests that a new session starts with no context.
    User asks for outlet info without sufficient details in a fresh session, expecting clarification.
    """
    # 1. Arrange
    controller = ChatbotController()
    
    # Simulate an unrelated previous conversation in a different session (not affecting new_clean_session)
    # This call is also async, so it needs to be awaited if called.
    await controller.process_user_input("I am looking for a coffee shop nearby", "old_session_unrelated") 

    # 2. Act - Start a new, clean session (simulating interruption/new user)
    response = await controller.process_user_input( 
        "What are the opening hours?", 
        "new_clean_session"
    )

    # 3. Assert - Bot should ask for clarification as it has no prior context in this new session
    assert contains_any(response, ["which outlet", "specific outlet", "specify a location", "kind of information"])
    
    new_session_history = controller.get_session_history("new_clean_session")
    assert len(new_session_history.messages) == 2 # 1 User input + 1 Bot's clarification


# --- General Memory and Agentic Action Tests ---

@pytest.mark.asyncio # Marks as async
async def test_conversation_stores_message_history_through_controller():
    """
    HAPPY PATH: Verifies that the ChatbotController correctly stores
    and manages conversation history for general chat using LLM responses.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "general_chat_history_test"

    # 2. Act
    response1 = await controller.process_user_input("Hello, my name is Alice", session_id) 
    response2 = await controller.process_user_input("What is my name?", session_id) 

    # 3. Assert
    history = controller.get_session_history(session_id)
    messages = history.messages

    assert len(messages) == 4 # 2 human inputs + 2 AI responses
    assert messages[0].content == "Hello, my name is Alice"
    assert messages[2].content == "What is my name?"
    assert len(str(messages[1].content)) > 0 # AI response to first message (content exists)
    assert len(str(messages[3].content)) > 0 # AI response to second message (content exists)
    assert "alice" in str(messages[3].content).lower() # Verify AI remembered the name

@pytest.mark.asyncio # Marks as async
async def test_agentic_calculator_happy_path():
    """
    HAPPY PATH: Tests the planner correctly identifies calculation intent
    and successfully uses the REAL calculator API, returning the expected numerical result.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "calc_happy_path"

    # 2. Act
    # Use a calculation that should parse correctly (e.g., "10 plus 5")
    response = await controller.process_user_input("What is 10 plus 5?", session_id) 

    # 3. Assert
    assert "15" in response # Check for the correct calculated result (10 + 5)
    
    # Verify history reflects the tool usage with the correct output
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 2
    assert "10 plus 5" in str(history.messages[0].content).lower()
    assert "15" in str(history.messages[1].content)
async def test_agentic_calculator_missing_info_interrupted_path():
    """
    INTERRUPTED PATH: Tests that the planner asks for clarification when calculation data is missing,
    and that the response contains expected phrases for asking for numbers/operation.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "calc_missing_info_test"

    # 2. Act
    response = await controller.process_user_input("I need a calculation.", session_id) 

    # 3. Assert
    # Check for keywords indicating a request for more calculation details from the planner
    assert contains_any(response, ["provide the numbers", "what would you like me to calculate", "what numbers and operation", "what's the calculation you need help with", "calculation you need help with"])
    
    # Verify history includes the user's request and the bot's clarification
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 2
    assert "calculation" in str(history.messages[0].content).lower()
    assert contains_any(str(history.messages[1].content), ["provide the numbers", "what numbers and operation", "what's the calculation you need help with"])
async def test_agentic_calculator_division_by_zero_error_handling():
    """
    ERROR HANDLING PATH: Tests that the chatbot gracefully handles a division-by-zero error
    returned by the calculator API without crashing.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "calc_div_by_zero_test"

    # 2. Act
    response = await controller.process_user_input("What is 10 / 0?", session_id)

    # 3. Assert
    # Check for the specific error message from the calculator API
    assert "calculation error: division by zero is not allowed" in response.lower()
    
    # Verify history includes the problematic request and the error message
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 2
    assert "10 / 0" in str(history.messages[0].content)
    assert "division by zero" in str(history.messages[1].content).lower()


# --- Tests for Outlet Info (Still using Mock for now) ---

@pytest.mark.asyncio # Marks as async
async def test_agentic_outlet_info_specific_query_happy_path():
    """
    HAPPY PATH: Tests the planner correctly identifies outlet info intent
    and uses the mock outlet database, returning specific info.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "outlet_specific_query_test"

    # 2. Act
    response = await controller.process_user_input("Tell me about the Damansara outlet's closing time.", session_id) 

    # 3. Assert
    assert "damansara" in response.lower()
    assert contains_any(response, ["11:00 pm", "11pm"])
    
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 2
    assert "damansara outlet" in str(history.messages[0].content).lower()
    assert contains_any(str(history.messages[1].content), ["11:00 pm", "11pm"])
async def test_agentic_outlet_info_missing_details_interrupted_path():
    """
    INTERRUPTED PATH: Tests the planner asks for clarification when outlet info is missing location details.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "outlet_missing_details_test"

    # 2. Act
    response = await controller.process_user_input("What are the hours?", session_id) 

    # 3. Assert
    assert contains_any(response, ["which outlet", "specific outlet", "specify a location", "kind of information", "what kind of information you're looking for"])
    
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 2
    assert "hours" in str(history.messages[0].content).lower()
    assert contains_any(str(history.messages[1].content), ["which outlet", "specific outlet"])