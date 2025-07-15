"""
Tests for the chatbot's conversational memory and agentic behavior (Part 1 & 2).
"""

import pytest
from main import ChatbotController
from planner import Intent, Action # These imports are good for context but not directly used in assertions here

# --- Helper function for flexible string checking ---
def contains_any(text: str, keywords: list) -> bool:
    """Checks if text contains any of the given keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

# --- Tests for Part 1: Sequential Conversation with Outlet Info ---

def test_three_turn_outlet_conversation_happy_path():
    """
    HAPPY PATH: Tests the full three-turn outlet information flow from the assessment.
    Verifies that the bot correctly handles context and provides expected mock answers.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "three_turn_outlet_test"

    # 2. Act
    # Turn 1: User asks generally about an outlet in Petaling Jaya
    print("\n--- Test Turn 1 ---")
    response_1 = controller.process_user_input(
        "Is there an outlet in Petaling Jaya?", 
        session_id
    )
    print(f"Bot (Turn 1): {response_1}")
    
    # Turn 2: User specifies SS 2 and asks for opening time
    print("\n--- Test Turn 2 ---")
    response_2 = controller.process_user_input(
        "SS 2, what's the opening time?", 
        session_id
    )
    print(f"Bot (Turn 2): {response_2}")

    # Turn 3: User asks for closing time, relying on previous context (SS 2)
    print("\n--- Test Turn 3 ---")
    response_3 = controller.process_user_input(
        "What about the closing time?", 
        session_id
    )
    print(f"Bot (Turn 3): {response_3}")

    # 3. Assert
    # Await coroutine responses if necessary (for async support)
    if hasattr(response_1, "__await__"):
        import asyncio
        response_1 = asyncio.run(response_1)
    if hasattr(response_2, "__await__"):
        import asyncio
        response_2 = asyncio.run(response_2)

    # Assertions for Turn 1 response: Bot should ask for specific outlet and acknowledge PJ
    assert contains_any(str(response_1), ["which outlet", "specific outlet", "petaling jaya", "pj"]) # Combines checks

    # Assertions for Turn 2 response: Should provide SS2 opening time
    assert contains_any(str(response_2), ["ss2", "ss 2"])
    assert contains_any(str(response_2), ["9:00 am", "9am"]) # Check for mock opening time

    # Await coroutine response if necessary (for async support)
    if hasattr(response_3, "__await__"):
        import asyncio
        response_3 = asyncio.run(response_3)

    # Assertions for Turn 3 response: Should provide SS2 closing time, remembering context
    assert contains_any(str(response_3), ["ss2", "ss 2"])
    assert contains_any(str(response_3), ["10:00 pm", "10pm"]) # Check for mock closing time

    # Verify message history contains all turns (3 human, 3 AI)
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 6
    assert "petaling jaya" in str(history.messages[0].content).lower()
    assert "ss 2" in str(history.messages[2].content).lower()
    assert "closing time" in str(history.messages[4].content).lower()
    # Also check that the bot's stored responses reflect the mock outputs
    assert contains_any(str(history.messages[1].content), ["which outlet", "specific outlet", "petaling jaya", "pj"])
    assert contains_any(str(history.messages[3].content), ["9:00 am", "9am"])
    assert contains_any(str(history.messages[5].content), ["10:00 pm", "10pm"])


def test_interrupted_conversation_new_session_for_outlet_info():
    """
    INTERRUPTED PATH: Tests that a new session doesn't carry context from previous.
    User asks for outlet info without providing enough details in a fresh session.
    """
    # 1. Arrange
    controller = ChatbotController()
    
    # Simulate a previous unrelated conversation that established "Petaling Jaya" context
    # (though for a new session, this history won't matter)
    import asyncio
    prev_result = controller.process_user_input("I am looking for a coffee shop nearby", "old_session_unrelated")
    if hasattr(prev_result, "__await__"):
        prev_result = asyncio.run(prev_result)

    # 2. Act - Start a new, clean session (simulating interruption/new user)
    # User immediately asks a context-dependent question without prior info
    response = controller.process_user_input(
        "What are the opening hours?", 
        "new_clean_session"
    )

    # 3. Assert - Bot should ask for clarification as it has no context
    # Await coroutine response if necessary (for async support)
    if hasattr(response, "__await__"):
        import asyncio
        response = asyncio.run(response)
    assert contains_any(str(response), ["which outlet", "specific outlet", "specify a location", "kind of information"])
    
    # Ensure the new session's history is clean and only contains this turn
    new_session_history = controller.get_session_history("new_clean_session")
    assert len(new_session_history.messages) == 2 # 1 User input + 1 Bot's clarification


# --- General Memory and Agentic Action Tests (existing and modified) ---

def test_conversation_stores_message_history_through_controller():
    """
    HAPPY PATH: Verifies that the ChatbotController correctly stores
    and manages conversation history for general chat.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "general_chat_history_test"

    # 2. Act
    response1 = controller.process_user_input("Hello, my name is Alice", session_id)
    response2 = controller.process_user_input("What is my name?", session_id)

    # 3. Assert
    history = controller.get_session_history(session_id)
    messages = history.messages

    assert len(messages) == 4 # 2 human inputs + 2 AI responses
    assert messages[0].content == "Hello, my name is Alice"
    assert messages[2].content == "What is my name?"
    # The AI responses exist (we don't care about exact content for this specific test)
    assert len(messages[1].content) > 0 
    assert len(messages[3].content) > 0 
    # Expect the AI to recall the name "Alice" from history (though LLM output varies)
    assert "alice" in str(messages[3].content).lower()

def test_agentic_calculator_happy_path():
    """
    HAPPY PATH: Tests the planner correctly identifies calculation intent
    and uses the mock calculator tool, returning the expected result.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "calc_happy_path"

    # 2. Act
    response = controller.process_user_input("What is 10 plus 5?", session_id) # Using 'plus'

    # 3. Assert
    # Await coroutine response if necessary (for async support)
    if hasattr(response, "__await__"):
        import asyncio
        response = asyncio.run(response)
    assert "15" in str(response) # Check for the correct calculated result (10 + 5)
    
    # Verify history reflects the tool usage
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 2
    assert "10 plus 5" in str(history.messages[0].content).lower()
    assert "15" in str(history.messages[1].content)
    """
    INTERRUPTED PATH: Tests the planner asks for clarification when calculation data is missing.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "calc_missing_info_test"

    # 2. Act
    response = controller.process_user_input("I need a calculation.", session_id)

    # 3. Assert
    # Await coroutine response if necessary (for async support)
    if hasattr(response, "__await__"):
        if hasattr(response, "__await__"):
            import asyncio
            response = asyncio.run(response)
        assert contains_any(str(response), ["provide the numbers", "what would you like me to calculate", "what numbers and operation", "what's the calculation you need help with"])
        
        # Verify history includes the request and the clarification
        history = controller.get_session_history(session_id)
        assert len(history.messages) == 2
        assert "calculation" in str(history.messages[0].content).lower()
        assert contains_any(
            str(history.messages[1].content),
            [
                "provide the numbers",
                "what would you like me to calculate",
                "what numbers and operation",
                "what's the calculation you need help with"
            ]
        )
        # HAPPY PATH: Tests the planner correctly identifies outlet info intent
        # and uses the mock outlet database, returning specific info.
        # 1. Arrange
        controller = ChatbotController()
    session_id = "outlet_specific_query_test"

    # 2. Act
    response = controller.process_user_input("Tell me about the Damansara outlet's closing time.", session_id)

    # 3. Assert
    # Await coroutine response if necessary (for async support)
    if hasattr(response, "__await__"):
        import asyncio
        response = asyncio.run(response)
    assert "damansara" in str(response).lower()
    assert contains_any(str(response), ["11:00 pm", "11pm"]) # Expected closing time for Damansara mock

    history = controller.get_session_history(session_id)
    assert len(history.messages) == 2
    assert "damansara outlet" in str(history.messages[0].content).lower()
    assert contains_any(str(history.messages[1].content), ["11:00 pm", "11pm"])


def test_agentic_outlet_info_missing_details_interrupted_path():
    """
    INTERRUPTED PATH: Tests the planner asks for clarification when outlet info is missing location details.
    """
    # 1. Arrange
    controller = ChatbotController()
    session_id = "outlet_missing_details_test"

    # 2. Act
    response = controller.process_user_input("What are the hours?", session_id)

    # 3. Assert
    # Await coroutine response if necessary (for async support)
    if hasattr(response, "__await__"):
        import asyncio
        response = asyncio.run(response)
    assert contains_any(str(response), ["which outlet", "specific outlet", "specify a location", "kind of information", "what kind of information you're looking for"])
    
    history = controller.get_session_history(session_id)
    assert len(history.messages) == 2
    assert "hours" in str(history.messages[0].content).lower()
    assert contains_any(str(history.messages[1].content).lower(), [
        "which outlet", 
        "specific outlet", 
        "specify a location", 
        "kind of information", 
        "what kind of information you're looking for"
    ])