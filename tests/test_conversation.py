# tests/test_conversation.py

"""
Tests for the chatbot's conversational memory (Part 1).
Tests both happy paths (success) and interrupted paths (failure/disruption scenarios).
"""

import pytest
from langchain_core.runnables import RunnableConfig
from main import create_chatbot


def test_three_turn_conversation_happy_path():
    """
    HAPPY PATH: Tests that the chatbot can maintain context across 3 related turns.
    This simulates the ideal user experience where everything works smoothly.
    """
    # 1. Arrange
    chatbot = create_chatbot()
    session_config = RunnableConfig(configurable={"session_id": "happy_test"})

    # 2. Act - Execute the three-turn flow
    # Turn 1: Ask about general location
    response_1 = chatbot.invoke(
        {"input": "Is there an outlet in Petaling Jaya?"}, 
        config=session_config
    )
    
    # Turn 2: Specify outlet and ask for opening time
    response_2 = chatbot.invoke(
        {"input": "SS 2, whats the opening time?"}, 
        config=session_config
    )
    
    # Turn 3: Ask follow-up question (should remember SS 2 context)
    response_3 = chatbot.invoke(
        {"input": "What about the closing time?"}, 
        config=session_config
    )

    # 3. Assert - Verify memory worked across all turns
    # Check that all 3 turns got responses
    assert len(response_1.content) > 0
    assert len(response_2.content) > 0  
    assert len(response_3.content) > 0

    # Check that conversation history was maintained (6 messages total: 3 human + 3 AI)
    session_history = chatbot.get_session_history("happy_test")
    assert len(session_history.messages) == 6

    # Verify the exact messages were stored
    messages = session_history.messages
    assert "Petaling Jaya" in messages[0].content
    assert "SS 2" in messages[2].content
    assert "closing time" in messages[4].content


def test_interrupted_conversation_new_session():
    """
    INTERRUPTED PATH: Tests what happens when conversation context is lost 
    (simulating a new session/page refresh/system restart).
    """
    # 1. Arrange
    chatbot = create_chatbot()
    
    # Start a conversation in one session
    session_1_config = RunnableConfig(configurable={"session_id": "session_1"})
    chatbot.invoke(
        {"input": "Is there an outlet in Petaling Jaya?"}, 
        config=session_1_config
    )

    # 2. Act - Try to continue conversation in a NEW session (simulating interruption)
    session_2_config = RunnableConfig(configurable={"session_id": "session_2"})
    response = chatbot.invoke(
        {"input": "What about the closing time?"}, 
        config=session_2_config
    )

    # 3. Assert - Bot should ask for clarification since context is lost
    response_text = response.content.lower()
    assert (
        "which outlet" in response_text or
        "what outlet" in response_text or
        "could you specify" in response_text or
        "which location" in response_text
    )


def test_interrupted_conversation_context_gap():
    """
    INTERRUPTED PATH: Tests what happens when user jumps topics mid-conversation
    (simulating user confusion or topic switching).
    """
    # 1. Arrange
    chatbot = create_chatbot()
    session_config = RunnableConfig(configurable={"session_id": "context_gap_test"})

    # 2. Act - Start outlet conversation, then suddenly ask about something unrelated
    chatbot.invoke(
        {"input": "Is there an outlet in Petaling Jaya?"}, 
        config=session_config
    )
    
    # Suddenly ask about weather (context gap)
    response = chatbot.invoke(
        {"input": "What's the weather like?"}, 
        config=session_config
    )

    # 3. Assert - Bot should handle the topic change gracefully
    assert len(response.content) > 0
    
    # Verify conversation history still maintained both topics
    session_history = chatbot.get_session_history("context_gap_test")
    assert len(session_history.messages) == 4  # 2 human + 2 AI messages
    assert "Petaling Jaya" in session_history.messages[0].content
    assert "weather" in session_history.messages[2].content


def test_memory_persistence_across_multiple_turns():
    """
    HAPPY PATH: Extended test with more turns to verify memory doesn't degrade.
    """
    # 1. Arrange
    chatbot = create_chatbot()
    session_config = RunnableConfig(configurable={"session_id": "extended_test"})

    # 2. Act - Have a longer conversation (5 turns)
    inputs = [
        "Is there an outlet in Petaling Jaya?",
        "SS 2, whats the opening time?", 
        "What about the closing time?",
        "Do you have WiFi there?",
        "Thanks for the SS 2 information!"
    ]
    
    responses = []
    for user_input in inputs:
        response = chatbot.invoke({"input": user_input}, config=session_config)
        responses.append(response.content)

    # 3. Assert - All responses should be generated and history maintained
    assert len(responses) == 5
    assert all(len(response) > 0 for response in responses)
    
    # Check total message count (5 human + 5 AI = 10 messages)
    session_history = chatbot.get_session_history("extended_test")
    assert len(session_history.messages) == 10