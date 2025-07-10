# main.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables from the .env file
load_dotenv()

def create_chatbot():
    """
    Configures and creates a stateful chatbot instance with knowledge about outlets.
    """
    llm = ChatGroq(
        temperature=0.7,
        model="llama3-8b-8192",
    )

    # Enhanced prompt with outlet knowledge and memory awareness
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for a coffee shop chain. You have knowledge about various outlets and their information.

Key outlet information you know:
- Petaling Jaya has several outlets including SS 2, SS 15, and Damansara
- SS 2 outlet opens at 9:00 AM and closes at 10:00 PM
- SS 15 outlet opens at 8:00 AM and closes at 9:00 PM  
- Damansara outlet opens at 7:00 AM and closes at 11:00 PM

When users ask about outlets:
1. If they mention a general location, ask which specific outlet
2. If they mention a specific outlet, provide the requested information
3. Always be conversational and remember what was discussed earlier

Remember the conversation context and refer back to previous messages when relevant."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chatbot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Expose the get_session_history function for testing
    chatbot.get_session_history = get_session_history
    
    return chatbot


def run_three_turn_conversation(chatbot, session_id: str = "default"):
    """
    Runs the three-turn conversation flow from the assessment.
    """
    config = RunnableConfig(configurable={"session_id": session_id})
    
    print("=== Three-Turn Conversation Flow ===\n")
    
    # Turn 1
    user_input_1 = "Is there an outlet in Petaling Jaya?"
    print(f"User: {user_input_1}")
    bot_response_1 = chatbot.invoke({"input": user_input_1}, config=config)
    print(f"Bot: {bot_response_1.content}\n")

    # Turn 2  
    user_input_2 = "SS 2, whats the opening time?"
    print(f"User: {user_input_2}")
    bot_response_2 = chatbot.invoke({"input": user_input_2}, config=config)
    print(f"Bot: {bot_response_2.content}\n")

    # Turn 3 - Additional turn to meet "at least three" requirement
    user_input_3 = "What about the closing time?"
    print(f"User: {user_input_3}")
    bot_response_3 = chatbot.invoke({"input": user_input_3}, config=config)
    print(f"Bot: {bot_response_3.content}\n")

    return [bot_response_1.content, bot_response_2.content, bot_response_3.content]


if __name__ == "__main__":
    chatbot = create_chatbot()

    print("Chatbot is ready!")
    print("-" * 50)

    responses = run_three_turn_conversation(chatbot)

    print("-" * 50)
    print("Three-turn conversation completed successfully!")