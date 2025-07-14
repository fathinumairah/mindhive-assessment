# main.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory

# Import components from our planner
from planner import AgenticPlanner, Intent, Action, perform_simple_calculation, get_mock_outlet_info

# Load environment variables from the .env file
load_dotenv()


class ChatbotController:
    """
    Acts as the central controller for the chatbot.
    It integrates the planner, LLM, and tool execution.
    """
    def __init__(self):
        self.planner = AgenticPlanner()
        self.llm = ChatGroq(
            temperature=0.7,
            model="llama3-8b-8192",
        )

        # Chat message history store (managed by get_session_history)
        self._history_store = {} 

        # Base chain for general conversation (when no tool is needed)
        # This prompt will be used for Intent.GENERAL_CHAT
        self.general_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly assistant."), # Relaxed prompt
            MessagesPlaceholder(variable_name="history"), # History placeholder
            ("human", "{input}"),
        ])
        
        # The RunnableWithMessageHistory automatically manages adding messages to history
        # for paths where its .invoke() method is called.
        self.conversation_with_history = RunnableWithMessageHistory(
            self.general_chat_prompt | self.llm, # Chain the prompt and LLM
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    
    def get_session_history(self, session_id: str):
        """Retrieves or creates a chat history for the given session."""
        # This is the function passed to RunnableWithMessageHistory
        if session_id not in self._history_store:
            self._history_store[session_id] = ChatMessageHistory()
        return self._history_store[session_id]

    def process_user_input(self, user_input: str, session_id: str = "default") -> str:
        """
        Main method to process user input, decide action, and execute it.
        """
        config = RunnableConfig(configurable={"session_id": session_id})

        # Step 1: The Planner analyzes intent and missing info
        planning_result = self.planner.plan_next_action(user_input)
        
        print(f"\n[DEBUG] Planner Intent: {planning_result.intent}")
        print(f"[DEBUG] Planner Action: {planning_result.action}")
        print(f"[DEBUG] Extracted Data: {planning_result.extracted_data}")
        print(f"[DEBUG] Missing Info: {planning_result.missing_info}\n")

        response_content = "" # Initialize variable to hold the final response

        # Step 2 & 3: Choose and Execute Action
        if planning_result.action == Action.ASK_FOR_INFO:
            response_content = planning_result.missing_info
            # Manually add to history for these tool-independent responses
            history = self.get_session_history(session_id)
            history.add_user_message(user_input)
            history.add_ai_message(response_content)
            
        elif planning_result.action == Action.USE_CALCULATOR:
            extracted = planning_result.extracted_data
            if extracted:
                response_content = perform_simple_calculation(
                    extracted['num1'], extracted['operator'], extracted['num2']
                )
            else: 
                response_content = "I encountered an issue with the calculation. Could you please rephrase the calculation clearly?"
            # Manually add to history for these tool-independent responses
            history = self.get_session_history(session_id)
            history.add_user_message(user_input)
            history.add_ai_message(response_content)
            
        elif planning_result.action == Action.USE_OUTLET_DB:
            extracted = planning_result.extracted_data
            if extracted:
                response_content = get_mock_outlet_info(
                    extracted.get('location'), extracted.get('info_type')
                )
            else:
                response_content = "I need more details to find outlet information. Please specify a location or what you're looking for."
            # Manually add to history for these tool-independent responses
            history = self.get_session_history(session_id)
            history.add_user_message(user_input)
            history.add_ai_message(response_content)
        
        elif planning_result.action == Action.RESPOND_DIRECTLY:
            # Use the LLM for general conversation. RunnableWithMessageHistory will add to history automatically.
            llm_response = self.conversation_with_history.invoke(
                {"input": user_input},
                config=config
            )
            response_content = llm_response.content
        
        else: # Fallback for UNKNOWN intent or unhandled action
            response_content = "I'm not sure how to handle that request. Can you rephrase?"
            # Manually add to history for this fallback
            history = self.get_session_history(session_id)
            history.add_user_message(user_input)
            history.add_ai_message(response_content)

        return str(response_content) if response_content is not None else ""


def run_interactive_conversation():
    """Runs an interactive loop for the user to chat with the bot."""
    controller = ChatbotController()
    session_id = "interactive_session"

    print("Chatbot is ready! Type 'exit' to end the conversation.")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        bot_response = controller.process_user_input(user_input, session_id)
        print(f"Bot: {bot_response}")
        print("-" * 30)

    print("Conversation ended.")

if __name__ == "__main__":
    run_interactive_conversation()