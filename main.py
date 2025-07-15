import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory
import asyncio 

from planner import AgenticPlanner, Intent, Action, call_calculator_api, get_mock_outlet_info

load_dotenv()

class ChatbotController:
    def __init__(self):
        self.planner = AgenticPlanner()
        self.llm = ChatGroq(
            temperature=0.7,
            model="llama3-8b-8192", 
        )

        self._history_store = {} 

        self.general_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly assistant."),
            MessagesPlaceholder(variable_name="history"), 
            ("human", "{input}"),
        ])
        
        self.conversation_with_history = RunnableWithMessageHistory(
            self.general_chat_prompt | self.llm, 
            self.get_session_history, 
            input_messages_key="input", 
            history_messages_key="history", 
        )
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._history_store:
            self._history_store[session_id] = ChatMessageHistory()
        return self._history_store[session_id]

    async def process_user_input(self, user_input: str, session_id: str = "default") -> str:
        config = RunnableConfig(configurable={"session_id": session_id})

        planning_result = self.planner.plan_next_action(user_input)
        
        print(f"\n[DEBUG] Planner Intent: {planning_result.intent}")
        print(f"[DEBUG] Planner Action: {planning_result.action}")
        print(f"[DEBUG] Extracted Data: {planning_result.extracted_data}")
        print(f"[DEBUG] Missing Info: {planning_result.missing_info}\n")

        # Initialize response_content to an empty string to guarantee it's always a string
        response_content: str = "" 

        history = self.get_session_history(session_id) 

        if planning_result.action == Action.ASK_FOR_INFO:
            # Ensure missing_info is always treated as a string, provide a fallback.
            response_content = planning_result.missing_info if planning_result.missing_info is not None else "I need more information."
            history.add_user_message(user_input)
            history.add_ai_message(response_content)
            
        elif planning_result.action == Action.USE_CALCULATOR:
            extracted = planning_result.extracted_data
            if extracted:
                print(f"[DEBUG] Attempting to call calculator API with: {extracted}")
                try: 
                    api_response_raw = await call_calculator_api(
                        extracted['num1'], extracted['operator'], extracted['num2']
                    )
                    # Ensure the API response is treated as a string.
                    response_content = api_response_raw if api_response_raw is not None else "Calculator API returned an empty response."
                    print(f"[DEBUG] Calculator API returned: {response_content}")
                    
                except Exception as e: 
                    response_content = f"An unexpected critical error occurred during calculator API call: {e}"
                    print(f"[DEBUG] Exception during calculator API call (critical): {e}") 
            else: 
                response_content = "I encountered an issue with the calculation. Could you please rephrase the calculation clearly?"
                print(f"[DEBUG] Calculator extraction failed, asking for rephrase: {response_content}")
            
            history.add_user_message(user_input)
            history.add_ai_message(response_content)
            
        elif planning_result.action == Action.USE_OUTLET_DB:
            extracted = planning_result.extracted_data
            if extracted:
                response_content = get_mock_outlet_info(
                    extracted.get('location'), extracted.get('info_type')
                )
                # Ensure the mock outlet response is treated as a string.
                response_content = response_content if response_content is not None else "Mock outlet info returned empty."
            else:
                response_content = "I need more details to find outlet information. Please specify a location or what you're looking for."
            history.add_user_message(user_input)
            history.add_ai_message(response_content)
        
        elif planning_result.action == Action.RESPOND_DIRECTLY:
            response = await self.conversation_with_history.ainvoke(
                {"input": user_input},
                config=config
            )
            response_content = response.content
            print(f"[DEBUG] LLM responded: {response_content}")
        
        else: # Fallback for UNKNOWN intent or any truly unhandled action type
            response_content = "I'm not sure how to handle that request. Can you rephrase?"
            history.add_user_message(user_input)
            history.add_ai_message(response_content)
            print(f"[DEBUG] Fallback to unknown/unhandled action: {response_content}")

        print(f"[DEBUG] Final response_content before return: {response_content}")
        return response_content


def run_interactive_conversation():
    controller = ChatbotController()
    session_id = "interactive_session" 

    print("Chatbot is ready! Type 'exit' to end the conversation.")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        bot_response = asyncio.run(controller.process_user_input(user_input, session_id)) 
        print(f"Bot: {bot_response}")
        print("-" * 30)

    print("Conversation ended.")

if __name__ == "__main__":
    run_interactive_conversation()