import streamlit as st
import httpx
import json
import asyncio
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Zus Coffee Assistant",
    page_icon="☕",
    layout="centered"
)

# Custom CSS for chat bubbles
st.markdown("""
<style>
/* Message containers */
.user-message-container, .assistant-message-container {
    display: flex;
    gap: 15px;
    align-items: flex-end;
    animation: fadeIn 0.3s ease-in-out;
    margin: 5px 0;
}

.user-message-container {
    justify-content: flex-end;
}

.assistant-message-container {
    justify-content: flex-start;
}

/* Avatar styling */
.avatar {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.assistant-avatar {
    background-color: #6B4F4F;
    color: white;
}

.user-avatar {
    background-color: #483434;
    color: white;
}

/* Chat bubble styling */
.chat-bubble {
    padding: 12px 18px;
    border-radius: 20px;
    max-width: 60%;
    white-space: pre-wrap;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    line-height: 1.5;
    font-size: 15px;
}

.user-message {
    background: linear-gradient(135deg, #483434 0%, #6B4F4F 100%);
    color: white;
    border-radius: 20px 20px 5px 20px;
    margin-left: auto;
}

.assistant-message {
    background: #FFF1E6;
    color: #483434;
    border-radius: 20px 20px 20px 5px;
    margin-right: auto;
}

/* Animations */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """👋 Hi! I'm your Zus Coffee assistant. How can I assist you today?

\nSamples:
"Is there any outlets in Petaling Jaya?"
"I want to know more about Zus Mug."
"What time does Zus Coffee SS2 open?"
"Calculate 2 + 3"
"""}
    ]

# Display chat header
st.title("☕ Zus Coffee Assistant")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'''
            <div class="user-message-container">
                <div class="chat-bubble user-message">{message["content"]}</div>
                <div class="avatar user-avatar">You</div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="assistant-message-container">
                <div class="avatar assistant-avatar">☕</div>
                <div class="chat-bubble assistant-message">{message["content"]}</div>
            </div>
        ''', unsafe_allow_html=True)
  
# Function to process the submitted message
def handle_submit():
    if st.session_state.user_input:
        user_message = st.session_state.user_input
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Get bot response
        response = asyncio.run(process_message(user_message))
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Reset the input
        st.session_state.user_input = ""

# Chat input
st.text_input(
    label="Chat input",
    key="user_input",
    placeholder="Ask me anything...",
    on_change=handle_submit,
    label_visibility="collapsed"
)

async def process_message(message: str) -> str:
    """Process the user message and return appropriate response"""
    message = message.lower().strip()
    
    try:
        # Check for product-related queries
        if any(word in message for word in ["menu", "product", "drink", "food", "coffee", "price"]):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/products",
                    json={"query": message, "top_k": 3}
                )
                if response.status_code == 200:
                    products = response.json()
                    response_text = "Here's what I found:\n\n"
                    for product in products:
                        response_text += f"• {product['name']} - RM{product['price']:.2f}\n"
                        response_text += f"  {product['description']}\n\n"
                    return response_text
        
        # Check for outlet-related queries
        elif any(word in message for word in ["outlet", "store", "location", "where", "open", "close"]):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/outlets",
                    json={"query": message}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data["results"]:
                        response_text = "Here are the outlets I found:\n\n"
                        for outlet in data["results"]:
                            response_text += f"📍 {outlet['name']}\n"
                            response_text += f"📫 {outlet['address']}\n"
                            response_text += f"⏰ {outlet['opening_time']} - {outlet['closing_time']}\n"
                            if outlet['services']:
                                response_text += f"✨ Services: {', '.join(outlet['services'])}\n"
                            response_text += "\n"
                        return response_text
                    else:
                        return "I couldn't find any outlets matching your query. Could you please try rephrasing?"
        
        # Check for calculation queries
        elif any(word in message for word in ["calculate", "sum", "add", "subtract", "multiply", "divide"]):
            # Extract numbers and operator from message
            # This is a simple implementation - you might want to make it more sophisticated
            parts = message.split()
            nums = []
            operator = None
            for part in parts:
                if part.replace('.', '').isdigit():
                    nums.append(float(part))
                elif part in ['+', '-', '*', '/']:
                    operator = part
            
            if len(nums) >= 2 and operator:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:8000/calculate",
                        json={"num1": nums[0], "operator": operator, "num2": nums[1]}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return f"The result is: {result['result']}"
        
        # Default responses for common queries
        elif "hi" in message or "hello" in message:
            return "👋 Hi there! How can I help you today?"
        
        elif "thank" in message:
            return "You're welcome! Is there anything else I can help you with?"
        
        elif "bye" in message:
            return "👋 Goodbye! Have a great day!"
        
        # Default response
        return ("I'm not sure how to help with that. I can:\n"
                "• Find Zus Coffee outlets\n"
                "• Show you our menu and products\n"
                "• Help with calculations for your order\n\n"
                "Could you please try rephrasing your question?")
    
    except Exception as e:
        return f"I apologize, but I encountered an error. Please try again or rephrase your question." 