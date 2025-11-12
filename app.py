import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

llm = ChatOpenAI(openai_api_key=api_key, temperature=0)

def get_llm_response(input_text: str, expert_type: str) -> str:
    """
    Sends a prompt to the LangChain LLM based on the input text and expert type.

    Args:
        input_text (str): The user-provided text input.
        expert_type (str): The selected expert type (e.g., 'A' or 'B').

    Returns:
        str: The response from the LLM.
    """
    # Define system messages for different expert types
    system_messages = {
        "A": "You are an expert in food. Answer questions related to food expertise.",
        "B": "You are an expert in travel. Answer questions related to travel expertise."
    }

    # Get the system message based on the expert type
    system_message = system_messages.get(expert_type, "You are a general expert.")

    # Create the conversation messages
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=input_text)
    ]

    # Get the response from the LLM
    response = llm(messages)

    return response.content

# Streamlit UI components
st.title("LLM Expert System")

# Input form for user text
user_input = st.text_input("Enter your text:")

# Radio buttons for selecting expert type
expert_type = st.radio(
    "Select the type of expert:",
    ["A (食の専門家)", "B (旅の専門家)"]
)

# Display app overview and instructions
st.markdown("""
# LLM Expert System
このアプリでは、以下の操作を行うことができます：
1. テキスト入力欄に質問を入力してください。
2. ラジオボタンで「食の専門家」または「旅の専門家」を選択してください。
3. 「Submit」ボタンを押すと、選択した専門家としてLLMが回答を生成します。

**注意**: 正確な回答を得るために、具体的な質問を入力してください。
""")

# Display the LLM response when the user submits input
if st.button("Submit"):
    if user_input:
        # Call the function to get the LLM response
        response = get_llm_response(user_input, expert_type)
        
        # Display the response on the app
        st.subheader("LLM Response:")
        st.write(response)
    else:
        st.warning("Please enter some text before submitting.")