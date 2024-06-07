import streamlit as st
from streamlit_chat import message
# from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from embedchain import Pipeline as EmbedChainPipeline

import os

#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Having trouble sleeping? Let's chat about it!"}
    ]

# Initialize ChatOpenAI and ConversationChain
# llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
llm = ChatGoogleGenerativeAI(model = "gemini-pro")
# llm = ChatAnthropic(model_name="claude-3-sonnet-20240229")

conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

# Create user interface
st.title("🌙 Fall Asleep Faster 😴")
st.subheader("Tips and advice from Huberman's podcast to improve your sleep")

# YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=0rp9PYP6lws&ab_channel=HubermanLabClips"

# Initialize the embedchain pipeline
embedchain_pipeline = EmbedChainPipeline()

# Add the YouTube video to the embedchain pipeline
embedchain_pipeline.add(youtube_url)

if prompt := st.chat_input("How's your sleep lately? Got any questions?"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Generate embeddings for the user's question
    question_embeddings = embedchain_pipeline.query(prompt)

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input = prompt, context=question_embeddings)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history
