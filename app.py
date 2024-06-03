# import streamlit as st
# from generator import Llama_3
# from vdb import get_relevant_contexts
# import os

# PINECONE_API_KEY = str(os.environ.get('PINECONE_API_KEY', 'DefaultAPIKeyIfNotSet'))
# LLAMA3_ENDPOINT = str(os.environ.get('LLAMA3_ENDPOINT', 'DefaultAPIKeyIfNotSet'))
# LLAMA3_API_KEY = str(os.environ.get('LLAMA3_API_KEY', 'DefaultAPIKeyIfNotSet'))

# st.title("Chat with the model trained to answer questions about my profile")

# # Initialize chat history. You must move this to an independent table during production.
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # React to user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     st.chat_message("user").markdown(prompt)
    
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     chatbot = Llama_3()

#     retreived_contexts = get_relevant_contexts(prompt, PINECONE_API_KEY, top_k=3)
    
#     # Query the RAG model with the prompt
#     llm_answer = chatbot.generate_answer(f"{prompt}", retreived_contexts, st.session_state.messages, LLAMA3_ENDPOINT, LLAMA3_API_KEY)
    
#     print(f"\t* QUESTION: {prompt}\n\t* CONTEXT: {retreived_contexts}\n\t* ANSWER: {llm_answer}")
#     # Display assistant response in chat message container
#     response = f"Virtual Makesh: {llm_answer}"
#     with st.chat_message("assistant"):
#         st.markdown(response)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})



import streamlit as st
from generator import Llama_3
from vdb import get_relevant_contexts
import os


PINECONE_API_KEY = str(st.secrets["PINECONE_API_KEY"])
LLAMA3_ENDPOINT = str(st.secrets["LLAMA3_ENDPOINT"])
LLAMA3_API_KEY = str(st.secrets["LLAMA3_API_KEY"])


st.caption("<h4 style='text-align: center;'>Makesh bot can answer any questions that you might have about my profile using Retrieval-Augmented Generation (RAG). Chat and learn more!</h4>", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='font-size: 48px;'>Makesh Bot</h1>", unsafe_allow_html=True)


# Button to clear chat history
if st.sidebar.button("New Chat"):
    st.session_state.messages = []
st.sidebar.markdown("</br>", unsafe_allow_html=True)

with st.sidebar.expander("**Quick Settings**", expanded=True):
    options_behaviour = ["Fun 🎨", "Professional 📚"]
    behaviour_choice = st.radio("How do you want the bot to behave?", options_behaviour, index=1, help="'Fun' allows the bot to be a little more creative and fun, 'Professional' encourages the bot to be exact and professional.")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("<p>About the back-end</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='font-size: 12px;'>Meta's Llama 3 8B is hosted on NVIDIA NIM thanks to their limited & free model hosting options. The vector store is accessed via Pinecone's API.</p>", unsafe_allow_html=True)
    st.sidebar.markdown("</br>", unsafe_allow_html=True)
    if behaviour_choice == "Fun 🎨":
        temperature_default = 70
        maxtokens_default = 1024
    else:  # professional
        temperature_default = 20
        maxtokens_default = 900


# Hidden advanced settings
with st.sidebar.expander("Advanced Settings", expanded=False):
    st.caption("<h6 style='text-align: center;'></h6>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 12px; text-align: center;'>If you opened this, you must be familiar with RAG. Play around with these settings!</p>", unsafe_allow_html=True)
    temperature_value = st.slider("Temperature 🌡️", 0, 100, temperature_default, help="High temperature means creative outputs, low means factual and exact!")
    maxlen_generation = st.slider("Max tokens 📝", value=maxtokens_default, step=20, max_value=1500, min_value=50, help="Maximum number of tokens you want to generate")
    k_for_context_retreival = st.number_input("K: Top-k contexts for retreival 🔎", value=3, max_value=5, min_value=2, help="Top k contexts that you want the model to retreive from RAG to generate your answer.")
    # stream_value = ['No', 'Yes']
    # default_option_stream_index = stream_value.index("No")
    # stream_bool_value = st.radio("Stream:", stream_value, index=default_option_stream_index, help="Stream allows the LLM decoder model to output texts to the UI as streams continuously. Selecting No makes the model print the answer on the UI AFTER the answer is generated entirely")
    # if stream_bool_value== "Yes":
    #     stream_bool_value_converted = True
    # else:
    #     stream_bool_value_converted = False





# Initialize chat history. You must move this to an independent table during production.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What does Makesh study?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retreival
    retreived_contexts = get_relevant_contexts(prompt, PINECONE_API_KEY, top_k=k_for_context_retreival)
    
    chatbot = Llama_3()
    llm_answer = chatbot.generate_answer(f"{prompt}", retreived_contexts, st.session_state.messages, LLAMA3_ENDPOINT, LLAMA3_API_KEY, custom_model_params={"temperature": temperature_value/100,"top_p": 1,"max_tokens": maxlen_generation,"stream": False})
    
    print(f"\t* QUESTION: {prompt}\n\t* CONTEXT: {retreived_contexts}\n\t* ANSWER: {llm_answer}")
    # Display assistant response in chat message container
    response = f"Virtual Makesh: {llm_answer}"
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})