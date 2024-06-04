import streamlit as st
from src.generator import Llama_3
from src.retreiver import get_relevant_contexts


# GENERATOR MODEL INIT
chatbot = Llama_3()


##### STREAMLIT UI SIDE-BAR CODE
st.caption("<h4 style='text-align: center;'>Makesh bot can answer any questions that you might have about my profile using Retrieval-Augmented Generation (RAG). Chat and learn more!</h4>", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='font-size: 48px;'>Makesh Bot</h1>", unsafe_allow_html=True)

st.sidebar.markdown("""
<style>
.inline-form {
    display: flex;
    align-items: left;
    justify-content: space-between;
}
.inline-logo {
    height: 30px;  /* Adjust the logo size as needed */
    width: auto;
    margin-left: 0px;  /* Space between button and logo */
}
</style>
""", unsafe_allow_html=True)

with st.sidebar.container():
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("New Chat"):
            st.session_state.messages = []
    with col2:
        st.markdown("""
            <a href="https://www.linkedin.com/in/makesh-srinivasan/">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/LinkedIn_Logo.svg/1280px-LinkedIn_Logo.svg.png" alt="LinkedIn" class="inline-logo">
            </a>
            """, unsafe_allow_html=True)
        
with st.sidebar.expander("**Quick Settings**", expanded=True):
    options_behaviour = ["Fun 🎨", "Professional 📚"]
    behaviour_choice = st.radio("How do you want the bot to behave?", options_behaviour, index=1, help="'Fun' allows the bot to be a little more creative and fun, 'Professional' encourages the bot to be exact and professional.")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("<p>About the back-end</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='font-size: 12px;'>Meta's Llama 3 8B is hosted on NVIDIA NIM thanks to their limited & free model hosting options. The vector store is accessed via Pinecone's API.</p>", unsafe_allow_html=True)
    st.sidebar.markdown("</br>", unsafe_allow_html=True)
    if behaviour_choice == "Fun 🎨":
        st.session_state.temperature_default = 70
        st.session_state.maxtokens_default = 1024
    else:  # professional
        st.session_state.temperature_default = 20
        st.session_state.maxtokens_default = 900

# Hidden advanced settings
with st.sidebar.expander("Advanced Settings", expanded=False):
    st.caption("<h6 style='text-align: center;'></h6>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 12px; text-align: center;'>If you opened this, you must be familiar with RAG. Play around with these settings!</p>", unsafe_allow_html=True)
    st.session_state.temperature_value = st.slider("Temperature 🌡️", 0, 100, st.session_state.temperature_default, help="High temperature means creative outputs, low means factual and exact!")
    st.session_state.maxlen_generation = st.slider("Max tokens 📝", value=st.session_state.maxtokens_default, step=20, max_value=1500, min_value=50, help="Maximum number of tokens you want to generate")
    st.session_state.k_for_context_retreival = st.number_input("K: Top-k contexts for retreival 🔎", value=10, max_value=10, min_value=2, help="Top k contexts that you want the model to retreive from RAG to generate your answer.")
    # stream_value = ['No', 'Yes']
    # default_option_stream_index = stream_value.index("No")
    # stream_bool_value = st.radio("Stream:", stream_value, index=default_option_stream_index, help="Stream allows the LLM decoder model to output texts to the UI as streams continuously. Selecting No makes the model print the answer on the UI AFTER the answer is generated entirely")
    # if stream_bool_value== "Yes":
    #     stream_bool_value_converted = True
    # else:
    #     stream_bool_value_converted = False



##### STREAMLIT FUNCTIONAL CODE

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
    retreived_contexts = get_relevant_contexts(prompt, top_k=st.session_state.k_for_context_retreival)
    
    custom_model_params = {"temperature": st.session_state.temperature_value/100,"top_p": 1,"max_tokens": st.session_state.maxlen_generation,"stream": False}
    llm_answer = chatbot.generate_answer(query=prompt, retreived_contexts=retreived_contexts, chat_history=st.session_state.messages, custom_model_params=custom_model_params)
    
    print(f"\t* QUESTION: {prompt}\n\t* CONTEXT: {retreived_contexts}\n\t* ANSWER: {llm_answer}")
    
    # Display assistant response in chat message container
    response = f"Virtual Makesh: {llm_answer}"
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})