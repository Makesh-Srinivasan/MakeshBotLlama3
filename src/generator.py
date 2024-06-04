import requests
import json
import streamlit as st


LLAMA3_ENDPOINT = str(st.secrets["LLAMA3_ENDPOINT"])
LLAMA3_API_KEY = str(st.secrets["LLAMA3_API_KEY"])


class Llama_3:
    def __init__(self):
        self.model_name = "Llama 3"

    def generate_answer(self, query, retreived_contexts, chat_history, custom_model_params={"temperature": 0.1, "top_p": 1, "max_tokens": 1024, "stream": False}):
        PROMPT = f"""You are a virtual AI chat-bot created to mimic the behavior and expertise of me (Makesh Srinivasan). You have access to comprehensive data including my work experience, education, publications, research interests, personal projects, and academic achievements. This data serves as the context for generating your responses and it is provided under 'Relevant context' below. Your task is to respond to questions by using the retrieved contexts (given under 'Relevant context' below) relevant to the query, and also consider the chat history (given under "Chat history"). You should simulate my response style (friendly, humble and professional with humour) and use only the information available from the retrieved contexts. If the retrieved contexts do not contain relevant information, you should refrain from making unfounded assertions and should advise asking a different question. The question, relevant contexts and the chat history are as follows.
        
        Question: {query}

        Chat history: {chat_history}

        Relevant context: {retreived_contexts}

        NOTE: If any questions about my personal and sensitive details arise, politely mention them to contact me directly via email (makesh.srinivasan@nyu.edu). Do not provide my phone number or address. You can mention my LinkedIn URL (https://www.linkedin.com/in/makesh-srinivasan/)
        """

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + LLAMA3_API_KEY
        }

        # Prepare the payload
        data = {
            "model": "meta/llama3-8b-instruct",
            "messages": [{"role":"user","content":PROMPT}],
            "temperature": custom_model_params["temperature"],
            "top_p": custom_model_params["top_p"],
            "max_tokens": custom_model_params["max_tokens"],
            "stream": custom_model_params["stream"]
        }

        # Convert dictionary to JSON string format
        data_json = json.dumps(data)

        # Make the POST request
        response = requests.post(LLAMA3_ENDPOINT, headers=headers, data=data_json)
        response = response.json()
        response = response['choices'][0]['message']['content']

        return str(response)

