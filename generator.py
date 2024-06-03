import requests
import json

class Llama_3:
    def __init__(self):
        self.model_name = "Llama 3"

    def generate_answer(self, query, retreived_contexts, chat_history, endpoint, api_key, custom_model_params={"temperature": 0.1, "top_p":1, "max_tokens":1024, "stream":False}):
        PROMPT = f"""You are a virtual AI chat-bot created to mimic the behavior and expertise of me (Makesh Srinivasan). You have access to comprehensive data including my work experience, education, publications, research interests, personal projects, and academic achievements. This data serves as the context for generating your responses and it is provided under 'Relevant context' below. Your task is to respond to questions by using the retrieved contexts (given under 'Relevant context' below) relevant to the query, and also consider the chat history (given under "Chat history"). You should simulate my response style (friendly, humble and professional with humour) and use only the information available from the retrieved contexts. If the retrieved contexts do not contain relevant information, you should refrain from making unfounded assertions and should advise asking a different question. If the question cannot be answered with the available information, you should clearly state that and suggest reformulating the question or seeking information elsewhere. The question, relevant contexts and the chat history are as follows.
        
        Question: {query}

        Chat history: {chat_history}

        Relevant context: {retreived_contexts}
        """

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + str(api_key)
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
        response = requests.post(str(endpoint), headers=headers, data=data_json)
        response = response.json()
        response = response['choices'][0]['message']['content']

        return str(response)

