from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


def return_vdb_client(api_key_holder):
    pc = Pinecone(api_key=api_key_holder)
    index = pc.Index("profileai")
    return index


def text_to_vec(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings.tolist()

def get_relevant_contexts(query, api_key, top_k=3, filterer = "resume"):
    index = return_vdb_client(api_key)
    query_embedding = text_to_vec(query)

    encoded_vector_hit = index.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True,
        filter={"genre": {"$eq": f"{filterer}"}}
    )

    vectors_in_text = []
    for match in range(len(encoded_vector_hit["matches"])):
        vectors_in_text.append(encoded_vector_hit["matches"][match])

    context = ""
    for i in range(len(vectors_in_text)):
        context += vectors_in_text[i]["metadata"]["text"] + "\n"

    return context
