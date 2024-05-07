import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from colorthief import ColorThief
import webcolors
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

# age = "22"
# disease = "Diabetes, Thyroid"
# MnC = "28"

if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text')
    st.session_state.loader = WebBaseLoader("https://www.healthline.com/health/vaginal-discharge-color-guide")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap = 200
    )
    
    print("Data Loaded!")
    
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    print("CromaDb Ready!")
    
    
llm = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = "llama3-70b-8192"
)

template = """
You are a medical chatbot specializing in menstrual health.
Analyse user's details and give personalise response.
You can ask some follow up questions and remember them to provide more personalised answers.

Ask questions like what is her age, normal flow cycle length, if she has any disease or not for better personalised information.
Keep a positive attitute and provide health tips.
Dont ask too much question.

You are created by "Sourik Poddar".

<context>
{context}
<context>

Chat History:
{chat_history}
Question: {user_input}
"""

print("Prompt Ready!")

prompt = PromptTemplate(
    input_variables=["chat_history", "user_input", "context"], template=template
)

print("Prompt Template Ready!")

# message_history = ChatMessageHistory()
# memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
print("message_history Ready!")

memory = ConversationBufferMemory(
            memory_key="chat_history",
            # output_key="answer",
            # chat_memory=message_history,
            # return_messages=True,
            input_key="user_input"
        )

print("memory Ready!")

chain = load_qa_chain(
    llm, chain_type="stuff", memory=memory, prompt=prompt
)

print("chain Ready!")

def get_blood_color(uploaded_file):

    ct = ColorThief(uploaded_file)
    dominant_color = ct.get_color(quality=1)

    def closest_color(rgb):
        differences = {}
        for color_hex, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
            r, g, b = webcolors.hex_to_rgb(color_hex)
            differences[sum([(r - rgb[0]) ** 2,
                            (g - rgb[1]) ** 2,
                            (b - rgb[2]) ** 2])] = color_name
            
        return differences[min(differences.keys())]
        
    return closest_color(dominant_color)
    

st.title("FlowGPT-3.5 💖")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your Blood Color Image Here! 🩸", type=['png', 'jpg'], help="Please keep a white background")
        

if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
else:
    for msg in st.session_state.chat_history:
        memory.save_context({'user_input': msg['human']}, {'output': msg['AI']})

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

user_input = st.chat_input("Pass Your Prompt Here")
user_input = ("My Menstrual Blood Color is: " + get_blood_color(uploaded_file)) if uploaded_file else None

if user_input:
    st.chat_message('user').markdown(user_input)
    st.session_state.messages.append({'role':'user', 'content': user_input})
    
    docs = st.session_state.vectors.similarity_search(user_input)

    response = chain(
        {"input_documents":docs, "user_input": user_input}
        , return_only_outputs=True)
    
    answer = response["output_text"]
    
    st.chat_message('assistant').markdown(answer)
    st.session_state.messages.append({'role':'assistant', 'content': answer})
    
    msg = {'human': user_input, 'AI': answer}
    st.session_state.chat_history.append(msg)