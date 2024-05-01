import streamlit as st
import random
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from googletrans import Translator
from langchain import OpenAI
import os


pdf_paths = [
    "Cuervo.pdf",
    
    # Add more paths as needed
]


def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=680,
        chunk_overlap=590,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    ##embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def summarize_response(response_content, max_tokens=8000):
    if len(response_content) <= max_tokens:
        return response_content
    else:
        return response_content[:max_tokens] + "..."

load_dotenv()
def get_conversation_chain(vectorstore):
    
    try:
        if "conversation_chain" not in st.session_state:
            llm = ChatOpenAI(
                temperature=0.47,
                openai_api_key=os.getenv('OPENAI_API_KEY') ,
            )
            # Aumentar la capacidad de la memoria aquí
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                max_messages=5, # Ajusta este número según tus necesidades
            )
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            st.session_state.conversation_chain = conversation_chain
        else:
            conversation_chain = st.session_state.conversation_chain

        return conversation_chain
    except KeyError as e:
        # Manejo de la excepción cuando no se encuentra la clave en st.session_state
        st.error(f"Error: {e}. Se esta trabajando en eso")
        return None
    except Exception as e:
        # Manejo de excepciones generales
        st.error(f"Ocurrió un error inesperado: {e}")
        return None


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def translate_text(text, target_lang):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    return translated.text


def start_chat(translated_text_2):
    # Aquí puedes agregar mensajes iniciales del bot para comenzar la conversación
    initial_message =  translated_text_2
    #second_message =  translated_text_3
    
    # Mostrar el mensaje inicial del bot en el chat
    st.write(bot_template.replace("{{MSG}}", initial_message), unsafe_allow_html=True)
    #st.write(user_template.replace("{{MSG}}", second_message), unsafe_allow_html=True)

def render_components_nav():
    bootstrap_css = """
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    """

    custom_css = """
    <style>
    @media (max-width: 768px) {
        .nav-link {
            display: none;
        }
        .img2 {
            display: inline-block;
        }
    }
    </style>
    """

    bootstrap_css_2 = """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    """

    navbar_html = """
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background: #bdb76b;">
        <a class="navbar-brand" href="#" target="_blank">
        <img src='https://upload.wikimedia.org/wikipedia/commons/e/e6/Logo_de_Jos%C3%A9_Cuervo.svg' width='125' /></a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <a href="https://upload.wikimedia.org/wikipedia/commons/e/e6/Logo_de_Jos%C3%A9_Cuervo.svg"><img class="img2" src='https://i.ibb.co/288YzSp/B2New.jpg' width='30' /></a>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
                <li class="nav-item active">
                    <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">Instagram</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Facebook</a>
                </li>
            </ul>
            <ul class="navbar-nav ml-auto">
                <li class="nav-item">
                    <a class="nav-link" href="" target="_blank"><img class="img2" src='https://i.ibb.co/288YzSp/B2New.jpg' width='50' /></a>
                </li>
            </ul>
        </div>
    </nav>
    """

    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

    # Render the HTML and CSS using Markdown
    st.markdown(bootstrap_css, unsafe_allow_html=True)
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(bootstrap_css_2, unsafe_allow_html=True)
    st.markdown(navbar_html, unsafe_allow_html=True)
    st.markdown(hide_st_style, unsafe_allow_html=True)

def main_Free():
    load_dotenv()
    st.set_page_config(page_title="Jose Cuervo | PEPE",
                       page_icon="https://upload.wikimedia.org/wikipedia/commons/e/e6/Logo_de_Jos%C3%A9_Cuervo.svg")
    
    
    st.write(css, unsafe_allow_html=True)
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    render_components_nav()
    


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.markdown('<div class="container">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        text_to_translate = "Mi nombre es PEPE, ¿Como puedo ayudarte?"
        text_to_translate2 = "Hola, estoy listo para ayudarte.." 
        #additional_text = "Por favor, introduce tus datos para continuar."
        tittle_select = "Selecciona el idioma | Select Language"
        text_thinking = "Pensando..."
        text_question = "Escribe tu pregunta aqui"

        target_language = st.selectbox(f":black[{tittle_select}]", ["en", "es", "fr", "de", "it", "ja", "ko", "zh-CN"])
        
        

        translated_text_1 = translate_text(text_to_translate, target_language)
        translated_text_2 = translate_text(text_to_translate2, target_language)
        translated_text_3 = translate_text(text_thinking, target_language)
        translated_text_4 = translate_text(text_question, target_language)

        st.write(
            f"<h1 style='text-align: center; gap: 0rem;'>{translated_text_1}</h1>",
            unsafe_allow_html=True
        )

    # Agregar contenido al segundo elemento (imagen)
    with col2:
        st.write(
    "<div style='display: flex; justify-content: center; padding-bottom: 2rem;'>"
    "<img src='https://i.ibb.co/x1sSxHf/IMG-2593.jpg' width='200' />"
    "</div>",
    unsafe_allow_html=True
    )


    # Process the PDFs from the defined paths
    with st.spinner(translated_text_3):
        raw_text = get_pdf_text(pdf_paths)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        start_chat(translated_text_2)
        

    # Create a container to hold the chat messages
    chat_container = st.empty()
    

    # Create a container for the user input at the bottom
    input_container = st.empty()


    
    # Place the input field inside the input container
    with input_container:
        prompt = st.chat_input(translated_text_4)
        
        
        
    
    

    # Process user input and update chat
    if prompt:
        handle_userinput(prompt)

    

    # Add spacing at the end to push the input container to the bottom
    st.markdown('<style>div.css-1aumxhk { margin-top: auto; }</style>', unsafe_allow_html=True)

def main_Pay():
    load_dotenv()
    st.set_page_config(page_title="Find Tequila",
                       page_icon="https://i.ibb.co/whtfXgN/IMG-3702-2.jpg")
    
    
    st.write(css, unsafe_allow_html=True)
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    render_components_nav()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.markdown('<div class="container">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        text_to_translate = "Mi nombre es Agavi, ¿Como puedo ayudarte?"
        additional_text = "Necesito ayuda personalizada | Click Aqui 🧑🏻‍💻"
        tittle_select = "Selecciona el idioma | Select Language"

        target_language = st.selectbox(f":green[{tittle_select}]", ["en", "es", "fr", "de", "it", "ja", "ko", "zh-CN"])

        translated_text_1 = translate_text(text_to_translate, target_language)
        translated_text_2 = translate_text(additional_text, target_language)

        st.write(
            f"<h1 style='text-align: center; gap: 0rem;'>{translated_text_1}</h1>",
            unsafe_allow_html=True
        )

    # Agregar contenido al segundo elemento (imagen)
    with col2:
        st.write(
    "<div style='display: flex; justify-content: center; padding-bottom: 2rem;'>"
    "<img src='https://i.ibb.co/TgcShwr/AgaviNew.jpg' width='200' />"
    "</div>",
    unsafe_allow_html=True
    )
    
    st.write(
    f"<a style='color: green' href='https://wa.me/3741011240'><h4 style='text-align: center;'>{translated_text_2}</h4></a>",
    unsafe_allow_html=True
    )


    # Process the PDFs from the defined paths
    raw_text = get_pdf_text(pdf_paths)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)

    # Create a container to hold the chat messages
    chat_container = st.empty()

    # Show chat messages from chat history if available
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                chat_container.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                summarized_response = summarize_response(message.content)
                chat_container.write(bot_template.replace("{{MSG}}", summarized_response), unsafe_allow_html=True)

    # Show chat messages of the test chat(s)
    

    # Create a container for the user input at the bottom
    input_container = st.empty()


    
    # Place the input field inside the input container
    with input_container:
        prompt = st.chat_input("Type Here.. | Pregunta Aqui... ")
        
        
        
    
    

    # Process user input and update chat
    if prompt:
        handle_userinput(prompt)
    #Aqui vamos a agregar una funcion que salgan anuncios


    # Add spacing at the end to push the input container to the bottom
    st.markdown('<style>div.css-1aumxhk { margin-top: auto; }</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    main_Free()
