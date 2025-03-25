import streamlit as st # for user interface 
from PyPDF2 import PdfReader # to read pdf 
import os #os.getenv() This method is useful to avoid hardcoding sensitive data like google api directly in the script
import requests #fetching content from URLs.
from bs4 import BeautifulSoup  # For extracting text from URLs
from datetime import datetime, timedelta # calculate a time difference, like adding or subtracting days, hours, or minutes from a given date and time

# Update imports for LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter # converts data into chunks 
from langchain_google_genai import GoogleGenerativeAIEmbeddings # converts chunks into vector 
from langchain_community.vectorstores import FAISS # stores vector locally and retrieve later for similarity search 
from langchain_google_genai import ChatGoogleGenerativeAI #provides access for gemini for chat based applications 
from langchain.chains.question_answering import load_qa_chain #helps the chatbot find answers to user questions using given documents (PDF or URL content).
from langchain.prompts import PromptTemplate #set of rule for ai model to behave

def nivethas_pdf_text_extractor(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def nivethas_url_text_extractor(url):
    """Extracts readable text from a webpage using BeautifulSoup"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        return text
    except Exception as e:
        st.error(f"Failed to fetch URL content: {e}")
        return ""

def nivethas_text_breaker(text, model_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def nivethas_faiss_vector_store(text_chunks, model_name, api_key=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def nivethas_rule_for_bot(model_name, vectorstore=None, api_key=None):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say: "answer is not available in the context".

    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def nivethas_genius_minibot(user_question, model_name, api_key, pdf_docs, url, conversation_history):
    if api_key is None:
        st.warning("Please provide API key before processing.")
        return
    
    if pdf_docs:
        text = nivethas_pdf_text_extractor(pdf_docs)
    elif url:
        text = nivethas_url_text_extractor(url)
    else:
        st.warning("Please upload a PDF file or enter a URL.")
        return

    text_chunks = nivethas_text_breaker(text, model_name)
    vector_store = nivethas_faiss_vector_store(text_chunks, model_name, api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = nivethas_rule_for_bot("Google AI", vectorstore=new_db, api_key=api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    user_question_output = user_question
    response_output = response['output_text']
    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    
    conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), url or ", ".join(pdf_names)))

def delete_old_conversations(conversation_history, days):
    cutoff_date = datetime.now() - timedelta(days=days)
    conversation_history[:] = [conv for conv in conversation_history if datetime.strptime(conv[3], '%Y-%m-%d %H:%M:%S') > cutoff_date]

def main():
    st.set_page_config(page_title=" 🔍 Decode PDFs & URLs with Me!", page_icon=":star2:")
    st.header("Let's Have a Chat about Your PDF or URL!🤖 ")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    linkedin_profile_link = "https://www.linkedin.com/in/nivetha-govindaraj-9882842b7/"
    github_profile_link = "https://github.com/nivethasairam"

    st.sidebar.markdown(f"""
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) 
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})
    """)

    model_name = st.sidebar.radio("Select the Model:", ("Google AI",))
    api_key = st.sidebar.text_input("Enter only your Google API Key:")
    st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")

    if not api_key:
        st.sidebar.warning("Please enter your Google API Key to proceed.")
        return

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    url = st.text_input("Enter a URL for text extraction")

    if pdf_docs and url:
        st.warning("Please provide either a PDF or a URL, not both.")
        return

    if st.button("Submit & Process"):
        if pdf_docs or url:
            with st.spinner("on the way ..."):
                st.success("Mission Successful!!")
        else:
            st.warning("Please upload a PDF or enter a URL.")

    user_question = st.text_input("Ask bot a relevant Question")

    if user_question:
        nivethas_genius_minibot(user_question, model_name, api_key, pdf_docs, url, st.session_state.conversation_history)
        st.session_state.user_question = ""  
    
    days_to_keep = st.sidebar.selectbox("Delete conversations older than:", [1, 3, 7, 30])
    if st.sidebar.button("Delete Old Conversations"):
        delete_old_conversations(st.session_state.conversation_history, days_to_keep)
        st.sidebar.success(f"Deleted conversations older than {days_to_keep} days!")

if __name__ == "__main__":
    main()
