import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import json
import faiss
import pickle

load_dotenv()

# Validate Groq API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Groq API Key not found. Please add GROQ_API_KEY to your .env file.")
    st.stop()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    faiss.write_index(vector_store.index, "faiss_index.bin")
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump({
            "docstore": vector_store.docstore,
            "index_to_docstore_id": vector_store.index_to_docstore_id
        }, f)


def load_vector_store():
    embeddings = get_embeddings()
    index = faiss.read_index("faiss_index.bin")
    with open("faiss_store.pkl", "rb") as f:
        store_data = pickle.load(f)
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=store_data["docstore"],
        index_to_docstore_id=store_data["index_to_docstore_id"]
    )
    return vector_store


def user_input(user_question):
    try:
        vector_store = load_vector_store()
        docs = vector_store.similarity_search(user_question)

        # Build context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Answer the question as detailed as possible from the provided context.
Use pointers and tables to make the answer readable.
If the answer is not in the context, clearly say "Answer not found in the document."
Use Markdown formatting to make the response clear.

Context:
{context}

Question:
{user_question}

Answer:"""

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )

        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        st.session_state.output_text = answer
        st.markdown("**Reply:**")
        st.markdown(answer)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def main():
    st.write("<h1><center>Ask To PDF</center></h1>", unsafe_allow_html=True)
    st.write("")

    try:
        with open('src/Robot.json', encoding='utf-8') as anim_source:
            animation = json.load(anim_source)
        st_lottie(animation, 1, True, True, "high", 100, -200)
    except Exception:
        pass

    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""
    if 'prompt_selected' not in st.session_state:
        st.session_state.prompt_selected = ""

    pdf_docs = st.file_uploader(
        "Upload your PDF Files and Click on the Submit & Process Button",
        accept_multiple_files=True
    )

    if st.button("Train & Process"):
        if pdf_docs:
            with st.spinner("Processing... (may take a minute on first run)"):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done! AI is trained on your PDF.")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
        else:
            st.warning("Please upload PDF files first.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    enter_button = st.button("Enter")

    if enter_button or st.session_state.prompt_selected:
        if st.session_state.prompt_selected:
            user_question = st.session_state.prompt_selected
            st.session_state.prompt_selected = ""

        if st.session_state.user_question != user_question:
            st.session_state.user_question = user_question
            st.session_state.output_text = ""

        if user_question:
            user_input(user_question)
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()