import sys
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader



def pdf_loader(pdf_path: str)->List:
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def youtube_url_loader(url: str)->List:
    loader = YoutubeLoader.from_youtube_url(
        youtube_url=url,
        )
    return loader.load()


def transform_and_store(documents, embedding, db_name=None):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200
        )
    chunk_texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(
        chunk_texts,
        embedding=embedding
        )
    return vector_store



def store_user_chat_history()->List:
    return []


def get_answer(llm_model, vector_store, query: str=None, chat_mode: bool = False):
    chat_history = store_user_chat_history()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=vector_store,
        return_source_documents=True,
        )
    if chat_mode and query==None:
        while True:
            query = input("USER: ").strip()
            while query == None:
                query

            if query == "exit" or query == "quit" or query == "stop":
                sys.exit(0)
            if query:
                retrieve_answer = chain.invoke({"question": query, "chat_history": chat_history})
                print(f"\n\nASSISTANT: {retrieve_answer['answer']}\n\n")
    else:
        retrieve_answer = chain.invoke({"question": query, "chat_history": chat_history})
        print(f"USER: {query}\n\nASSISTANT: {retrieve_answer['answer']}\n\n")



def prompt_template():
    return PromptTemplate(
        input_variables=["question", "chat_history"],
        template="""
        The user is asking: {question}
        Previous conversation history: {chat_history}

        If the user asks about a previous conversation or question, please check the chat history and provide a polished response with the previous question. Otherwise, provide an appropriate answer to the current question.
        """
        )

