import sys
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain




def pdf_loader(pdf_path):
    """
    The function `pdf_loader` loads text content from a PDF file and returns it as a single string.
    
    :param pdf_path: The `pdf_loader` function you provided seems to be a Python function that loads
    text content from a PDF file using the `PyPDF2` library. It reads each page of the PDF and extracts
    the text content from it
    :return: The function `pdf_loader` returns a string containing the extracted text from the PDF file
    located at the `pdf_path` provided as input. If there is an error during the processing of the PDF
    file, it will print an error message and return an empty string.
    """
    if pdf_path is None:
        raise ValueError("Please Provide PDF file")
    
    raw_texts = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_texts += content
    except Exception as e:
        print("An error occurred while processing the PDF:", e)
        sys.exit(1)
    return raw_texts



def transform_and_store(raw_texts, embedding, db_name=None):
    """
    The function `transform_and_store` takes raw texts, splits them into chunks, converts them into
    vectors using a specified embedding, and stores the vectors in a database using FAISS.
    
    :param raw_texts: Raw_texts is the input text data that you want to transform and store. It can be a
    single text or a collection of texts that you want to process
    :param embedding: The `embedding` parameter in the `transform_and_store` function refers to a
    method of representing text data in a numerical format. Embeddings are commonly used in natural
    language processing tasks to convert words or sentences into dense vectors that capture semantic
    relationships
    :param db_name: The `db_name` parameter in the `transform_and_store` function is used to specify the
    name of the database where the vectors will be stored. If no `db_name` is provided, the vectors will
    still be stored but the database name will be default or unspecified. It is optional and can
    :return: The function `transform_and_store` returns a vector store created using the FAISS library
    from the input raw texts after splitting them into chunks and processing them with the specified
    embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    chunk_texts = text_splitter.split_text(raw_texts)
    vector_store = FAISS.from_texts(
        chunk_texts,
        embedding=embedding
    )
    return vector_store



def store_user_chat_history()->List:
    return []


def get_answer(llm_model, vector_store, query: str=None, chat: bool = False):
    """
    This Python function retrieves answers using a conversational retrieval chain based on a language
    model and vector store, with an option for continuous chat interaction.
    
    :param llm_model: The `llm_model` parameter likely refers to a language model used for generating
    responses in a conversational system. It could be a pre-trained model like GPT-3 or a custom
    language model tailored for specific tasks
    :param vector_store: The `vector_store` parameter in the `get_answer` function refers to a
    data structure or repository that stores vector representations of documents or text data. This
    could be used for semantic similarity calculations or information retrieval tasks in the
    conversational system. The specific implementation and type of `vector_store` would
    :param query: The `query` parameter in the `get_answer` function is a string that represents the
    user's input or question that they want the assistant to respond to. It is used to retrieve an
    answer from the conversational retrieval chain based on the provided query. If `continuous_chat` is
    set to `
    :type query: str
    :param continuous_chat: The `continuous_chat` parameter in the `get_answer` function is a boolean
    flag that determines whether the function should enter a continuous chat mode or not. When
    `continuous_chat` is set to `True`, the function will keep prompting the user for input and
    providing responses until the user decides to exit, defaults to False
    :type continuous_chat: bool (optional)
    """
    chat_history = store_user_chat_history()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=vector_store,
        return_source_documents=True,
    )
    if chat and query==None:
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