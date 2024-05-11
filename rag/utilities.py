import sys
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader



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
    :param embedding: The `embedding` parameter in the `transform_and_store` function likely refers to a
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