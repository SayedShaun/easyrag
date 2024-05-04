from typing import Tuple
import torch
from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings



def pdf_loader(pdf_path):
        raw_texts = ""
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    raw_texts += content
        except Exception as e:
            print("An error occurred while processing the PDF:", e)
        return raw_texts
    
    
def transform_and_store(raw_texts, embedding, db_name=None):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunk_texts = text_splitter.split_text(raw_texts)
    vector_store = FAISS.from_texts(chunk_texts, embedding=embedding)
    return vector_store


class OpensourceModel:
    def __init__(
            self, 
            pdf_path:str,
            model_id:str=None, 
            embedding_model:str=None, 
            hf_token:str=None, 
            temperature:float=0.5, 
            max_token:int = 200
        )->None:
        self.__pdf_path = pdf_path
        self.__max_token = max_token
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            self.__llm, embedding = self.__huggingface_llm(
                model_id, embedding_model, temperature, max_token, hf_token)
        else:
            raise RuntimeError(
                "Please use GPU to use the open-source model or Use API Based Model(E.g. GoogleGemini, OpenAI)")
        
        raw_texts = pdf_loader(pdf_path)
        self.__vector_store = transform_and_store(raw_texts, embedding)
     

    def __huggingface_llm(self, model_id:str, embedding_model:str, temperature:float, max_token:int, hf_token:str)->Tuple:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path = model_id, 
            token=hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = model_id,
            token=hf_token,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = model_id,
            token=hf_token
        )
        hf_pipeline = pipeline(
            model = model,
            task="text-generation",
            tokenizer = tokenizer,
            return_full_text = True,
            temperature=temperature,
            max_new_tokens=max_token,
            repetition_penalty=1
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        embedding = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device":"cuda"})
        return llm, embedding
    
    def retrieve_answer(self, query:str):
        chain = ConversationalRetrievalChain.from_llm(
            self.__llm, 
            self.__vector_store.as_retriever(), 
            return_source_documents=True
        )

        chat_history = []
        answer = chain({"question": query, "chat_history": chat_history})
        
        text_store = []
        for text in answer["answer"].split("Helpful Answer: "):
            text_store.append(text)

        for x in text_store[-1].split("\n"):
            print(x)


class GoogleGemini:
    def __init__(self, pdf_path:str, google_api_key:str):
        self.__llm, embedding = self.__google_gen_ai(google_api_key)
        raw_texts = pdf_loader(pdf_path)
        self.__vector_store = transform_and_store(raw_texts, embedding)

    def __google_gen_ai(self, api_key:str)->Tuple:
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        return llm, embedding
    
    def retrieve_answer(self, query:str):
        chain = ConversationalRetrievalChain.from_llm(
            self.__llm, 
            self.__vector_store.as_retriever(), 
            return_source_documents=True
        )
        chat_history = []
        answer = chain({"question": query, "chat_history": chat_history})
        
        for text in answer["answer"].split("\n"):
            print(text)


class OpenAI:
    def __init__(self, pdf_path:str, openai_api_key:str):
        self.__llm, embedding = self.__openai(openai_api_key)
        raw_texts = pdf_loader(pdf_path)
        self.__vector_store = transform_and_store(raw_texts, embedding)

    def __openai(self, api_key:str)->Tuple:
        llm = ChatOpenAI(model="gpt-3.5-turbo", google_api_key=api_key)
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002", google_api_key=api_key)
        return llm, embedding
    
    def retrieve_answer(self, query:str):
        chain = ConversationalRetrievalChain.from_llm(
            self.__llm, 
            self.__vector_store.as_retriever(), 
            return_source_documents=True
        )
        chat_history = []
        answer = chain({"question": query, "chat_history": chat_history})
        
        for text in answer["answer"].split("\n"):
            print(text)