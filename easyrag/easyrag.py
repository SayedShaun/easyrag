
import os
import sys
import torch
from typing import Tuple
from pydantic import SecretStr
from easyrag.utils import (
    pdf_loader, 
    store_user_chat_history, 
    transform_and_store, 
    get_answer
    )
from transformers import (
    BitsAndBytesConfig, 
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
    )
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings



class RagWithHuggingFaceModel:
    def __init__(
            self,
            pdf_path: str,
            model_id: str = None,
            hf_token: SecretStr = None,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            temperature: float = 0.5,
            max_token: int = 1000
        ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            self._llm, embedding = self._huggingface_llm(
                model_id,
                embedding_model,
                temperature,
                max_token,
                hf_token
            )
        else:
            raise RuntimeError(
                """Please use GPU to use the Open-Source model 
                or Use API Based Model(E.g. GoogleGemini, OpenAI)"""
            )

        raw_texts = pdf_loader(pdf_path)
        self._vector_store = transform_and_store(raw_texts, embedding)

    def _huggingface_llm(
            self, model_id: str,
            embedding_model: str,
            temperature: float,
            max_token: int,
            hf_token: SecretStr
        ) -> Tuple:
        """
        The function `__huggingface_llm` initializes a Hugging Face model for text generation with
        specified configurations and returns the model and tokenizer.
        
        :param model_id: The `model_id` parameter in the function `__huggingface_llm` is used to specify
        the identifier or name of the pre-trained language model that you want to load for text
        generation. This could be the name of a specific model like "gpt2" or a custom model identifier
        :type model_id: str
        :param embedding_model: The `embedding_model` parameter refers to the name or identifier of the
        Hugging Face model that will be used for generating embeddings. This model will be responsible
        for converting input text into numerical representations that capture the semantic meaning of
        the text
        :type embedding_model: str
        :param temperature: The `temperature` parameter in the function `__huggingface_llm` is used in
        text generation models to control the randomness of the generated text. A higher temperature
        value will result in more diverse and creative outputs, while a lower temperature value will
        produce more conservative and predictable outputs. It essentially scales
        :type temperature: float
        :param max_token: The `max_token` parameter in the function `__huggingface_llm` specifies the
        maximum number of tokens that the text generation model will produce. This parameter controls
        the length of the generated text output
        :type max_token: int
        :param hf_token: The `hf_token` parameter in the function `__huggingface_llm` is used as a token
        for the Hugging Face model. It is typically a string that represents the specific token or model
        configuration you want to use. This token is used when loading the model and tokenizer from the
        H
        :type hf_token: str
        :return: A tuple containing an instance of the HuggingFacePipeline (`llm`) and an instance of
        the HuggingFaceEmbeddings (`embedding`) is being returned.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_id,
            token=hf_token
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            token=hf_token,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=os.getcwd()
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            token=hf_token
        )

        hf_pipeline = pipeline(
            model=model,
            task="text-generation",
            tokenizer=tokenizer,
            return_full_text=True,
            temperature=temperature,
            max_new_tokens=max_token,
            repetition_penalty=1,
            trust_remote_code=True,
            do_sample=False
        )
        
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda"}
        )
        return llm, embedding

    def retrieve_answer(self, query: str=None, continuous_chat: bool = False):
        get_answer(
            llm_model=self._llm,
            vector_store=self._vector_store.as_retriever(),
            query=query,
            continuous_chat=continuous_chat
        )
        
        #clear memory cache    
        torch.cuda.empty_cache()


class RagWithGoogleGemini:
    def __init__(
            self,
            pdf_path: str,
            google_api_key: SecretStr,
            temperature: float = 0.1,
            max_token: int = 200
        ) -> None:
        self._llm, embedding = self._google_gen_ai(
            google_api_key,
            temperature,
            max_token
        )
        raw_texts = pdf_loader(pdf_path)
        self._vector_store = transform_and_store(raw_texts, embedding)

    def _google_gen_ai(
            self,
            api_key: SecretStr,
            temperature: float,
            max_token: int
        ) -> Tuple:
        """
        The function `__google_gen_ai` initializes instances of GoogleGenerativeAI and
        GoogleGenerativeAIEmbeddings using the provided API key.
        
        :param api_key: The `api_key` parameter is a string that represents the API key required to
        access the Google Generative AI services. This key is used for authentication and authorization
        purposes when making requests to the Google Generative AI API
        :type api_key: str
        :return: A tuple containing two instances: `llm` which is an instance of the GoogleGenerativeAI
        class with the model "gemini-pro" and `embedding` which is an instance of the
        GoogleGenerativeAIEmbeddings class with the model "models/embedding-001".
        """
        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_token
        )
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        return llm, embedding

    def retrieve_answer(self, query: str=None, continuous_chat: bool = False):
        get_answer(
            llm_model=self._llm,
            vector_store=self._vector_store.as_retriever(),
            query=query,
            continuous_chat=continuous_chat
        )


class RagWithOpenAI:
    def __init__(self, pdf_path: str, openai_api_key: SecretStr, temperature: float = 0.1) -> None:
        self._llm, embedding = self._openai(openai_api_key, temperature)
        raw_texts = pdf_loader(pdf_path)
        self._vector_store = transform_and_store(raw_texts, embedding)

    def _openai(self, api_key: SecretStr, temperature: float) -> Tuple:
        """
        The function `__openai` takes an API key as input and returns instances of the ChatOpenAI and
        OpenAIEmbeddings classes initialized with the provided API key.
        
        :param api_key: The `api_key` parameter is a string that represents the API key required to
        access the OpenAI services. This key is used for authentication and authorization purposes when
        making requests to the OpenAI API
        :type api_key: str
        :return: A tuple containing two objects: a ChatOpenAI model initialized with the "gpt-3.5-turbo"
        model and a Google API key, and an OpenAIEmbeddings model initialized with the
        "text-embedding-ada-002" model and a Google API key.
        """
        llm = ChatOpenAI(model="gpt-3.5-turbo", google_api_key=api_key, temperature=temperature)
        embedding = OpenAIEmbeddings(
            model="text-embedding-ada-002", google_api_key=api_key
        )
        return llm, embedding

    def retrieve_answer(self, query: str=None, continuous_chat: bool = False):
        get_answer(
            llm_model=self._llm,
            vector_store=self._vector_store.as_retriever(),
            query=query,
            continuous_chat=continuous_chat
        )



class RagWithOllamaLLM:
    def __init__(
            self, 
            pdf_path: str, 
            model: str = "phi3", 
            embedding_model: str = "all-minilm", 
            temperature: float = 0.2
        ) -> None:
        self._model = model
        self._embedding_model = embedding_model
        self._chat_history = store_user_chat_history()
        try:
            self._llm, embedding = self._ollama_local_llm(temperature)
            raw_texts = pdf_loader(pdf_path)
            self._vector_store = transform_and_store(raw_texts, embedding)
        except Exception as e:
            print("Make sure Ollama is running on your system.")
            sys.exit(1)

    def _ollama_local_llm(self, temperature: float)->Tuple:
        llm = Ollama(model=self._model, temperature=temperature)
        embedding = OllamaEmbeddings(model=self._embedding_model)
        return llm, embedding

    def retrieve_answer(self, query: str=None, continuous_chat: bool = False):
        get_answer(
            llm_model=self._llm,
            vector_store=self._vector_store.as_retriever(),
            query=query,
            continuous_chat=continuous_chat
        )
