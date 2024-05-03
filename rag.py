import torch
from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


class RAG:
    def __init__(
            self,
            model_id, 
            embedding_model, 
            hf_token, 
            pdf_path,
            temperature:float=0.5,
            max_token:int = 200, 
        ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.hf_token = hf_token
        self.pdf_path = pdf_path
        self.max_token = max_token
        self.vector_store = self._transform_and_store()
        self.llm = self._huggingface_llm()

    def _huggingface_llm(self):
        """
        The function `_huggingface_llm` initializes a Hugging Face pipeline for text generation using a
        specified model and token.
        :return: The function `_huggingface_llm` returns an instance of the `HuggingFacePipeline` class
        initialized with a pipeline for text generation using a pre-trained language model from Hugging
        Face.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_id, 
            token=self.hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = self.model_id,
            token=self.hf_token,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            token=self.hf_token
        )
        hf_pipeline = pipeline(
            model = model,
            task="text-generation",
            tokenizer = tokenizer,
            return_full_text = True,
            temperature=0.5,
            max_new_tokens=self.max_token,
            repetition_penalty=1
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        return llm

    def _pdf_loader(self, pdf_path):
        """
        The function `_pdf_loader` reads text content from a PDF file using the `PdfReader` class and
        returns the concatenated text from all pages.
        
        :param pdf_path: The `pdf_path` parameter in the `_pdf_loader` function is a string that
        represents the file path to the PDF file that you want to load and extract text from. This
        function reads the PDF file using the `PdfReader` class and extracts text content from each page
        in the PDF file
        :return: The `_pdf_loader` function returns a string containing the extracted text from all
        pages of the PDF located at the specified `pdf_path`.
        """
        raw_texts = ""
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_texts += content
        return raw_texts
    
    def _weblink_loader(self):
        pass

    def _transform_and_store(self):
        """
        The function `_transform_and_store` processes text data from a PDF file, splits it into chunks,
        embeds the chunks using a Hugging Face model on a CUDA device, and stores the resulting vectors
        using FAISS.
        :return: The `_transform_and_store` method returns a vector store created using the
        HuggingFaceEmbeddings model with specified model name and model arguments, along with text
        splitting and chunking using RecursiveCharacterTextSplitter. The method then creates a vector
        store using FAISS from the chunked texts and returns this vector store.
        """
        embedding = HuggingFaceEmbeddings(
            model_name=self.embedding_model, 
            model_kwargs={"device":"cuda"}
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20
        )
        chunk_texts = text_splitter.split_text(
            self._pdf_loader(self.pdf_path)
        )
        vector_store = FAISS.from_texts(
            chunk_texts, embedding=embedding
        )
        return vector_store
    
    def retrieve_answer(self, query):
        """
        The function retrieves and prints the helpful answer from a conversational retrieval chain based
        on a given query.
        
        :param query: The code you provided seems to be a method for retrieving an answer using a
        conversational retrieval chain. It splits the answer text by "Helpful Answer: " and then prints
        each line of the final part of the answer
        """
        chain = ConversationalRetrievalChain.from_llm(
            self.llm, 
            self.vector_store.as_retriever(), 
            return_source_documents=True
        )

        chat_history = []
        answer = chain({"question": query, "chat_history": chat_history})
        
        text_store = []
        for text in answer["answer"].split("Helpful Answer: "):
            text_store.append(text)

        for x in text_store[-1].split("\n"):
            print(x)

