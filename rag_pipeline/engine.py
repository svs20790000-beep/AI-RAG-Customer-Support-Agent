import torch
import faiss
import numpy as np
import os
import logging
import streamlit as st
import yaml
import json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import pandas as pd
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv("./config/secrets.env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. Configuration & Device Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#logger.info(f"System initialized. Using device: {DEVICE}")

class RAGEngine:

        def __init__(self, config_path='./config/config.yaml'):
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            with open(config_path, 'r') as f:
                # This line creates the 'cfg' attribute
                self.cfg = yaml.safe_load(f)
            
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.path = self.cfg['paths']['knowledge_base']
            self.knowledge_texts = []
            self.faiss_index = None
            
            # Define mapping as an instance attribute to avoid NameErrors
            self.loader_mapping = {
                ".txt": TextLoader,
                ".pdf": PyPDFLoader
            }
            
            models = self.load_models(self.cfg['models'])
            self.ctx_tok, self.ctx_mod, self.q_tok, self.q_mod, self.gen_tok, self.gen_mod = models

            # Initialize Knowledge Base
            self.knowledge_texts, self.faiss_index = self.prepare_knowledge_base()


            # Placeholders for models
            #self.ctx_tok = self.ctx_mod = None
            #self.q_tok = self.q_mod = None
            #self.gen_tok = self.gen_mod = None
            logger.info(f"System initialized. Using device: {self.DEVICE}")

        @staticmethod
        @st.cache_resource
        def load_models(model_cfg): # model_cfg is the dictionary passed above
            logger.info("Loading models into memory...")
            try:
                # We need BOTH encoders for DPR to work correctly
                ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_cfg['ctx_encoder'])
                ctx_model = DPRContextEncoder.from_pretrained(model_cfg['ctx_encoder']).to(DEVICE)
                
                q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_cfg['q_encoder'])
                q_model = DPRQuestionEncoder.from_pretrained(model_cfg['q_encoder']).to(DEVICE)
                
                # Using flan-t5-large for significantly better reasoning if memory allows
                gen_tokenizer = AutoTokenizer.from_pretrained(model_cfg['gen_model'])
                gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg['gen_model']).to(DEVICE)
                
                return ctx_tokenizer, ctx_model, q_tokenizer, q_model, gen_tokenizer, gen_model
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                raise e
        
        # 2. Data Ingestion
        @st.cache_data
        def prepare_knowledge_base(_self):
            logger.info(f"Preparing knowledge base from: {_self.path}")
            if not os.path.exists(_self.path):
                os.makedirs(_self.path)
                logger.warning(f"path {_self.path} does not exist. Creating directory.")
                return [], None
            
            
            # 1. Data Layer - Load/Process texts
            processed_path = os.path.join(_self.path, "processed")
            chunked_json = os.path.join(processed_path, "chunked_documents.json")
            
            # (Assuming text loading logic from previous step is here)
            with open(chunked_json, 'r') as f:
                texts = json.load(f)

            # 2. Embeddings Layer - Initialize Manager
            from embeddings.create_embeddings import EmbeddingsManager
            _self.emb_manager = EmbeddingsManager(_self)
            
            # Try to load existing index
            index = _self.emb_manager.load_index()
            
            if index is None:
                # Generate new if not found
                index = _self.emb_manager.create_and_save(texts)
            
            
            logger.info("FAISS index built and ready.")
            return texts, index

        
        # 3. RAG Functions
        def retrieve(self, query, k=3):
            logger.info(f"Retrieving top {k} contexts for query: '{query}'")
            # Lazy initialize the retriever layer
            from retriever.retriever import Retriever
            search_layer = Retriever(self)
            return search_layer.get_relevant_chunks(query, k)

        def generate_answer(self, query):
            if not self.knowledge_texts:
                logger.error("Attempted to generate answer with empty knowledge base.")
                return "Knowledge base is empty. Please add .txt files to the folder."
            
            # 1. Retrieve (from Retriever Layer)
            context_chunks = self.retrieve(query)
            
            # 2. Generate (from LLM Layer)
            from llm.response_generator import ResponseGenerator
            generator = ResponseGenerator(self)
            
            response = generator.generate_final_response(query, context_chunks)
            return response
            
            #return self.gen_tok.decode(output[0], skip_special_tokens=True)