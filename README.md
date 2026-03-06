## Overview of the App
The objective of this project is to design and implement an AI-powered customer support agent using Retrieval-Augmented Generation (RAG) that retrieves accurate information from a curated knowledge base before generating responses. The system ensures grounded, explainable, and reliable answers, significantly reducing human agent workload while improving response accuracy and customer satisfaction.

## Files Description
1.requirements.txt - Requirements file generated using [pipreqs]
pip install pipreqs
pipreqs

2.content\knowledge_base folder - This folder contains txt files.  

3.genaimaincustsupport.py - core programming using Retrieval-Augmented Generation (RAG)
#import the library
torch
DPRQuestionEncoder,DPRQuestionEncoderTokenizer,DPRContextEncoder,DPRContextEncoderTokenizer (from transformers)
faiss
numpy
streamlit
os
DirectoryLoader (from langchain_community.document_loaders)
RecursiveCharacterTextSplitter (from langchain_text_splitters)

#load the huggingface model
context_encoder=DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer=DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
rag_model=AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
rag_tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base')

#load the file and get content
loader = DirectoryLoader('./content/knowledge_base/', glob="**/*.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)
content = docs[0].page_content
doc_embeddings=[]

#embedding
inputs=context_tokenizer(content,return_tensors='pt',padding=True)
embedding=context_encoder(**inputs).pooler_output.detach().numpy()
doc_embeddings.append(embedding)
doc_embeddings = np.array(doc_embeddings)
doc_embeddings.shape[1]
doc_embeddings=np.vstack(doc_embeddings)

#create faiss index for fast retrieval
dimension=doc_embeddings.shape[1]
faiss_index=faiss.IndexFlatL2(dimension)
faiss_index.add(doc_embeddings)

#query processing and retrieval process return top 2 search result
def retrieve_top_k(query,k=2):
  query_inputs=context_tokenizer(query,return_tensors='pt')
  query_embeddings=context_encoder(**query_inputs).pooler_output.detach().numpy()
  distances,indices=faiss_index.search(query_embeddings,k)
  retrieved_docs=[content[i] for i in indices[0]]

#generate the response using RAG
def generate_response(query):
  retrieved_docs=retrieve_top_k(query,k=2)
  context=" ".join(retrieved_docs)
  inputs=rag_tokenizer(f"Question:{query} context:{context}",return_tensors='pt')
  output=rag_model.generate(**inputs)
  response=rag_tokenizer.decode(output[0],skip_special_tokens=True)

#question answer bot using command line
def chat():
  print('Hi Ask me or type stop to end the conversation')
  
  while True:
    query=input('you: ')
    if query.lower()=='stop':
      print('Goodbye')
      break
    response=generate_response(query)
    print(f'GPT: {response}')

#use streamlit for question answer BOT display 
with st.form('my_form') :
  text=st.text_area('Hi Ask me..','...') 
  submitted = st.form_submit_button('Submit')
  if submitted:
    response=generate_response(text)
    st.info(response)


## Prerequisite libraries
torch
DPRQuestionEncoder,DPRQuestionEncoderTokenizer,DPRContextEncoder,DPRContextEncoderTokenizer (from transformers)
faiss
numpy
streamlit
os
DirectoryLoader (from langchain_community.document_loaders)
RecursiveCharacterTextSplitter (from langchain_text_splitters)


## Demo App


## Run it locally



"# AI-Powered-RAG-CustomerSupportAgent" 
