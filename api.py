from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
import warnings
warnings.filterwarnings('ignore')

# from transformers import LlamaTokenizer

DB_FAISS_PATH = 'data/db_faiss'

custom_prompt_template = """You are an expert in generating short summary for automotive issues.
Based on the issue given in input, you will be provided with a context text that needs to summarized and given as output.
Generate a very short abstractive summary of the context. 
Context : {context}
Question : {question}
Summary:"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=['context', 'question']
)

# print("#### Prompt Template ready !!")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}
)

# print("#### Embeddings model loaded !!")

db = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# print("#### FAISS Database loaded !!")

model = AutoModelForCausalLM.from_pretrained("neuralmagic/Meta-Llama-3.1-8B-FP8").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("neuralmagic/Meta-Llama-3.1-8B-FP8")
# tokenizer = LlamaTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF")

# print("#### Model & Tokenizer loaded !!")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0, max_new_tokens = 1024)
llm = HuggingFacePipeline(pipeline=pipe)

# print("#### Pipeline created !!")


sum_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)


def generate_response(query):
    response = sum_chain({'query': query})

    # print("#### Summary generated successfully !!")
    generated_text = response['result'].strip()
    if 'Summary:' in generated_text:
        summary = generated_text.split('Summary:')[-1].strip()
    else:
        summary = generated_text

    if 'Context : ':
        text = generated_text.split('Context :')[-1].strip()
        retrieved_docs = text.split('Question :')[0].strip()
    else:
        retrieved_docs = ""

    return summary, retrieved_docs
