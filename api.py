from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
import warnings
warnings.filterwarnings('ignore')

from transformers import LlamaTokenizer

DB_FAISS_PATH = 'data/db_faiss'

# custom_prompt_template = """You are a summarization agent that summarizes an automotive issue.
# Inputs will be as follows -
#     1) Question as a JSON object with keys : make (company of the car), model (model of the car), year (year of the model launch) and issue (issue with the model).
#     2) Context (based on which the summary will be genrated)

# Context: {context}
# Question: {question}

# Output should strictly follow the following format :
# ```
# Summary:
# ```
# """

custom_prompt_template = """You are an expert in creating summaries for automotive issues based on some context documents. You will be given a json as input, generate only a very short summary.
Question : {question}
Context : {context}

Summary:"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=['context', 'question']
)

print("#### Prompt Template ready !!")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}
)

print("#### Embeddings model loaded !!")

db = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("#### FAISS Database loaded !!")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B" ,token="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").to('cuda')
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")

print("#### Model & Tokenizer loaded !!")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0, max_new_tokens = 1024)
llm = HuggingFacePipeline(pipeline=pipe)

print("#### Pipeline created !!")

# Create the sum chain once
sum_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=False,
    chain_type_kwargs={'prompt': prompt}
)



def generate_response(query):
    response = sum_chain({'query': query})
    print("#### Summary generated successfully !!")
    generated_text = response['result'].strip()
    if 'Summary:' in generated_text:
        summary = generated_text.split('Summary:')[-1].strip()
    else:
        summary = generated_text
    return summary
