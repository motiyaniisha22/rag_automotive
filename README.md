## Automobile Issue Summarization using RAG (Retrieval-Augmented Generation)

Task : Leverage RAG abilities to summarize automobile issues. Relevant documents were retrieved using FAISS and were provided to quantized Llama 3.1 along with the query to generate summary.

### Steps to Run the Project
Follow the below steps to run the code or use run_script.ipynb.

#### 1. Installing required dependencies.
You can intall the dependies using the following command :
```bash pip install -r requirements.txt```

#### 2. Create the Vector Database
The next step is to create the vector database which will be further used by retriever to get the relevant documents. Run the code as follows :
```bash python create_db.py```

#### 3. Generate a Summary
The final step is to create the summary of retrieved documents. This can be performed by running the script as follows:
```bash python inference.py --query <your json query>```
