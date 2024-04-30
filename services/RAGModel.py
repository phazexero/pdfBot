from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Together
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

k = 5

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(input_variables=['context', 'question'],
                        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep answers descriptive and mention the process.\nQuestion: {question} \nContext: {context} \nAnswer:")

embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma(persist_directory="../chroma_d", embedding_function=embedding_function)

retriever = vectorstore.as_retriever(search_kwargs={"k":k})
response = Together(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        together_api_key= "dc50d15b1bed2012efe79427bafd87b100c9bf0d20609c832c485aec433ddd7d",
        temperature=0.3,
        max_tokens=512
    )

rag_chain = (
    {"context": retriever 
    | format_docs, "question": RunnablePassthrough()}
    | prompt
    | response
    | StrOutputParser())