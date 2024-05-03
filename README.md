# pdfBot
A chatbot for asking questions to your pdf files/books.

The main code is present in LangchainRAG -> PyMuPDFRAG.ipynb
The code uses an api key from together.ai which you can find in in their website for free initially. (They give you $25 free credits.) 
Please put your own pdfs in DocumentStore. Change the prompt of the code according to your use case. You can also change the model to be used according to your preferences.

For document loading, used PyMuPDF (or fitz). This library gives a great metadata section and can handle UTF-8 based characters easily. Tested the code with PyPDF earlier, but on some text it did not pass through the pdf at all (due to invalid starting character). For more detailed and unstrcutured type documents can use pdfminer or the unstrtured library.

Please create a .env file for API keys and tokens. You will need your HF key, AstraDb key and endpoint and the together API inferenceing key.

Part 2: 
Used vector database integration. Compared MilvusDb and AstraDb.

Milvus: 
    1. Milvus is an open source vector db. This one can be hosted on cloud or as a local standalone on docker.
    2 .To install milvus: wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
    3. Then run it on docker as docker compose up -d.
    4. For vector store: 
    vectorstore = Milvus.from_documents(
                                        splits,
                                        embedding_function,
                                        connection_args={"host": "127.0.0.1", "port": "19530"},
                                        collection_name = "part2_langchain", ## custom collection name 
                                        search_params = {"metric":"IP","offset":0}, ## search params
                                        )
    5. The speed on milvus will depend on your host pc, if hosted on docker. Else Zilliz provides cloud based storage.

AstraDb:
    1. Needs an AstraDb account on DataStax. 
    2. ASTRA_DB_APPLICATION_TOKEN="TOKEN"
        ASTRA_DB_API_ENDPOINT="API_ENDPOINT"
        ASTRA_DB_KEYSPACE="default_keyspace"
    3. Sample code for db creation/connection:
        ASTRA_DB_API_ENDPOINT = "Use your own API Endpoint"
        ASTRA_DB_APPLICATION_TOKEN = "Use your own application token"

        desired_namespace = input("(optional) Namespace = ")
        if desired_namespace:
            ASTRA_DB_KEYSPACE = desired_namespace
        else:
            ASTRA_DB_KEYSPACE = None

        vectorstore = AstraDBVectorStore(
            embedding=embedding_function,
            collection_name="astra_vector_demo",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )
    4. The db generation is faster, but for some reason the retrieval does not giv answers relevant to the context provided

If you want to use unstructured as your document parser, the second ipynb program is there, althoguh the requirements to run that program are very high. If you want further information on that, I will upload and refine that jupyter file later.
