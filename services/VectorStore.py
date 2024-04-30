from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

k = 5
loader = PyMuPDFLoader("../DocumentStore/FRDs-170424-070159.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap = 200, 
                                                length_function=len,
                                                is_separator_regex=False,
                                                separators=[
                                                            "\n\n",
                                                            "\n",
                                                            ".",
                                                            ",",
                                                            "\u200B",  # Zero-width space
                                                            "\uff0c",  # Fullwidth comma
                                                            "\u3001",  # Ideographic comma
                                                            "\uff0e",  # Fullwidth full stop
                                                            "\u3002",  # Ideographic full stop
                                                            " ",
                                                            "",
                                                        ]
                                            )
splits = text_splitter.split_documents(pages)
embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory="../chroma_d")
retriever = vectorstore.as_retriever(search_kwargs={"k":k})



