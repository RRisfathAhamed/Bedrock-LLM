import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import get_embedding

VECTOR_DB_PATH = "database"

def populate_dbcv(texts):
    documents = [Document(page_content=text) for text in texts]
    chunks = split_documents(documents)
    return add_to_vector_db(chunks)

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(documents)
    return documents

def add_to_vector_db(chunks: list[Document]):
    try:
        vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH, embedding_function=get_embedding.get_embedding_function()
        )
        chunks_with_ids = calculate_chunk_ids(chunks)
        existing_items = vector_db.get(include=[])
        existing_ids = set(existing_items["ids"])
        add_db_message = f"Number of existing documents in DB: {len(existing_ids)}"

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            added_db_message = f"Adding new documents: {len(new_chunks)}"
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            vector_db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            added_db_message = "No new documents to add"

        response_messages = [add_db_message, added_db_message]
        return response_messages
    except Exception as e:
        raise

def calculate_chunk_ids(chunks):
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"chunk_{i}"
    return chunks

def clear_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
