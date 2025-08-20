import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()


class RAGManager:
    # --- 2. The Constructor (__init__) ---
    def __init__(self, csv_path: str, db_path: str = "./chroma_db"):
        """
        Initializes the RAGManager.
        - Sets up the OpenAIEmbeddings model.
        - Checks if the database at db_path exists.
        - If not, it calls the private _ingest_from_csv method.
        - If it does, it loads the existing database.
        """
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.db_path = db_path
        self.csv_path = csv_path

        # Hint: Use os.path.exists(self.db_path) to check for the database
        if not os.path.exists(self.db_path):
            print("Database not found. Ingesting from CSV...")
            # Call your ingestion function here
            self._ingest_from_csv()
        else:
            print("Loading existing database.")
            # Load the Chroma database from the persistent directory
            self.vector_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings_model
            )

    # --- 3. The Ingestion Method (_ingest_from_csv) ---
    def _ingest_from_csv(self):
        """
        (Private method) Reads the CSV, creates embeddings, and saves to ChromaDB.
        This should only run once.
        """
        # Hint: Use pandas.read_csv(self.csv_path) to load your KB file
        df = pd.read_csv(self.csv_path)

        documents = []
        # Loop through the pandas DataFrame rows
        for index, row in df.iterrows():
            # For each row, create a LangChain 'Document' object.
            # The page_content should be the text from the 'question_text' column.
            # The metadata should be a dictionary containing the answer,
            # like: {'answer': row['answer_text']}
            doc = Document(page_content=row['question_text'],metadata={'answer': row['answer_text'],'faq_id': row['faq_id'],'category': row['category']})
            documents.append(doc)

        # After the loop, create the Chroma database from the documents.
        # This will automatically calculate embeddings and store them.
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings_model,
            persist_directory=self.db_path
        )
        print("Database ingestion complete.")

    # --- 4. The Search Method (search) ---
    def search(self, query: str) -> str | None:
        """
        Searches the vector database for a given query.
        Returns the answer text if a relevant document is found, otherwise None.
        """
        # Use the self.vector_db.similarity_search() method.
        # It takes the query and returns a list of matching documents.
        # Let's just look at the top result for now.
        results = self.vector_db.similarity_search(query)

        if results:
            # The result is a Document object. The answer is in its metadata.
            retrieved_answer = results[0].metadata['answer']
            return retrieved_answer
        else:
            return None

if __name__ == "__main__":

    # Make sure this path matches your file structure exactly
    csv_file_path = "/home/pranavnamburi/Desktop/pipecat-demo/lead_agent/data/rag_source/kb_unresponsive.csv" 

    rag_manager = RAGManager(csv_file_path)

    # Test with a real query
    test_query = "Monthly kitna earning hota hai?"
    result = rag_manager.search(test_query)

    print(f"Query: {test_query}")
    print(f"Retrieved Answer: {result}")