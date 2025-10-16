import json
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def load_config():
    """Load configuration from config.json"""
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def setup_rag_system(config):
    """Setup RAG system with configuration"""
    # Load knowledge file
    knowledge_file = config["data"]["knowledge_file"]
    if not os.path.exists(knowledge_file):
        print(f"Warning: {knowledge_file} tidak ditemukan. Chatbot akan bekerja tanpa knowledge base.")
        return None, None
    
    with open(knowledge_file, "r", encoding="utf-8") as f:
        data = f.read()
    
    # Split data into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["rag"]["chunk_size"], 
        chunk_overlap=config["rag"]["chunk_overlap"]
    )
    docs = splitter.create_documents([data])
    
    # Generate embeddings & create vector store
    embeddings = OpenAIEmbeddings(
        model=config["openai"]["embedding_model"],
        openai_api_key=config["openai"]["api_key"]
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": config["rag"]["similarity_search_k"]}
    )
    
    # Setup LLM
    llm = ChatOpenAI(
        model=config["openai"]["model"], 
        temperature=config["openai"]["temperature"], 
        openai_api_key=config["openai"]["api_key"]
    )
    
    # Create RAG chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    return qa, llm

def main():
    """Main interactive Q&A session"""
    print("[Bot] Chatbot RAG - Sesi Tanya Jawab")
    print("=" * 40)
    
    try:
        # Load configuration
        config = load_config()
        print("[OK] Konfigurasi berhasil dimuat")
        
        # Setup RAG system
        qa, llm = setup_rag_system(config)
        
        if qa:
            print("[OK] RAG system berhasil diinisialisasi")
        else:
            print("[Warning] RAG system tidak tersedia, menggunakan LLM saja")
            # Setup LLM only if RAG fails
            llm = ChatOpenAI(
                model=config["openai"]["model"], 
                temperature=config["openai"]["temperature"], 
                openai_api_key=config["openai"]["api_key"]
            )
        
        print("\nKetik pertanyaan Anda (ketik 'quit' atau 'exit' untuk keluar):")
        print("-" * 40)
        
        while True:
            # Get user input
            query = input("\n[Anda]: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'keluar']:
                print("\n[Bye] Terima kasih! Sampai jumpa!")
                break
            
            # Skip empty queries
            if not query:
                continue
            
            try:
                # Get response
                if qa:
                    # Use RAG if available
                    result = qa.run(query)
                else:
                    # Use LLM only
                    result = llm.predict(query)
                
                print(f"\n[Bot]: {result}")
                
            except Exception as e:
                print(f"\n[Error]: {str(e)}")
                
    except FileNotFoundError:
        print("[Error] config.json tidak ditemukan!")
    except json.JSONDecodeError:
        print("[Error] Format config.json tidak valid!")
    except Exception as e:
        print(f"[Error]: {str(e)}")

if __name__ == "__main__":
    main()