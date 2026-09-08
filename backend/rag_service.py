import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# Using Google Generative AI Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from database import get_db_connection

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Configure Gemini API Key
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

class RAGService:
    def __init__(self):
        self.vector_store_path = os.path.join(BASE_DIR, "faiss_index")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        self.vector_store = None
        self.chain = None
        self._initialize_rag()

    def _initialize_rag(self):
        # Check if vector store already exists
        if os.path.exists(self.vector_store_path):
            try:
                print("Loading existing FAISS vector store...")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True # required for local faiss loading
                )
            except Exception as e:
                print(f"Error loading FAISS store: {e}, recreating...")
                self._create_vector_store()
        else:
            print("Creating new FAISS vector store...")
            self._create_vector_store()
            
        self._setup_chain()

    def _create_vector_store(self):
        # We will load the project summary as the primary knowledge base
        knowledge_doc_path = os.path.join(PROJECT_DIR, "project_summary.txt")
        
        if not os.path.exists(knowledge_doc_path):
            print(f"Warning: Knowledge document not found at {knowledge_doc_path}")
            # Create an empty vector store as a fallback so it doesn't crash
            self.vector_store = FAISS.from_texts(["AgriTwin Digital Twin initialized without specific knowledge base."], self.embeddings)
            return

        loader = TextLoader(knowledge_doc_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)
        print(f"Vector store created and saved to {self.vector_store_path}")

    def _setup_chain(self):
        if not self.vector_store:
            return

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        self.system_prompt = (
            "You are an expert agricultural advisor acting as the brain of the AgriTwin Digital Twin. "
            "Your primary goal is to provide ACTIONABLE FARMING ADVICE based on the data you receive. "
            "DO NOT just explain how the software or AI models work unless explicitly asked. "
            "Instead, USE the Live Application Data (Live Sensor Telemetry and Market Prices) to answer questions like 'What should I plant?', 'Is my soil healthy?', or 'Where should I sell?'. "
            "If the user asks for crop recommendations, analyze the provided Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall to logically suggest the best crops using your general agricultural knowledge. "
            "CRITICAL INSTRUCTION: DO NOT mention or suggest that you can detect Plant Diseases. The plant disease feature has been removed from this system. Do not list it as an available model. "
            "Always be helpful, practical, and conversational with the farmer.\n\n"
            "--- KNOWLEDGE BASE CONTEXT (Project Background) ---\n{context}\n\n"
            "--- LIVE APPLICATION DATA (Current Farm State) ---\n{live_data}"
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])

    def _fetch_live_context(self) -> str:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get latest telemetry
            cursor.execute("SELECT * FROM telemetry_data ORDER BY id DESC LIMIT 1")
            latest_telemetry = cursor.fetchone()
            
            # Get latest 3 market prices
            cursor.execute("SELECT commodity, market, modal_price, arrival_date FROM market_data ORDER BY id DESC LIMIT 3")
            latest_market = cursor.fetchall()
            
            conn.close()
            
            live_context = ""
            if latest_telemetry:
                live_context += f"Latest Sensor Telemetry (Timestamp: {latest_telemetry['timestamp']}):\n"
                live_context += f"- Temperature: {latest_telemetry['temperature']} °C\n"
                live_context += f"- Humidity: {latest_telemetry['humidity']}%\n"
                live_context += f"- Soil Moisture: {latest_telemetry['soil_moisture']}%\n"
                live_context += f"- Rainfall: {latest_telemetry['rainfall']} mm\n"
                live_context += f"- Soil Nutrients: Nitrogen(N)={latest_telemetry['N']}, Phosphorus(P)={latest_telemetry['P']}, Potassium(K)={latest_telemetry['K']}\n"
                live_context += f"- Soil pH: {latest_telemetry['ph']}\n\n"
            
            if latest_market:
                live_context += "Latest Market Prices:\n"
                for row in latest_market:
                    live_context += f"- {row['commodity']} at {row['market']} market: ₹{row['modal_price']} (Date: {row['arrival_date']})\n"
                    
            if not live_context:
                return "No live data available in the database currently."
                
            return live_context
        except Exception as e:
            print(f"Failed to fetch live context: {e}")
            return "Live application data is temporarily unavailable."

    def query(self, user_message: str) -> str:
        if not hasattr(self, 'retriever') or not self.retriever:
            return "I'm sorry, my knowledge base is currently unavailable."
            
        try:
            # 1. Retrieve relevant documents
            docs = self.retriever.invoke(user_message)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Fetch Live DB Data
            live_data_text = self._fetch_live_context()
            
            # 3. Format the prompt
            messages = self.prompt_template.invoke({
                "context": context_text,
                "live_data": live_data_text,
                "input": user_message
            })
            
            # 4. Call the LLM directly
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"Error during RAG query: {e}")
            return "An error occurred while processing your request. Please try again later."
