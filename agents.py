"""
DataSmith AI - GenAI Intern Assignment
Multi-Agent Medical AI System - Core Agent Implementation
"""

import os
import json
import sqlite3
from typing import Dict, List, Optional, Annotated
from dotenv import load_dotenv
import requests
import logging
import chromadb
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
import PyPDF2

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agents_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State management for the LangGraph multi-agent workflow"""
    messages: Annotated[List, add_messages]
    patient_name: Optional[str]
    patient_data: Optional[Dict]
    current_agent: str
    route_to_clinical: bool

class MedicalAgentSystem:
    """Core multi-agent system for post-discharge medical care"""

    def __init__(self):
        """Initialize the complete medical agent system"""
        logger.info("Initializing Medical Agent System...")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )

        self.chroma_client = None
        self.nephrology_collection = None
        self.graph = None

        self.init_db()
        self.init_vector_db()
        self.graph = self.build_workflow()

        logger.info("Medical Agent System initialized")

    def init_db(self):
        """Initialize SQLite database with patient discharge data"""
        try:
            conn = sqlite3.connect('patients.db')
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT NOT NULL,
                    discharge_date TEXT NOT NULL,
                    primary_diagnosis TEXT NOT NULL,
                    medications TEXT NOT NULL,
                    dietary_restrictions TEXT,
                    follow_up TEXT,
                    warning_signs TEXT,
                    discharge_instructions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_patient_name
                ON patients(patient_name)
            ''')

            with open('data/patients.json', 'r', encoding='utf-8') as f:
                patients = json.load(f)

            cursor.execute('DELETE FROM patients')

            for patient in patients:
                cursor.execute('''
                    INSERT INTO patients (patient_name, discharge_date, primary_diagnosis,
                                        medications, dietary_restrictions, follow_up,
                                        warning_signs, discharge_instructions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patient['patient_name'],
                    patient['discharge_date'],
                    patient['primary_diagnosis'],
                    json.dumps(patient['medications']),
                    patient['dietary_restrictions'],
                    patient['follow_up'],
                    patient['warning_signs'],
                    patient['discharge_instructions']
                ))

            conn.commit()
            conn.close()
            logger.info(f"Database initialized with {len(patients)} patient records")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def init_vector_db(self):
        """Initialize ChromaDB Vector Database for RAG implementation"""
        try:
            if self.chroma_client is None:
                self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

            self.nephrology_collection = self.chroma_client.get_or_create_collection(
                name="nephrology_knowledge",
                metadata={"description": "Nephrology knowledge base for RAG"}
            )

            current_count = self.nephrology_collection.count()

            if current_count >= 0:  # Always reload
                self.chroma_client.delete_collection("nephrology_knowledge")
                self.nephrology_collection = self.chroma_client.get_or_create_collection(
                    name="nephrology_knowledge",
                    metadata={"description": "Nephrology knowledge base for RAG"}
                )

                pdf_path = "data/comprehensive-clinical-nephrology.pdf"
                if os.path.exists(pdf_path):
                    docs = self.load_pdf_content(pdf_path)

                    if docs:
                        self.nephrology_collection.add(
                            documents=docs,
                            ids=[f"nephrology_pdf_{i}" for i in range(len(docs))],
                            metadatas=[{
                                "source": "Comprehensive Clinical Nephrology",
                                "doc_id": i,
                                "category": "medical_literature"
                            } for i in range(len(docs))]
                        )
                        logger.info(f"Loaded {len(docs)} chunks from PDF")
                    else:
                        # Fallback knowledge
                        docs = [
                            "Chronic Kidney Disease (CKD) is a progressive loss of kidney function over time. Stages range from 1-5 based on GFR.",
                            "Common medications for CKD include ACE inhibitors, ARBs, diuretics, and phosphate binders.",
                            "Dietary restrictions for CKD include limiting sodium, phosphorus, potassium, and protein.",
                            "Warning signs include swelling, shortness of breath, fatigue, and changes in urination."
                        ]
                        self.nephrology_collection.add(
                            documents=docs,
                            ids=[f"nephrology_fallback_{i}" for i in range(len(docs))],
                            metadatas=[{
                                "source": "Nephrology Guidelines",
                                "doc_id": i,
                                "category": "medical_knowledge"
                            } for i in range(len(docs))]
                        )
                else:
                    raise FileNotFoundError(f"Nephrology PDF not found: {pdf_path}")

        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            self.nephrology_collection = None
            raise

    def load_pdf_content(self, pdf_path: str) -> List[str]:
        """Load and process PDF content for RAG knowledge base"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            cleaned_text = page_text.replace('\n', ' ').strip()
                            if len(cleaned_text) > 100:
                                text_content.append(cleaned_text)
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {str(e)}")
                        continue

                logger.info(f"Extracted {len(text_content)} text chunks from PDF")
                return text_content

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            return []

    def get_patient_data(self, patient_name: str) -> Optional[Dict]:
        """Patient Data Retrieval Tool - Core requirement from assignment"""
        try:
            conn = sqlite3.connect('patients.db')
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM patients
                WHERE LOWER(patient_name) LIKE LOWER(?)
                ORDER BY patient_name
                LIMIT 1
            ''', (f'%{patient_name}%',))

            result = cursor.fetchone()
            conn.close()

            if result:
                patient_data = {
                    'patient_name': result[1],
                    'discharge_date': result[2],
                    'primary_diagnosis': result[3],
                    'medications': json.loads(result[4]),
                    'dietary_restrictions': result[5],
                    'follow_up': result[6],
                    'warning_signs': result[7],
                    'discharge_instructions': result[8]
                }
                logger.info(f"Patient data retrieved for {patient_data['patient_name']}")
                return patient_data
            else:
                logger.warning(f"No patient found matching: {patient_name}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving patient data for {patient_name}: {str(e)}")
            return None

    def search_nephrology_docs(self, query: str, k: int = 3) -> List[str]:
        """RAG Implementation - Semantic search through nephrology knowledge"""
        if not self.nephrology_collection:
            self.init_vector_db()

        if not self.nephrology_collection:
            return []

        try:
            results = self.nephrology_collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas']
            )

            if results and results['documents'] and results['documents'][0]:
                retrieved_docs = results['documents'][0]
                logger.info(f"RAG retrieved {len(retrieved_docs)} documents")
                return retrieved_docs
            else:
                return []

        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}")
            return []

    def web_search(self, query: str) -> str:
        """Web Search Tool - Fallback for queries outside reference materials"""
        try:
            enhanced_query = query + " nephrology kidney disease medical research"

            params = {
                "q": enhanced_query,
                "api_key": os.getenv("SERPAPI_API_KEY"),
                "num": 3,
                "engine": "google",
                "hl": "en",
                "gl": "us"
            }

            response = requests.get("https://serpapi.com/search", params=params, timeout=10)

            if response.status_code == 200:
                results = response.json()
                if "organic_results" in results and results["organic_results"]:
                    summaries = []
                    for i, result in enumerate(results["organic_results"][:3], 1):
                        title = result.get('title', 'N/A')
                        snippet = result.get('snippet', 'N/A')
                        link = result.get('link', 'N/A')
                        summaries.append(f"Source {i}: {title}\n{snippet}\nURL: {link}")

                    web_results = "\n\n".join(summaries)
                    logger.info("Web search completed")
                    return web_results
                else:
                    return "No relevant web search results found."
            else:
                return "Web search temporarily unavailable."

        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return "Web search unavailable due to technical issues."

    def receptionist_node(self, state: AgentState) -> AgentState:
        """Receptionist Agent Node - First point of contact in multi-agent workflow"""
        last_message = state["messages"][-1].content if state["messages"] else ""

        # Patient Identification
        if not state["patient_name"] and state["messages"]:
            words = last_message.split()
            if len(words) >= 2 and words[0].lower() not in ['hello', 'hi', 'hey', 'good']:
                potential_name = ' '.join(words[:2])
                if any(c.isupper() for c in potential_name):
                    state["patient_name"] = potential_name

        if not state["patient_name"]:
            response = "Hello! I'm your post-discharge care assistant. What's your name?"
            state["messages"].append(AIMessage(content=response))
            return state

        # Patient Data Retrieval Tool
        patient_data = self.get_patient_data(state["patient_name"])
        if not patient_data:
            response = f"I couldn't find a discharge report for {state['patient_name']}. Could you please check the spelling of your name?"
            state["messages"].append(AIMessage(content=response))
            return state

        state["patient_data"] = patient_data

        # Medical Query Detection and Routing
        medical_keywords = [
            'pain', 'swelling', 'medication', 'symptoms', 'side effects', 'dosage',
            'treatment', 'worried', 'concern', 'problem', 'hurt', 'ache', 'tired',
            'weak', 'blood', 'pressure', 'feeling', 'sick', 'nausea', 'dizzy',
            'research', 'study', 'latest', 'recent', 'SGLT2', 'sglt2', 'inhibitor',
            'drug', 'therapy', 'clinical', 'medical', 'advice', 'help', 'food', 'foods',
            'diet', 'eat', 'eating', 'avoid', 'nutrition', 'sodium', 'protein', 'kidney'
        ]

        if any(keyword in last_message.lower() for keyword in medical_keywords):
            response = f"Hi {patient_data['patient_name']}! I found your discharge report from {patient_data['discharge_date']} for {patient_data['primary_diagnosis']}. This sounds like a medical concern. Let me connect you with our Clinical AI Agent."
            state["messages"].append(AIMessage(content=response))
            state["route_to_clinical"] = True
            state["current_agent"] = "clinical"
            return state

        # General Conversation Handling
        try:
            prompt = f"You are a medical receptionist. Patient {patient_data['patient_name']} was discharged on {patient_data['discharge_date']} for {patient_data['primary_diagnosis']}. They said: '{last_message}'. Respond warmly in 1-2 sentences."
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            state["messages"].append(AIMessage(content=response))
        except Exception as e:
            logger.error(f"LLM error in Receptionist Agent: {str(e)}")
            response = f"Hi {patient_data['patient_name']}! I found your discharge report. How can I help you today?"
            state["messages"].append(AIMessage(content=response))

        return state

    def clinical_node(self, state: AgentState) -> AgentState:
        """Clinical AI Agent Node - Specialized medical advice with RAG and web search"""
        # Get the actual user query (last human message)
        user_query = ""
        for msg in reversed(state["messages"]):
            if str(type(msg)).find('HumanMessage') != -1:
                user_query = msg.content
                break

        if not user_query:
            user_query = state["messages"][-1].content if state["messages"] else ""

        intro = "Hello! I'm the Clinical AI Agent. " if state["route_to_clinical"] else ""

        # RAG Implementation - Search nephrology knowledge base
        rag_results = self.search_nephrology_docs(user_query)
        sources = []
        context = ""

        if rag_results:
            context = "\n".join(rag_results)
            sources.append("Nephrology Guidelines")

        # Web Search Logic - Only when RAG insufficient or for recent research
        research_keywords = ['latest', 'recent', 'new', 'research', 'study', '2024', '2023']
        needs_web_search = False

        if any(keyword in user_query.lower() for keyword in research_keywords):
            needs_web_search = True
        elif not rag_results:
            needs_web_search = True

        if needs_web_search:
            web_results = self.web_search(user_query)

            if not rag_results:
                intro += "Let me search for recent information... "
            else:
                intro += "Let me also check for recent research... "

            context += f"\n\nWeb Search Results:\n{web_results}"
            sources.append("Recent Medical Literature")

        # Generate Evidence-Based Medical Response
        prompt = f"""You are a Clinical AI Assistant for CKD patients.

Patient: {state.get('patient_data', {}).get('patient_name', 'Patient')}
Question: {user_query}

Context: {context[:500] if context else 'General CKD management'}

Provide a brief medical response (2-3 sentences). Be helpful and direct."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            final_response = intro + response

            if sources:
                final_response += f"\n\n**Sources:** {', '.join(sources)}"

            state["messages"].append(AIMessage(content=final_response))

        except Exception as e:
            logger.error(f"Error generating clinical response: {str(e)}")
            response = "I apologize, but I'm experiencing technical difficulties. Please consult your healthcare provider for medical advice."
            state["messages"].append(AIMessage(content=response))

        state["route_to_clinical"] = False
        return state

    def should_route_to_clinical(self, state: AgentState) -> str:
        """Agent Routing Decision Function for LangGraph workflow"""
        if state["route_to_clinical"]:
            return "clinical"
        return "receptionist"

    def build_workflow(self):
        """Build LangGraph Multi-Agent Workflow"""
        workflow = StateGraph(AgentState)

        workflow.add_node("receptionist", self.receptionist_node)
        workflow.add_node("clinical", self.clinical_node)

        workflow.set_entry_point("receptionist")

        workflow.add_conditional_edges(
            "receptionist",
            self.should_route_to_clinical,
            {
                "receptionist": END,
                "clinical": "clinical"
            }
        )

        workflow.add_edge("clinical", END)

        compiled_graph = workflow.compile()
        logger.info("LangGraph workflow compiled")
        return compiled_graph

    def create_new_session(self, patient_name: Optional[str] = None):
        """Create new patient session state"""
        return {
            "messages": [],
            "patient_name": patient_name,
            "patient_data": None,
            "current_agent": "receptionist",
            "route_to_clinical": False
        }

    def process_message(self, state: Dict, message: str):
        """Process patient message through LangGraph multi-agent workflow"""
        try:
            state["messages"].append(HumanMessage(content=message))
            result_state = self.graph.invoke(state)

            last_ai_message = None
            agent_name = "Receptionist Agent"
            sources = []

            for msg in reversed(result_state["messages"]):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg.content

                    if "**Sources:**" in last_ai_message:
                        parts = last_ai_message.split("**Sources:**")
                        last_ai_message = parts[0].strip()
                        sources = [s.strip() for s in parts[1].split(",")]

                    if "Clinical AI Agent" in last_ai_message or result_state.get("current_agent") == "clinical":
                        agent_name = "Clinical AI Agent"

                    break

            return {
                "state": result_state,
                "response": last_ai_message or "I apologize, but I couldn't process your request.",
                "agent": agent_name,
                "sources": sources if sources else None
            }

        except Exception as e:
            logger.error(f"Error in LangGraph workflow: {str(e)}")
            return {
                "state": state,
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
                "agent": "System Error",
                "sources": None
            }
