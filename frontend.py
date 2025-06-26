"""
DataSmith AI - GenAI Intern Assignment
Post Discharge Medical AI Assistant - Streamlit Frontend

This module implements the web interface for the multi-agent medical AI system using Streamlit.
It provides a simple, user-friendly chat interface for patients to interact with the
Receptionist and Clinical AI agents.

Key Features:
- Simple chat interface for patient interactions
- Patient name input and session management
- Real-time communication with FastAPI backend
- Agent identification and source citations display
- Session clearing and patient switching capabilities
- Medical disclaimers and usage instructions

Author: DataSmith AI GenAI Intern
Purpose: Interview Assignment - Multi-Agent Medical AI POC
Architecture: Streamlit Frontend + FastAPI Backend + LangGraph Agents
"""

import streamlit as st
import requests
import json
import logging

# Configure logging for frontend interactions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== STREAMLIT PAGE CONFIGURATION ====================

# Configure Streamlit page settings for the medical AI application
st.set_page_config(
    page_title="Post Discharge Medical AI Assistant",
    page_icon="🏥",
    layout="centered",  # Center the content for better readability
    initial_sidebar_state="collapsed"  # Hide sidebar for cleaner interface
)

# ==================== API CONFIGURATION ====================

# FastAPI backend endpoint configuration
API_URL = "http://localhost:8000"

# ==================== SESSION STATE INITIALIZATION ====================

# Initialize Streamlit session state for conversation management
# This maintains conversation history and patient information across page reloads
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat conversation history
    logger.info("Initialized empty message history")

if "patient_name" not in st.session_state:
    st.session_state.patient_name = None  # Current patient identification
    logger.info("Initialized patient name as None")

# ==================== UI HEADER AND BRANDING ====================

# Main application title and branding
st.title("🏥 Post Discharge Medical AI Assistant")
st.caption("DataSmith AI - Multi-Agent Medical Care System")
st.markdown("---")  # Visual separator

# ==================== PATIENT IDENTIFICATION AND CONTROLS ====================

# Patient name input and session management controls
# Layout: 3/4 width for name input, 1/4 width for clear button
col1, col2 = st.columns([3, 1])

with col1:
    # Patient name input field with current value persistence
    patient_name_input = st.text_input(
        "Enter your name:",
        value=st.session_state.patient_name or "",
        placeholder="e.g., John Smith",
        help="Enter your full name to retrieve your discharge report"
    )

with col2:
    # Add spacing for visual alignment
    st.write("")

    # Clear chat session button
    if st.button("🗑️ Clear Chat", help="Start a new conversation"):
        logger.info("User requested chat session clear")

        # Clear local session state
        st.session_state.messages = []
        st.session_state.patient_name = None

        # Clear backend session via API call
        try:
            response = requests.post(f"{API_URL}/clear-session", timeout=5)
            if response.status_code == 200:
                logger.info("Backend session cleared successfully")
            else:
                logger.warning(f"Backend session clear failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error clearing backend session: {str(e)}")
            # Continue anyway - local state is cleared

        # Trigger page rerun to refresh UI
        st.rerun()

# ==================== MEDICAL DISCLAIMER ====================

# Required medical disclaimer as specified in assignment
st.info("⚠️ This is an AI assistant for educational purposes only. Always consult healthcare professionals for medical advice.")

# Visual separator before chat interface
st.markdown("---")

# ==================== CHAT INTERFACE ====================

# Display conversation history
# Iterate through all messages in session state and render them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display message content
        st.write(message["content"])

        # Display agent information if available (for assistant messages)
        if "agent" in message:
            st.caption(f"🤖 Agent: {message['agent']}")

        # Display source citations if available (for medical responses)
        if "sources" in message and message["sources"]:
            st.caption(f"📚 Sources: {', '.join(message['sources'])}")

# ==================== MESSAGE INPUT AND PROCESSING ====================

# Chat input widget for patient messages
if prompt := st.chat_input("Type your message here...", key="chat_input"):
    logger.info(f"User submitted message: {prompt[:50]}...")

    # Add user message to conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)

    # ==================== API REQUEST PREPARATION ====================

    # Prepare payload for FastAPI backend
    payload = {
        "message": prompt,
        "patient_name": patient_name_input if patient_name_input else st.session_state.patient_name
    }

    logger.info(f"Sending API request for patient: {payload.get('patient_name', 'Unknown')}")

    # ==================== API COMMUNICATION ====================

    try:
        # Show loading spinner while processing
        with st.spinner("🤖 AI is thinking..."):
            response = requests.post(
                f"{API_URL}/chat",
                json=payload,
                timeout=60  # Increased to 60 second timeout for medical queries
            )

        # ==================== RESPONSE PROCESSING ====================

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Received response from {result['agent']}")

            # Update patient name in session state if newly provided
            if patient_name_input and not st.session_state.patient_name:
                st.session_state.patient_name = patient_name_input
                logger.info(f"Updated session patient name: {patient_name_input}")

            # Prepare assistant message with metadata
            assistant_message = {
                "role": "assistant",
                "content": result["response"],
                "agent": result["agent"]
            }

            # Add source citations if provided by Clinical Agent
            if result.get("sources"):
                assistant_message["sources"] = result["sources"]

            # Add to conversation history
            st.session_state.messages.append(assistant_message)

            # Display assistant response with metadata
            with st.chat_message("assistant"):
                st.write(result["response"])
                st.caption(f"🤖 Agent: {result['agent']}")

                # Display sources for medical information
                if result.get("sources"):
                    st.caption(f"📚 Sources: {', '.join(result['sources'])}")

        else:
            # Handle API errors
            logger.error(f"API error: {response.status_code}")
            st.error(f"❌ API Error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        # Handle connection errors (server not running)
        logger.error("Cannot connect to FastAPI server")
        st.error("❌ Cannot connect to API. Please make sure the FastAPI server is running on port 8000.")

    except requests.exceptions.Timeout:
        # Handle timeout errors
        logger.error("API request timeout")
        st.error("⏱️ Request timeout. Please try again.")

    except Exception as e:
        # Handle any other errors
        logger.error(f"Unexpected error: {str(e)}")
        st.error(f"❌ Error: {str(e)}")

# ==================== USAGE INSTRUCTIONS AND SYSTEM INFO ====================

# Expandable instructions section for user guidance
with st.expander("ℹ️ How to use this system", expanded=False):
    st.markdown("""
    ### Getting Started
    **Step 1:** Enter your name in the text box above to identify yourself

    **Step 2:** Ask questions about your post-discharge care, such as:
    - "How am I feeling today?"
    - "I'm having swelling in my legs. Should I be worried?"
    - "What's the latest research on SGLT2 inhibitors?"
    - "Can you explain my medications?"

    ### System Architecture
    This multi-agent AI system includes:

    - 🤖 **Receptionist Agent**:
      - Handles patient identification and data retrieval
      - Manages general conversation and follow-up questions
      - Routes medical queries to the Clinical Agent

    - 🩺 **Clinical AI Agent**:
      - Provides evidence-based medical advice
      - Uses RAG (Retrieval Augmented Generation) with nephrology guidelines
      - Searches recent medical literature when needed
      - Includes source citations for all medical information

    - 🔍 **Web Search Integration**:
      - Finds latest medical research for recent developments
      - Supplements knowledge base with current information

    - 📚 **Source Citations**:
      - All medical advice includes source references
      - Distinguishes between guidelines and recent research

    ### Technical Features
    - **Patient Data Retrieval**: Automatic lookup of discharge reports
    - **Multi-Agent Workflow**: Intelligent routing between specialized agents
    - **RAG Implementation**: Vector-based search through medical knowledge
    - **Comprehensive Logging**: All interactions are logged for audit trail

    ### Medical Disclaimer
    ⚠️ This system is for educational purposes only. Always consult healthcare professionals for medical advice.
    """)

# Footer with helpful tip
st.markdown("---")
st.caption("💡 **Tip:** Enter your name and ask questions about your post-discharge care to get started!")

# ==================== SYSTEM STATUS INDICATOR ====================

# Optional: Add a small status indicator in the sidebar
with st.sidebar:
    st.markdown("### System Status")

    # Check API connectivity
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("🟢 System Online")
            health_data = health_response.json()
            st.caption(f"Active Sessions: {health_data.get('active_sessions', 0)}")
        else:
            st.warning("🟡 System Issues")
    except:
        st.error("🔴 System Offline")

    st.markdown("---")
    st.markdown("### About")
    st.caption("DataSmith AI GenAI Intern Assignment")
    st.caption("Multi-Agent Medical AI POC")
    st.caption("Version 1.0.0")
