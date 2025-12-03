"""
Streamlit Web Interface for IRDAI Insurance Circulars RAG System

This app provides a chatbot-style interface to query IRDAI insurance regulations
and get AI-generated answers with source citations.
"""

import streamlit as st
import requests
import os
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.debug_utils import is_debug_basic, is_debug_verbose, print_debug_header, print_debug_footer

# Configuration
API_URL = "https://irda-rag-backend.onrender.com"

# Page configuration
st.set_page_config(
    page_title="IRDAI Insurance Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
    <style>
    /* Global styles */
    .main {
        background-color: #202123;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main header */
    .main-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: #ececf1;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #10a37f 0%, #1a7f64 100%);
        border-radius: 8px;
    }

    /* Chat container scrolling */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }

    /* Message styling */
    .chat-message {
        padding: 1.5rem 1rem;
        margin-bottom: 0;
        border-bottom: 1px solid #444654;
    }

    .user-message {
        background-color: #202123;
    }

    .assistant-message {
        background-color: #444654;
    }

    /* Message header (avatar + name) */
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
    }

    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 3px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 12px;
        font-size: 1rem;
    }

    .user-avatar {
        background-color: #5436DA;
        color: white;
    }

    .assistant-avatar {
        background-color: #10a37f;
        color: white;
    }

    .message-role {
        color: #ececf1;
        font-weight: 600;
    }

    /* Message content */
    .message-content {
        color: #ececf1;
        line-height: 1.75;
        font-size: 1rem;
        margin-left: 42px;
    }

    /* Sources section */
    .sources-section {
        margin-top: 1rem;
        margin-left: 42px;
        padding: 0.75rem;
        background-color: rgba(16, 163, 127, 0.1);
        border-left: 3px solid #10a37f;
        border-radius: 4px;
    }

    .sources-header {
        font-size: 0.875rem;
        font-weight: 600;
        color: #10a37f;
        margin-bottom: 0.5rem;
    }

    .source-item {
        font-size: 0.8rem;
        color: #c5c5d2;
        padding: 0.25rem 0;
    }

    /* Input container - removed fixed positioning */
    .input-container {
        padding: 1rem 0;
        background-color: #202123;
        border-top: 1px solid #444654;
    }

    .stTextInput > div > div > input {
        background-color: #202123;
        border: 1px solid #565869;
        color: #ececf1;
        border-radius: 8px;
        padding: 12px 15px;
        font-size: 1rem;
        height: 48px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2);
    }

    /* Buttons */
    .stButton > button {
        background-color: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 12px 16px;
        font-weight: 600;
        transition: background-color 0.2s;
        height: 48px;
    }

    .stButton > button:hover {
        background-color: #1a7f64;
    }

    .stButton > button:active {
        background-color: #0d8066;
    }

    /* Secondary button (New Chat) */
    .stButton > button[kind="secondary"] {
        background-color: #40414f;
        border: 1px solid #565869;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #52525f;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #ececf1;
    }

    /* Metrics and info boxes */
    .stMetric {
        background-color: #2a2b32;
        padding: 1rem;
        border-radius: 8px;
    }

    /* Info/Success/Warning boxes */
    .stInfo, .stSuccess, .stWarning {
        background-color: #2a2b32;
        border-left: 4px solid #10a37f;
        color: #ececf1;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #2a2b32;
    }

    ::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #6e6f80;
    }

    /* Welcome message */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #ececf1;
        text-align: center;
        padding: 2rem;
    }

    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }

    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .welcome-subtitle {
        font-size: 1.1rem;
        color: #c5c5d2;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []


def check_api_health() -> Dict:
    """Check if the API is running and get system stats."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def is_follow_up_question(question: str) -> bool:
    """Detect if a question is a follow-up question."""
    follow_up_indicators = [
        'it', 'this', 'that', 'these', 'those', 'they', 'them',
        'can it', 'does it', 'is it', 'was it', 'will it',
        'when was', 'how was', 'why was', 'where was',
        'who made', 'what about', 'tell me more'
    ]

    question_lower = question.lower()
    first_words = ' '.join(question.split()[:3]).lower() if len(question.split()) >= 3 else question_lower

    return any(indicator in first_words or question_lower.startswith(indicator) for indicator in follow_up_indicators)


def regenerate_answer_with_context(question: str, merged_results: list, conversation_history: list = None, llm_provider: str = None) -> Dict:
    """Regenerate answer using merged context (for follow-up questions)."""
    try:
        # Format context from merged results
        context_parts = []
        sources = []

        for result in merged_results:
            chunk = result.get('chunk', '')
            source = result.get('source_file', '')
            if source not in sources:
                sources.append(source)
            context_parts.append(f"[Source: {source}]\n{chunk}\n")

        context = "\n---\n".join(context_parts)

        # Import answer generator
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / 'src'))
        from llm.answer_generator import AnswerGenerator

        # Prepare conversation history for answer generator (same adaptive logic as query_rag)
        api_conversation_history = None
        if conversation_history and len(conversation_history) > 0:
            # Dynamically determine how many exchanges to send (4-5) based on total token estimate
            max_exchanges = 5
            exchanges_to_send = conversation_history[-max_exchanges:]

            api_conversation_history = []
            total_chars = 0
            max_total_chars = 3000  # Roughly 750 tokens, safe limit

            for entry in exchanges_to_send:
                q = entry.get('question', '')
                a = entry.get('answer', '')
                s = entry.get('sources', [])[:3]

                entry_chars = len(q) + len(a) + sum(len(src) for src in s)

                if total_chars + entry_chars > max_total_chars:
                    remaining = max_total_chars - total_chars - len(q) - sum(len(src) for src in s)
                    if remaining > 200:
                        api_conversation_history.append({
                            "question": q,
                            "answer": a[:remaining] + "...",
                            "sources": s
                        })
                    break
                else:
                    api_conversation_history.append({
                        "question": q,
                        "answer": a,
                        "sources": s
                    })
                    total_chars += entry_chars

        # Generate answer with merged context using selected provider
        provider = llm_provider or st.session_state.get('llm_provider', 'openai')
        generator = AnswerGenerator(provider=provider)
        answer_result = generator.generate_answer(
            question=question,
            context=context,
            conversation_history=api_conversation_history
        )

        return {
            'answer': answer_result.get('answer', 'Failed to generate answer.'),
            'llm_provider': answer_result.get('provider', 'unknown'),
            'llm_model': answer_result.get('model', 'unknown'),
            'tokens_used': answer_result.get('tokens_used')
        }
    except Exception as e:
        st.warning(f"Could not regenerate answer with merged context: {e}")
        return None


def query_rag(question: str, top_k: int = 5, conversation_history: list = None) -> Dict:
    """Query the RAG system with conversation context."""
    try:
        # Detect if this is a follow-up question (for UI indicators)
        is_follow_up = False
        if conversation_history:
            is_follow_up = is_follow_up_question(question)

        # Prepare conversation history for API
        api_conversation_history = None
        if conversation_history and len(conversation_history) > 0:
            # Dynamically determine how many exchanges to send (4-5) based on total token estimate
            # Start with last 5 exchanges, truncate if needed
            max_exchanges = 5
            exchanges_to_send = conversation_history[-max_exchanges:]

            # Build conversation history with adaptive truncation
            api_conversation_history = []
            total_chars = 0
            max_total_chars = 3000  # Roughly 750 tokens, safe limit

            for entry in exchanges_to_send:
                entry_question = entry.get('question', '')
                entry_answer = entry.get('answer', '')
                entry_sources = entry.get('sources', [])[:3]  # Limit to 3 sources

                # Estimate character count for this entry
                entry_chars = len(entry_question) + len(entry_answer) + sum(len(s) for s in entry_sources)

                # If adding this entry would exceed limit, truncate answer
                if total_chars + entry_chars > max_total_chars:
                    # Calculate remaining space
                    remaining = max_total_chars - total_chars - len(entry_question) - sum(len(s) for s in entry_sources)
                    if remaining > 200:  # Only include if we have meaningful space
                        truncated_answer = entry_answer[:remaining] + "..."
                        api_conversation_history.append({
                            "question": entry_question,
                            "answer": truncated_answer,
                            "sources": entry_sources
                        })
                    break  # Stop adding more entries
                else:
                    # Add full entry
                    api_conversation_history.append({
                        "question": entry_question,
                        "answer": entry_answer,
                        "sources": entry_sources
                    })
                    total_chars += entry_chars

        # Get LLM provider from session state
        llm_provider = st.session_state.get('llm_provider', 'openai')

        # Prepare request payload
        # Send the original question - let the pipeline handle query rewriting
        request_payload = {
            "question": question,  # Original question - pipeline will handle rewriting
            "top_k": top_k,
            "conversation_history": api_conversation_history,
            "llm_provider": llm_provider,
            "original_question": None  # Not needed - question is already original
        }

        # DEBUG: Basic level - Show request summary
        if is_debug_basic():
            print(f"\n[Streamlit → API] Question: {question}")
            if api_conversation_history:
                print(f"  → With {len(api_conversation_history)} conversation history entries")
            if is_follow_up:
                print(f"  → Detected as follow-up question")

        # DEBUG: Verbose level - Show full request details
        if is_debug_verbose():
            print_debug_header("REQUEST TO API (Streamlit)", level=2)
            print(f"Question: {question}")
            print(f"Is follow-up: {is_follow_up}")
            print(f"LLM Provider: {llm_provider}")
            print(f"Conversation history entries: {len(api_conversation_history) if api_conversation_history else 0}")
            if api_conversation_history:
                for i, entry in enumerate(api_conversation_history, 1):
                    print(f"\n  Exchange {i}:")
                    print(f"    Q: {entry.get('question', '')[:80]}...")
                    print(f"    A: {entry.get('answer', '')[:80]}...")
                    print(f"    Sources: {entry.get('sources', [])}")
            print_debug_footer(level=2)

        # Send both rephrased query (for retrieval) and original question (for LLM prompt)
        response = requests.post(
            f"{API_URL}/query",
            json=request_payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            needs_answer_regeneration = False

            # If this is a follow-up question and we have conversation history
            # Merge previous context with new results to maintain document continuity
            if is_follow_up and conversation_history and len(conversation_history) > 0:
                # Get the previous query's results
                prev_entry = conversation_history[-1]
                if 'retrieved_chunks' in prev_entry:
                    prev_chunks = prev_entry['retrieved_chunks']

                    # Create a combined context that prioritizes previous sources
                    # This ensures follow-up questions can reference the same documents
                    current_results = result.get('results', [])

                    # Build a map of source -> chunks for deduplication
                    source_chunks = {}

                    # First, add previous chunks (higher priority)
                    for chunk in prev_chunks[:3]:  # Keep top 3 from previous query
                        source = chunk.get('source_file', '')
                        if source not in source_chunks:
                            source_chunks[source] = []
                        source_chunks[source].append(chunk)

                    # Then add new chunks if they're from different sources or highly relevant
                    for chunk in current_results:
                        source = chunk.get('source_file', '')
                        if source not in source_chunks:
                            source_chunks[source] = []
                        # Add if from new source or if it's a different chunk from same source
                        chunk_text = chunk.get('chunk', '')
                        is_duplicate = any(c.get('chunk', '') == chunk_text for c in source_chunks[source])
                        if not is_duplicate:
                            source_chunks[source].append(chunk)

                    # Flatten back to list, prioritizing sources from previous query
                    merged_results = []
                    for chunks in source_chunks.values():
                        merged_results.extend(chunks[:2])  # Max 2 chunks per source

                    # Limit to top_k total
                    result['results'] = merged_results[:top_k]

                    # Flag that we need to regenerate the answer with merged context
                    needs_answer_regeneration = True

            # Regenerate answer if we merged contexts
            if needs_answer_regeneration:
                regenerated = regenerate_answer_with_context(question, result['results'], conversation_history, llm_provider)
                if regenerated:
                    result['answer'] = regenerated['answer']
                    result['llm_provider'] = regenerated.get('llm_provider')
                    result['llm_model'] = regenerated.get('llm_model')
                    if regenerated.get('tokens_used'):
                        result['tokens_used'] = regenerated['tokens_used']

            # Extract supporting texts from results
            supporting_texts = {}
            if result.get('results'):
                for i, chunk_result in enumerate(result['results'][:5], 1):
                    source = chunk_result.get('source_file', f'Source {i}')
                    chunk_text = chunk_result.get('chunk', '')
                    if source not in supporting_texts:
                        supporting_texts[source] = chunk_text

            result['supporting_texts'] = supporting_texts
            result['rephrased_query'] = None  # Pipeline handles query rewriting now
            result['is_follow_up'] = is_follow_up

            return result
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the API server is running at http://127.0.0.1:8000")
        st.info("Run: `cd rag-irdai-chatbot/src/api && uvicorn main:app --reload`")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def display_chat_message(message: Dict, message_type: str):
    """Display a chat message in ChatGPT style."""
    import re

    css_class = "user-message" if message_type == "user" else "assistant-message"
    avatar_class = "user-avatar" if message_type == "user" else "assistant-avatar"
    avatar_icon = "👤" if message_type == "user" else "🤖"
    role_name = "You" if message_type == "user" else "IRDAI Assistant"

    # Get the raw content
    content = message.get('content', '')

    # Convert markdown-style formatting to HTML for display
    # Convert **bold** to <strong>bold</strong>
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    # Convert line breaks to <br>
    content = content.replace('\n', '<br>')

    message_html = f"""
    <div class="chat-message {css_class}">
        <div class="message-header">
            <div class="avatar {avatar_class}">{avatar_icon}</div>
            <div class="message-role">{role_name}</div>
        </div>
        <div class="message-content">{content}</div>
    </div>
    """

    st.markdown(message_html, unsafe_allow_html=True)

    # Add sources section for assistant messages (outside the main message div for better rendering)
    if message_type == "assistant" and message.get('sources'):
        sources_html = """
        <div class="sources-section">
            <div class="sources-header">📚 Sources</div>
        """
        for i, source in enumerate(message['sources'], 1):
            # Truncate long source names
            source_display = source if len(source) <= 80 else source[:77] + "..."
            sources_html += f'<div class="source-item">[{i}] {source_display}</div>'
        sources_html += "</div>"

        st.markdown(sources_html, unsafe_allow_html=True)

    # Show supporting text in expandable sections for each source
    if message_type == "assistant" and message.get('supporting_texts'):
        for i, (source, text) in enumerate(message['supporting_texts'].items(), 1):
            source_display = source if len(source) <= 60 else source[:57] + "..."
            with st.expander(f"📄 View supporting text from [{i}] {source_display}"):
                st.text_area(
                    "Supporting Text",
                    value=text,
                    height=150,
                    key=f"support_{message.get('timestamp')}_{i}",
                    label_visibility="collapsed"
                )


def main():
    """Main application."""

    # Header
    st.markdown('<p class="main-header">🏛️ IRDAI Insurance Chatbot</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Check API health
        health = check_api_health()
        if health:
            st.success("✅ API is running")
            st.metric("Embedding Model", health.get('embedding_model', 'N/A').split(':')[-1])
        else:
            st.error("❌ API is not running")

        st.markdown("---")

        # LLM Provider Selection
        st.markdown("### 🤖 LLM Provider")

        # Initialize LLM provider in session state if not exists
        if 'llm_provider' not in st.session_state:
            st.session_state.llm_provider = os.getenv("LLM_PROVIDER", "openai")

        llm_provider = st.selectbox(
            "Select LLM Provider",
            options=["openai", "groq"],
            index=1 if st.session_state.llm_provider == "openai" else 1,
            help="OpenAI (paid, high quality) vs Groq (free, fast)",
            key="llm_provider_selector"
        )

        # Update session state
        st.session_state.llm_provider = llm_provider

        # Show model info
        if llm_provider == "openai":
            st.info("💰 Using OpenAI GPT-4o-mini (paid)")
        else:
            st.success("🆓 Using Groq Llama-3.3-70b (free)")

        st.markdown("---")

        # Query settings
        st.markdown("### 🔧 Query Settings")
        top_k = st.slider(
            "Number of results to retrieve",
            min_value=1,
            max_value=20,
            value=5,
            help="How many relevant document chunks to retrieve"
        )

        use_context = st.checkbox(
            "Use conversation context",
            value=True,
            help="Include previous conversation history for better context understanding"
        )

        st.markdown("---")

    # Main chat area - Chatbot Container
    chat_container = st.container(height=600)
    with chat_container:
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                display_chat_message(message, message['type'])

            # Add an invisible anchor at the bottom for auto-scroll
            st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
        else:
            # ChatGPT-style welcome message
            st.markdown("""
                <div class="welcome-container">
                    <div class="welcome-icon">🏛️</div>
                    <div class="welcome-title">IRDAI Insurance Assistant</div>
                    <div class="welcome-subtitle">Ask me anything about IRDAI insurance regulations and circulars</div>
                </div>
            """, unsafe_allow_html=True)

    # Auto-scroll script - runs after container renders
    if st.session_state.chat_history:
        st.markdown("""
            <script>
                // Wait for DOM to be ready
                setTimeout(function() {
                    // Find all scrollable containers
                    var containers = window.parent.document.querySelectorAll('[data-testid="stVerticalBlock"]');
                    containers.forEach(function(container) {
                        if (container.scrollHeight > container.clientHeight) {
                            container.scrollTop = container.scrollHeight;
                        }
                    });
                }, 100);
            </script>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Question input section
    # Check if we need to clear the input
    if 'clear_input' in st.session_state and st.session_state.clear_input:
        # Reset the widget's session state directly
        if 'question_input' in st.session_state:
            st.session_state.question_input = ""
        st.session_state.clear_input = False

    # Input area with Send and New Chat buttons using form for Enter key support
    with st.form(key="question_form", clear_on_submit=False):
        col_input, col_send, col_new = st.columns([7, 1, 2])

        with col_input:
            question = st.text_input(
                "Ask a question:",
                placeholder="Send a message...",
                label_visibility="collapsed",
                key="question_input"
            )

        with col_send:
            send_button = st.form_submit_button("➤", type="primary", use_container_width=True, help="Send message (or press Enter)")

        with col_new:
            new_chat_button = st.form_submit_button("New Chat", use_container_width=True, help="Start a new conversation")

    # Handle New Chat button click
    if new_chat_button:
        st.session_state.chat_history = []
        st.session_state.conversation_context = []
        st.rerun()

    # Process query (triggered by Enter key or Send button)
    if send_button and question.strip():
        # Add user message to chat
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_message = {
            'type': 'user',
            'content': question,
            'timestamp': timestamp
        }
        st.session_state.chat_history.append(user_message)

        # Query the RAG system
        with st.spinner("🔄 Thinking..."):
            context_to_use = st.session_state.conversation_context
            result = query_rag(
                question,
                top_k,
                conversation_history=context_to_use,
            )

            if result and result.get('answer'):
                # Show rephrased query if different
                if result.get('rephrased_query') and result['rephrased_query'] != question:
                    st.info(f"🔄 Rephrased query for better retrieval: {result['rephrased_query'][:150]}...")

                # Show follow-up context indicator
                if result.get('is_follow_up'):
                    st.success("🔗 Follow-up question detected - using context from previous query")

                # Add assistant message to chat with supporting texts
                assistant_message = {
                    'type': 'assistant',
                    'content': result['answer'],
                    'sources': result.get('sources', []),
                    'supporting_texts': result.get('supporting_texts', {}),
                    'timestamp': timestamp,
                    'metadata': {
                        'llm_provider': result.get('llm_provider'),
                        'llm_model': result.get('llm_model'),
                        'total_results': result.get('total_results', 0),
                        'rephrased_query': result.get('rephrased_query')
                    }
                }
                st.session_state.chat_history.append(assistant_message)

                # Update conversation context with richer information including retrieved chunks
                st.session_state.conversation_context.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'retrieved_chunks': result.get('results', []),  # Store retrieved chunks for follow-up context
                    'timestamp': timestamp,
                    'metadata': {
                        'total_results': result.get('total_results', 0),
                        'is_follow_up': result.get('is_follow_up', False)
                    }
                })

                # Keep only last 5 exchanges in context for better multi-turn memory
                # (reduced from 10 to avoid context bloat with retrieved_chunks)
                if len(st.session_state.conversation_context) > 5:
                    st.session_state.conversation_context = st.session_state.conversation_context[-5:]

            else:
                error_message = {
                    'type': 'assistant',
                    'content': "I apologize, but I couldn't generate an answer. Please try rephrasing your question.",
                    'timestamp': timestamp
                }
                st.session_state.chat_history.append(error_message)

        # Set flag to clear input on next render
        st.session_state.clear_input = True
        st.rerun()

    # Show detailed view in expander
    if st.session_state.chat_history:
        with st.expander("📊 View Detailed Conversation Analytics"):
            st.json({
                'total_messages': len(st.session_state.chat_history),
                'conversation_context_size': len(st.session_state.conversation_context),
                'chat_history': st.session_state.chat_history
            })


if __name__ == "__main__":
    main()
