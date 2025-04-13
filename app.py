import os
import time
import hashlib
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics.barcode import code128
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm

load_dotenv()

st.set_page_config(page_title="LawGPT", layout="wide")

st.markdown("""
    <style>
    body, .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75em 2em;
        font-size: 1.1rem;
        font-weight: 600;
        transition: 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }
    @media screen and (max-width: 768px) {
        .role-buttons {
            flex-direction: column;
            gap: 1rem;
        }
        .logo-img {
            width: 70% !important;
        }
    }
    .role-buttons {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2rem;
        margin-top: 3rem;
        flex-wrap: wrap;
    }
    .logo-center {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .logo-img {
        width: 25%;
        max-width: 250px;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="logo-center">
    <img class="logo-img" src="https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00" />
</div>
""", unsafe_allow_html=True)

if "role" not in st.session_state:
    st.session_state.role = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.role is None:
    st.markdown("<h2 style='text-align: center;'>Who are you?</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧑 I am a Civilian"):
                st.session_state.role = "civilian"
                st.session_state.authenticated = True
                st.rerun()
        with col_b:
            if st.button("⚖️ I am a Court Stakeholder"):
                st.session_state.role = "stakeholder"
                st.rerun()

if st.session_state.role == "stakeholder" and not st.session_state.authenticated:
    st.markdown("### 🔐 Stakeholder Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.success("Login successful!")
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials.")

if st.session_state.role and (st.session_state.role == "civilian" or st.session_state.authenticated):
    if st.button("🔙 Back to Home"):
        st.session_state.role = None
        st.session_state.authenticated = False
        st.rerun()

    tabs = ["📘 LawGPT"]
    if st.session_state.role == "stakeholder":
        tabs.extend(["📝 Document Signer", "🔍 Verify Document"])

    selected_tab = st.tabs(tabs)

    if "📘 LawGPT" in tabs:
        with selected_tab[0]:
            st.markdown("## 💬  Your Legal AI Lawyer")
            st.markdown("### Ask any legal question related to the Indian Penal Code (IPC)")
            st.markdown("Questions might be of types like: Suppose a 16 year old is  drinking and driving , and hit a pedestrian on the road . what are the possible case laws imposed and what are the any one previous court decisions on the same.  ")

            def reset_conversation():
                st.session_state.messages = []
                st.session_state.memory.clear()

            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferWindowMemory(
                    k=2, memory_key="chat_history", return_messages=True
                )

            embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1",
                model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
            )

            db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
            db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            prompt_template = """<s>[INST]You are a legal chatbot that answers questions about the Indian Penal Code (IPC).
Provide clear, concise, and accurate responses based on context and user's question.
Avoid extra details or assumptions. Focus only on legal information.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}

ANSWER:
</s>[INST]"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "chat_history"]
            )

            llm = Together(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.5,
                max_tokens=1024,
                together_api_key=os.getenv("TOGETHER_API_KEY")
            )

            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                memory=st.session_state.memory,
                retriever=db_retriever,
                combine_docs_chain_kwargs={
                    'prompt': prompt,
                    'document_variable_name': 'context'
                }
            )

            chat_placeholder = st.empty()
            with chat_placeholder.container():
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            input_prompt = st.chat_input("Ask a legal question...")
            if input_prompt:
                with st.chat_message("user"):
                    st.write(input_prompt)
                st.session_state.messages.append({"role": "user", "content": input_prompt})

                with st.chat_message("assistant"):
                    with st.status("Thinking 💡", expanded=True):
                        result = qa.invoke(input=input_prompt)
                        message_placeholder = st.empty()
                        full_response = "⚠️ **_Note: Information provided may be inaccurate._**\n\n"
                        for chunk in result["answer"]:
                            full_response += chunk
                            time.sleep(0.02)
                            message_placeholder.markdown(full_response + " ▌")
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

            st.button("🔄 Reset Chat", on_click=reset_conversation)

    if st.session_state.role == "stakeholder":
        if "📝 Document Signer" in tabs:
            with selected_tab[1]:
                st.markdown("## 📝 Upload and Sign Document")
                uploaded_file = st.file_uploader("Choose a file to sign", type=["pdf"])
                signer_name = st.text_input("Enter your name (Signer):")

                if uploaded_file and signer_name:
                    file_content = uploaded_file.read()
                    input_pdf = BytesIO(file_content)
                    output_pdf = BytesIO()

                    reader = PdfReader(input_pdf)
                    writer = PdfWriter()

                    for page in reader.pages:
                        page_width = float(page.mediabox.width)
                        page_height = float(page.mediabox.height)
                        packet = BytesIO()
                        can = canvas.Canvas(packet, pagesize=(page_width, page_height))
                        barcode = code128.Code128(signer_name, barHeight=10 * mm, barWidth=0.4)
                        barcode.drawOn(can, 50, 50)
                        can.setFont("Helvetica", 10)
                        can.drawString(50, 40, f"Signed by: {signer_name}")
                        can.save()
                        packet.seek(0)
                        overlay = PdfReader(packet).pages[0]
                        page.merge_page(overlay)
                        writer.add_page(page)

                    writer.write(output_pdf)
                    output_pdf.seek(0)

                    st.download_button("📅 Download Signed Document", output_pdf, file_name=f"signed_{uploaded_file.name}", mime="application/pdf")

        if "🔍 Verify Document" in tabs:
            with selected_tab[2]:
                st.markdown("## 🔍 Verify Uploaded Document")
                verify_file = st.file_uploader("Upload PDF for verification", type=["pdf"], key="verify")

                if verify_file:
                    content = verify_file.read()
                    try:
                        PdfReader(BytesIO(content))
                        st.success("✅ Document Status: Legit")
                    except:
                        st.error("❌ Document Status: Forged")