
import streamlit as st
import pdfplumber
from PyPDF2 import PdfReader
import base64
import re
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Helper function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Resume section extractor
def extract_sections(text):
    sections = {
        "summary": re.findall(r"(summary|about me)[\s\S]*?(education|experience|skills|projects)", text, re.I),
        "education": re.findall(r"(education)[\s\S]*?(experience|skills|projects)", text, re.I),
        "experience": re.findall(r"(experience)[\s\S]*?(skills|projects|certifications)", text, re.I),
        "skills": re.findall(r"(skills)[\s\S]*?(certifications|projects)", text, re.I),
    }
    return sections

# Simple keyword matching
def keyword_match(text, job_keywords):
    text = text.lower()
    matched = [kw for kw in job_keywords if kw.lower() in text]
    missing = [kw for kw in job_keywords if kw.lower() not in text]
    return matched, missing

# Load PDF and return text
def load_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

# AI bullet suggestion via LangChain
def ai_suggest_bullets(text, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.3)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(docs)
    return summary

# Streamlit App UI
st.set_page_config(layout="wide", page_title="Resume Scorer AI")

st.sidebar.title("Resume Scorer 🔍")
uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
job_keywords_input = st.sidebar.text_area("Enter Job Description Keywords (comma-separated)", height=100)
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

if uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📄 Resume Preview")
        base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        uploaded_file.seek(0)

    with col2:
        st.subheader("📊 Resume Analysis")

        full_text = extract_text_from_pdf(uploaded_file)
        word_count = len(full_text.split())

        st.markdown(f"**Total Words:** {word_count}")

        sections = extract_sections(full_text.lower())
        for sec in ["summary", "education", "experience", "skills"]:
            st.markdown(f"### 📌 {sec.capitalize()}")
            if sections[sec]:
                st.success("✅ Section found.")
            else:
                st.error("❌ Section missing.")

        if job_keywords_input:
            job_keywords = [kw.strip() for kw in job_keywords_input.split(",")]
            matched, missing = keyword_match(full_text, job_keywords)

            st.markdown("### 🔎 Keyword Match")
            st.markdown(f"**Matched ({len(matched)}):** {', '.join(matched)}")
            st.markdown(f"**Missing ({len(missing)}):** {', '.join(missing)}")

        if openai_key:
            st.markdown("### 🤖 AI Bullet Suggestions")
            suggestion = ai_suggest_bullets(full_text, openai_key)
            st.info(suggestion)
