from dotenv import load_dotenv
import streamlit as st
import os
import PyPDF2
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Initialize Haystack components for RAG
document_store = InMemoryDocumentStore()
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="Facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="Facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True
)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
pipe = ExtractiveQAPipeline(reader, retriever)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_gemini_response(input_text, resume_text, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input_text, resume_text, prompt])
    return response.text

def input_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def extract_keywords(text):
    vectorizer = CountVectorizer(stop_words='English', max_features=10)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def retrieve_relevant_content(query, documents):
    # Add documents to the document store
    for doc in documents:
        document_store.write_documents([{"content": doc, "meta": {"name": "document"}}])
    document_store.update_embeddings(retriever)
    
    # Retrieve relevant content
    retrieved_docs = retriever.retrieve(query)
    return " ".join([doc.content for doc in retrieved_docs])

# Streamlit App configuration
st.set_page_config(page_title="Resume Reboot: Your ATS Optimization Expert")
st.header("Resume Reboot: Your ATS Optimization Expert")

# Input fields
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

# Check if the file is uploaded successfully
if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

# Buttons for actions
submit1 = st.button("Identify Resume Flaws")
submit2 = st.button("Skill Enhancement Recommendations")
submit3 = st.button("Keyword Optimization")
submit4 = st.button("Percentage Match")

# Input prompts
input_prompt1 = """
You are an experienced Technical Human Resource Manager. Review the provided resume against the job description.
Please share your professional evaluation on whether the candidate's profile aligns with the role.
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt2 = """
You are a skilled HR professional. Evaluate the provided resume against the job description and identify the skills
that the candidate needs to improve or acquire to be a better fit for the role.
"""

input_prompt3 = """
You are an ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
Evaluate the resume against the provided job description. Identify and list the important keywords missing from the resume.
"""

input_prompt4 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
Evaluate the resume against the provided job description. Give the percentage match, list the missing keywords, and provide your final thoughts.
"""

# Handle submit actions
if submit1:
    if uploaded_file is not None:
        resume_text = input_pdf_text(uploaded_file)
        relevant_content = retrieve_relevant_content(input_text, [resume_text])
        response = get_gemini_response(input_text, relevant_content, input_prompt1)
        st.subheader("Resume Evaluation")
        st.write(response)
    else:
        st.write("Please upload the resume")

elif submit2:
    if uploaded_file is not None:
        resume_text = input_pdf_text(uploaded_file)
        relevant_content = retrieve_relevant_content(input_text, [resume_text])
        response = get_gemini_response(input_text, relevant_content, input_prompt2)
        st.subheader("Skill Enhancement Recommendations")
        st.write(response)
    else:
        st.write("Please upload the resume")

elif submit3:
    if uploaded_file is not None:
        resume_text = input_pdf_text(uploaded_file)
        relevant_content = retrieve_relevant_content(input_text, [resume_text])
        response = get_gemini_response(input_text, relevant_content, input_prompt3)
        st.subheader("Keyword Optimization")
        st.write(response)
    else:
        st.write("Please upload the resume")

elif submit4:
    if uploaded_file is not None:
        resume_text = input_pdf_text(uploaded_file)
        relevant_content = retrieve_relevant_content(input_text, [resume_text])
        
        # Get BERT embeddings
        job_embedding = get_bert_embedding(input_text)
        resume_embedding = get_bert_embedding(relevant_content)
        
        # Compute cosine similarity
        similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0] * 100
        
        # Generate response from the model
        response = get_gemini_response(input_text, relevant_content, input_prompt4)
        st.subheader("Percentage Match")
        st.write(f"Percentage Match: {similarity:.2f}%")
        st.write(response)
    else:
        st.write("Please upload the resume")