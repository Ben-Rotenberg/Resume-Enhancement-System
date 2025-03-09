import os
import streamlit as st
import tempfile
from pathlib import Path
import base64
from typing import Dict, List, Any, Optional, Tuple
import uuid
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
import operator
from pydantic import BaseModel, Field

# Configure page
st.set_page_config(page_title="Resume Enhancement System", layout="wide")

# State management
class AppState:
    def __init__(self):
        self.resume_content = None
        self.resume_analysis = None
        self.interview_questions = None
        self.interview_chat_history = []
        self.interview_insights = None
        self.enhanced_resume = None
        self.verification_result = None
        self.current_step = "upload"  # upload, analysis, interview, enhancement, verification, download
        self.api_key = None

app_state = AppState()

# UI functions
def render_sidebar():
    st.sidebar.title("Resume Enhancement System")
    st.sidebar.markdown("---")
    
    # API Key input
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="api_key_input")
    if api_key:
        app_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.sidebar.markdown("---")
    
    # Progress tracker
    st.sidebar.subheader("Progress")
    steps = ["Resume Upload", "Analysis", "Interview", "Enhancement", "Verification", "Download"]
    current_step_index = ["upload", "analysis", "interview", "enhancement", "verification", "download"].index(app_state.current_step)
    
    for i, step in enumerate(steps):
        if i < current_step_index:
            st.sidebar.markdown(f"✅ {step}")
        elif i == current_step_index:
            st.sidebar.markdown(f"🔄 {step}")
        else:
            st.sidebar.markdown(f"⬜ {step}")

def create_download_link(content, filename, link_text):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

def text_to_pdf(text, filename):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Set up text properties
    c.setFont("Helvetica", 10)
    y_position = height - 40
    margin = 50
    line_height = 14
    
    # Split text into lines
    lines = text.split('\n')
    
    for line in lines:
        # If line would go off page, start a new page
        if y_position < margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = height - 40
        
        # Write the line
        c.drawString(margin, y_position, line)
        y_position -= line_height
    
    c.save()
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content

# LangChain agent definitions
def create_llm(temperature=0):
    """Create an LLM with the provided API key"""
    if not app_state.api_key:
        st.error("Please provide an OpenAI API key in the sidebar")
        st.stop()
    
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        api_key=app_state.api_key
    )

# Agent 1: Resume Analyzer
def analyze_resume(resume_content):
    """Analyze resume for gaps and weaknesses"""
    llm = create_llm(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a professional resume analyzer. Your task is to:
1. Analyze the resume in detail
2. Identify gaps, weaknesses, and areas for improvement
3. Note any missing information that would strengthen the resume
4. Evaluate the resume's structure, format, and content
5. Suggest specific improvements

Be thorough in your analysis. Format your response with clear sections and bullet points."""),
        HumanMessage(content=f"Here is the resume to analyze:\n\n{resume_content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

# Agent 2: Interview Question Generator
def generate_interview_questions(resume_content, resume_analysis):
    """Generate interview questions based on resume analysis"""
    llm = create_llm(temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert interview question generator. Your task is to:
1. Create 8-10 thoughtful interview questions based on the resume and its analysis
2. Focus questions on areas that need clarification or expansion
3. Include questions about missing information identified in the analysis
4. Design questions that will help extract the candidate's accomplishments and skills not fully represented in the resume
5. Format each question with clear numbering

The questions should be conversational and help the interviewer gather valuable information to enhance the resume."""),
        HumanMessage(content=f"""Here is the resume:
{resume_content}

Here is the analysis of the resume:
{resume_analysis}

Based on this information, generate interview questions to help fill gaps and strengthen the resume.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

# Agent 3: Chat Interviewer
class ChatInterviewer:
    def __init__(self, resume_content, resume_analysis, interview_questions):
        self.resume_content = resume_content
        self.resume_analysis = resume_analysis
        self.interview_questions = interview_questions
        self.llm = create_llm(temperature=0.7)
        self.system_prompt = f"""You are an AI interviewer named Alex conducting a friendly conversation to help improve the candidate's resume. 

Here is their current resume:
{resume_content}

Here is an analysis of gaps and weaknesses in the resume:
{resume_analysis}

Here are some questions you should try to naturally incorporate into the conversation:
{interview_questions}

IMPORTANT INSTRUCTIONS:
1. Be conversational and friendly, not robotic or interrogative
2. Don't ask all questions at once - weave them naturally into the conversation
3. Listen to the candidate's responses and ask meaningful follow-up questions
4. Focus on extracting specific achievements, metrics, skills, and experiences
5. Use the questions as a guide, but prioritize conversation flow
6. Mirror and reflect the candidate's language and ideas back to them
7. Your goal is to uncover valuable information missing from the resume
8. Don't strictly follow the question list - be adaptable and responsive
9. Keep responses concise and focused on one topic at a time

Remember: This is a conversation to help enhance their resume, not a formal interview."""
    
    def get_response(self, message_history):
        """Get a response from the chat interviewer based on conversation history"""
        messages = [SystemMessage(content=self.system_prompt)]
        
        for msg in message_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        response = self.llm.invoke(messages)
        return response.content

# Agent 4: Insights Generator
def generate_insights(resume_content, chat_history):
    """Extract insights from interview chat to enhance resume"""
    llm = create_llm(temperature=0.1)
    
    # Format chat history for the prompt
    formatted_chat = "\n".join([f"{'User' if msg['role'] == 'user' else 'Interviewer'}: {msg['content']}" for msg in chat_history])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert at extracting valuable insights from interviews to enhance resumes. Your task is to:
1. Analyze the interview conversation carefully
2. Identify specific achievements, skills, experiences, and metrics mentioned
3. Note any clarifications or additional context provided about resume items
4. Extract insights about the candidate's strengths not fully represented in the original resume
5. Organize these insights into clear categories (e.g., Skills, Achievements, Experience, Education, etc.)
6. Format your findings with clear sections and bullet points

Your insights will be used to enhance the candidate's resume."""),
        HumanMessage(content=f"""Here is the original resume:
{resume_content}

Here is the interview conversation:
{formatted_chat}

Based on this conversation, extract valuable insights that could enhance the resume.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

# Agent 5: Resume Enhancer
def enhance_resume(original_resume, insights):
    """Create an enhanced resume based on original and insights"""
    llm = create_llm(temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a professional resume writer. Your task is to:
1. Create an enhanced version of the resume incorporating the insights from the interview
2. Maintain the original resume's structure but improve it where needed
3. Add specific achievements, metrics, and experiences from the insights
4. Strengthen the language and impact of bullet points
5. Ensure the resume remains factual and truthful - don't invent information
6. Keep the resume concise and professional
7. Focus on quantifiable achievements and specific skills

Output the complete enhanced resume in a clean, professional format."""),
        HumanMessage(content=f"""Here is the original resume:
{original_resume}

Here are the insights from the interview to incorporate:
{insights}

Create an enhanced version of the resume that incorporates these insights.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

# Agent 6: Fact Checker
def verify_resume(original_resume, enhanced_resume):
    """Verify the enhanced resume for accuracy"""
    llm = create_llm(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a resume fact checker and accuracy verifier. Your task is to:
1. Compare the original and enhanced resumes carefully
2. Identify any potential inaccuracies, exaggerations, or fabrications in the enhanced resume
3. Verify that all information in the enhanced resume is factually supported by either the original resume or could be reasonably inferred
4. If you find issues, provide corrections that maintain the improved quality while ensuring accuracy
5. If no issues are found, confirm the enhanced resume's accuracy

Be thorough in your verification. The final resume must be both improved AND accurate."""),
        HumanMessage(content=f"""Here is the original resume:
{original_resume}

Here is the enhanced resume:
{enhanced_resume}

Please verify the enhanced resume for accuracy and provide a corrected version if needed.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

# Streamlit UI
def main():
    render_sidebar()
    
    # Upload step
    if app_state.current_step == "upload":
        st.title("Resume Enhancement System")
        st.markdown("### Upload your resume to get started")
        st.markdown("This system will analyze your resume, conduct an interview, and create an enhanced version.")
        
        uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
        
        if uploaded_file is not None and app_state.api_key:
            # Process the uploaded file
            with st.spinner("Processing your resume..."):
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    file_path = tmp_file.name
                
                try:
                    # Load the document based on file type
                    if file_extension == "pdf":
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                        app_state.resume_content = "\n".join([page.page_content for page in pages])
                    else:  # txt
                        loader = TextLoader(file_path)
                        documents = loader.load()
                        app_state.resume_content = documents[0].page_content
                    
                    # Clean up the temporary file
                    os.unlink(file_path)
                    
                    # Move to the next step
                    app_state.current_step = "analysis"
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        elif uploaded_file is not None and not app_state.api_key:
            st.warning("Please enter your OpenAI API key in the sidebar before proceeding.")
    
    # Analysis step
    elif app_state.current_step == "analysis":
        st.title("Resume Analysis")
        
        if not app_state.resume_analysis:
            with st.spinner("Analyzing your resume..."):
                app_state.resume_analysis = analyze_resume(app_state.resume_content)
        
        st.markdown("### Resume Analysis")
        st.markdown(app_state.resume_analysis)
        
        if not app_state.interview_questions:
            with st.spinner("Generating interview questions..."):
                app_state.interview_questions = generate_interview_questions(
                    app_state.resume_content, app_state.resume_analysis
                )
        
        st.markdown("### Interview Questions")
        st.markdown(app_state.interview_questions)
        
        if st.button("Continue to Interview"):
            app_state.current_step = "interview"
            st.experimental_rerun()
    
    # Interview step
    elif app_state.current_step == "interview":
        st.title("Interview Chat")
        st.markdown("Chat with our AI interviewer to help enhance your resume. The interviewer will ask questions to gather additional information about your experience and skills.")
        
        # Initialize interviewer if not already done
        if not hasattr(app_state, "interviewer"):
            app_state.interviewer = ChatInterviewer(
                app_state.resume_content,
                app_state.resume_analysis,
                app_state.interview_questions
            )
        
        # Initialize chat history if empty
        if not app_state.interview_chat_history:
            initial_message = {
                "role": "assistant", 
                "content": "Hi there! I'm Alex, and I'll be chatting with you today to help enhance your resume. I've reviewed your current resume and noticed some areas we could expand on. Let's have a conversation about your experience and skills to gather more details. How does that sound?"
            }
            app_state.interview_chat_history.append(initial_message)
        
        # Display chat history
        for message in app_state.interview_chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        if user_input:
            # Add user message to chat history
            app_state.interview_chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            # Get response from interviewer
            with st.spinner("Thinking..."):
                response = app_state.interviewer.get_response(app_state.interview_chat_history)
            
            # Add assistant response to chat history
            app_state.interview_chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            
            # Force UI refresh
            st.experimental_rerun()
        
        # Finish interview button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if len(app_state.interview_chat_history) > 3:  # At least 2 user responses
                if st.button("Finish Interview"):
                    app_state.current_step = "enhancement"
                    st.experimental_rerun()
    
    # Enhancement step
    elif app_state.current_step == "enhancement":
        st.title("Resume Enhancement")
        
        if not app_state.interview_insights:
            with st.spinner("Generating insights from interview..."):
                app_state.interview_insights = generate_insights(
                    app_state.resume_content, 
                    app_state.interview_chat_history
                )
        
        st.markdown("### Insights from Interview")
        st.markdown(app_state.interview_insights)
        
        if not app_state.enhanced_resume:
            with st.spinner("Creating enhanced resume..."):
                app_state.enhanced_resume = enhance_resume(
                    app_state.resume_content,
                    app_state.interview_insights
                )
        
        st.markdown("### Enhanced Resume Draft")
        st.markdown(app_state.enhanced_resume)
        
        if st.button("Continue to Verification"):
            app_state.current_step = "verification"
            st.experimental_rerun()
    
    # Verification step
    elif app_state.current_step == "verification":
        st.title("Resume Verification")
        
        if not app_state.verification_result:
            with st.spinner("Verifying enhanced resume for accuracy..."):
                app_state.verification_result = verify_resume(
                    app_state.resume_content,
                    app_state.enhanced_resume
                )
                
                # Check if verification result contains a corrected resume
                # This is a simple heuristic - if the verification result is long, it likely contains a corrected resume
                if len(app_state.verification_result.split()) > 200:
                    # Extract the corrected resume from the verification result
                    app_state.enhanced_resume = app_state.verification_result
                    app_state.verification_result = "The enhanced resume has been verified and corrected for accuracy."
        
        st.markdown("### Verification Result")
        st.markdown(app_state.verification_result)
        
        st.markdown("### Final Enhanced Resume")
        st.markdown(app_state.enhanced_resume)
        
        if st.button("Continue to Download"):
            app_state.current_step = "download"
            st.experimental_rerun()
    
    # Download step
    elif app_state.current_step == "download":
        st.title("Download Enhanced Resume")
        
        st.markdown("### Final Enhanced Resume")
        st.markdown(app_state.enhanced_resume)
        
        st.markdown("### Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text download
            st.download_button(
                label="Download as Text",
                data=app_state.enhanced_resume,
                file_name="enhanced_resume.txt",
                mime="text/plain"
            )
        
        with col2:
            # PDF download
            pdf_data = text_to_pdf(app_state.enhanced_resume, "enhanced_resume.pdf")
            st.download_button(
                label="Download as PDF",
                data=pdf_data,
                file_name="enhanced_resume.pdf",
                mime="application/pdf"
            )
        
        # Copy to clipboard option
        st.markdown("### Copy to Clipboard")
        st.code(app_state.enhanced_resume)
        
        # Start over button
        if st.button("Start Over with a New Resume"):
            # Reset the state
            app_state.__init__()
            st.experimental_rerun()

if __name__ == "__main__":
    main()
