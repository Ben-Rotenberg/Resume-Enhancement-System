"""
This is an alternative implementation using LangGraph more extensively
for orchestrating the multi-agent workflow.

Note: This is not used in the main application but provided
as a reference for a more formal agent architecture approach.
"""

import operator
from typing import Dict, List, Any, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# State definition
class ResumeState(TypedDict):
    resume_content: str
    resume_analysis: str
    interview_questions: str
    chat_history: List[Dict[str, str]]
    interview_insights: str
    enhanced_resume: str
    verification_result: str
    final_resume: str

# Node definitions
def analyze_resume(state: ResumeState) -> ResumeState:
    """Agent 1: Analyze the resume for gaps and weaknesses"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a professional resume analyzer. Your task is to:
1. Analyze the resume in detail
2. Identify gaps, weaknesses, and areas for improvement
3. Note any missing information that would strengthen the resume
4. Evaluate the resume's structure, format, and content
5. Suggest specific improvements

Be thorough in your analysis. Format your response with clear sections and bullet points."""),
        HumanMessage(content=f"Here is the resume to analyze:\n\n{state['resume_content']}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({})
    
    return {"resume_analysis": analysis}

def generate_questions(state: ResumeState) -> ResumeState:
    """Agent 2: Generate interview questions based on resume analysis"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert interview question generator. Your task is to:
1. Create 8-10 thoughtful interview questions based on the resume and its analysis
2. Focus questions on areas that need clarification or expansion
3. Include questions about missing information identified in the analysis
4. Design questions that will help extract the candidate's accomplishments and skills
5. Format each question with clear numbering

The questions should be conversational and help gather valuable information to enhance the resume."""),
        HumanMessage(content=f"""Here is the resume:
{state['resume_content']}

Here is the analysis of the resume:
{state['resume_analysis']}

Based on this information, generate interview questions to help fill gaps and strengthen the resume.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    questions = chain.invoke({})
    
    return {"interview_questions": questions}

def generate_insights(state: ResumeState) -> ResumeState:
    """Agent 4: Extract insights from interview chat"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # Format chat history for the prompt
    formatted_chat = "\n".join([f"{'User' if msg['role'] == 'user' else 'Interviewer'}: {msg['content']}" for msg in state['chat_history']])
    
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
{state['resume_content']}

Here is the interview conversation:
{formatted_chat}

Based on this conversation, extract valuable insights that could enhance the resume.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    insights = chain.invoke({})
    
    return {"interview_insights": insights}

def enhance_resume(state: ResumeState) -> ResumeState:
    """Agent 5: Create an enhanced resume"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
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
{state['resume_content']}

Here are the insights from the interview to incorporate:
{state['interview_insights']}

Create an enhanced version of the resume that incorporates these insights.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    enhanced = chain.invoke({})
    
    return {"enhanced_resume": enhanced}

def verify_resume(state: ResumeState) -> ResumeState:
    """Agent 6: Verify the enhanced resume for accuracy"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a resume fact checker and accuracy verifier. Your task is to:
1. Compare the original and enhanced resumes carefully
2. Identify any potential inaccuracies, exaggerations, or fabrications in the enhanced resume
3. Verify that all information in the enhanced resume is factually supported by either the original resume or could be reasonably inferred
4. If you find issues, provide corrections that maintain the improved quality while ensuring accuracy
5. If no issues are found, confirm the enhanced resume's accuracy

Be thorough in your verification. The final resume must be both improved AND accurate."""),
        HumanMessage(content=f"""Here is the original resume:
{state['resume_content']}

Here is the enhanced resume:
{state['enhanced_resume']}

Please verify the enhanced resume for accuracy and provide a corrected version if needed.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    verification = chain.invoke({})
    
    # Check if verification result contains a corrected resume
    # This is a simple heuristic - if the verification result is long, it likely contains a corrected resume
    if len(verification.split()) > 200:
        return {
            "verification_result": "The enhanced resume has been verified and corrected for accuracy.",
            "final_resume": verification
        }
    else:
        return {
            "verification_result": verification,
            "final_resume": state['enhanced_resume']
        }

def decide_next_step(state: ResumeState) -> str:
    """Decision function to determine the next step in the workflow"""
    # Initial workflow
    if "resume_analysis" not in state or not state["resume_analysis"]:
        return "analyze_resume"
    
    if "interview_questions" not in state or not state["interview_questions"]:
        return "generate_questions"
    
    # After interview is complete
    if "chat_history" in state and len(state["chat_history"]) > 3:  # Assuming interview is complete
        if "interview_insights" not in state or not state["interview_insights"]:
            return "generate_insights"
        
        if "enhanced_resume" not in state or not state["enhanced_resume"]:
            return "enhance_resume"
        
        if "verification_result" not in state or not state["verification_result"]:
            return "verify_resume"
        
        return END
    
    # Interview is not complete yet
    return "conduct_interview"  # This would be handled in the UI

# Build the graph
def build_resume_workflow():
    """Build the LangGraph workflow for resume enhancement"""
    # Initialize the graph
    workflow = StateGraph(ResumeState)
    
    # Add nodes
    workflow.add_node("analyze_resume", analyze_resume)
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("generate_insights", generate_insights)
    workflow.add_node("enhance_resume", enhance_resume)
    workflow.add_node("verify_resume", verify_resume)
    
    # Add edges
    workflow.add_conditional_edges(
        "",
        decide_next_step,
        {
            "analyze_resume": "analyze_resume",
            "generate_questions": "generate_questions",
            "generate_insights": "generate_insights",
            "enhance_resume": "enhance_resume",
            "verify_resume": "verify_resume",
            END: END
        }
    )
    
    workflow.add_edge("analyze_resume", "generate_questions")
    workflow.add_edge("generate_questions", "conduct_interview")  # This would be handled externally
    workflow.add_edge("generate_insights", "enhance_resume")
    workflow.add_edge("enhance_resume", "verify_resume")
    workflow.add_edge("verify_resume", END)
    
    # Compile the graph
    return workflow.compile()

# Example usage:
# resume_workflow = build_resume_workflow()
# initial_state = {"resume_content": "Your resume content here"}
# for event in resume_workflow.stream(initial_state, {'recursion_limit': 25}):
#     if event["type"] == "state":
#         # Process state updates
#         print(f"Step: {event['step']}")
#     elif event["type"] == "end":
#         final_state = event["state"]
#         print("Workflow complete!")
