# Resume Enhancement System

This application uses a multi-agent AI system powered by LangChain and LangGraph to analyze, interview, and enhance resumes. The system uses six specialized AI agents that work together to create an improved resume based on the original document and information gathered through an interactive interview.

## Features

- **Resume Analysis**: Identifies gaps, weaknesses, and areas for improvement in your resume
- **AI Interview**: Conducts a natural conversation to uncover valuable information missing from your resume
- **Resume Enhancement**: Creates an improved version of your resume with added details and stronger language
- **Fact Verification**: Ensures the enhanced resume remains factual and accurate
- **Export Options**: Download the enhanced resume as TXT or PDF

## How It Works

This system uses six specialized AI agents that work in sequence:

1. **Resume Analyzer**: Evaluates the uploaded resume for gaps and weaknesses
2. **Interview Questioner**: Creates tailored questions based on the analyzer's insights
3. **Chat Interviewer**: Conducts a conversational interview to gather additional information
4. **Insights Generator**: Processes the interview to extract key insights
5. **Resume Enhancer**: Creates an improved resume using the original document and interview insights
6. **Fact Checker**: Verifies the enhanced resume for accuracy and makes corrections if necessary

## Requirements

- Python 3.9+
- OpenAI API key (gpt-4o model access required)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter your OpenAI API key in the sidebar
2. Upload your resume (PDF or TXT format)
3. Review the analysis of your resume
4. Complete the interactive interview
5. Review the insights extracted from your interview
6. Download your enhanced resume in your preferred format

## Architecture

The application is built using:

- **Streamlit**: For the web interface
- **LangChain**: For agent creation and orchestration
- **LangGraph**: For workflow management (alternative implementation)
- **OpenAI GPT-4o**: For the underlying language model

## Privacy and Security

- Your resume and API key are not stored permanently
- Data is processed in memory and not shared with third parties
- Your OpenAI API key is used only for the duration of your session

## Limitations

- The quality of enhancements depends on the information provided in the original resume and during the interview
- The system requires access to OpenAI's GPT-4o model
- Processing may take time, especially for longer resumes or interviews

## Future Improvements

- Support for more resume formats (DOCX, HTML, etc.)
- Additional customization options for resume style and format
- Integration with job descriptions for targeted enhancements
- Local language model support
