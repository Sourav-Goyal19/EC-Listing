import os
import fitz
import streamlit as st
import pandas as pd
import plotly.express as px
from phi.agent import Agent
from dotenv import load_dotenv
from phi.model.anthropic.claude import Claude
from phi.tools.website import WebsiteTools
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv("CLAUDE_API_KEY")


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])


def custom_slider(label, min_val, max_val, default_val, key):
    col1, col2 = st.columns([3, 1])
    with col1:
        value = st.slider(label, min_val, max_val, default_val, key=key)
    with col2:
        st.write(f"{value:.2f}")
    return value


def create_weight_sliders():
    st.subheader("Match Priority Weights")
    return {
        "skills": custom_slider("Skills", 0.0, 1.0, 0.3, "skills_slider"),
        "experience": custom_slider("Experience", 0.0, 1.0, 0.25, "experience_slider"),
        "salary": custom_slider("Salary Fit", 0.0, 1.0, 0.2, "salary_slider"),
        "location": custom_slider("Location", 0.0, 1.0, 0.15, "location_slider"),
        "education": custom_slider("Education", 0.0, 1.0, 0.1, "education_slider"),
    }


scraper_agent = Agent(
    model=Claude(
        id="claude-3-5-sonnet-20240620",
        api_key=os.getenv("CLAUDE_API_KEY"),
    ),
    tools=[WebsiteTools()],
    show_tool_calls=True,
    markdown=True,
    structured_outputs=True,
)

matcher_agent = Agent(
    model=Claude(
        id="claude-3-5-sonnet-20240620",
        api_key=os.getenv("CLAUDE_API_KEY"),
    ),
    show_tool_calls=True,
    markdown=True,
    structured_outputs=True,
)

finalizer_agent = Agent(
    model=Claude(
        id="claude-3-5-sonnet-20240620",
        api_key=os.getenv("CLAUDE_API_KEY"),
    ),
    show_tool_calls=True,
    markdown=True,
    structured_outputs=True,
)

test_creator_agent = Agent(
    model=Claude(
        id="claude-3-5-sonnet-20240620",
        api_key=os.getenv("CLAUDE_API_KEY"),
    ),
    show_tool_calls=True,
    markdown=True,
    structured_outputs=True,
)


def generate_test_questions(job_description):
    prompt = f"""Generate 5-7 technical interview questions based on these job requirements:
    {job_description}
    
    Return the questions in JSON format:
    {{"questions": ["question1", "question2", ...]}}"""
    return test_creator_agent.run(prompt).content


def create_google_form(service, job_title, questions):
    form = {
        "info": {
            "title": f"{job_title} Skills Assessment",
            "documentTitle": f"{job_title} Test",
        }
    }

    items = [
        {
            "questionItem": {
                "question": {
                    "required": True,
                    "textQuestion": {"paragraph": False},
                    "questionId": str(i),
                }
            },
            "title": q,
        }
        for i, q in enumerate(questions.get("questions", []))
    ]

    request = service.forms().create(body={"info": form["info"]})
    result = request.execute()
    service.forms().batchUpdate(
        formId=result["formId"],
        body={
            "requests": [
                {"createItem": {"item": item, "location": {"index": i}}}
                for i, item in enumerate(items)
            ]
        },
    ).execute()
    return f"https://docs.google.com/forms/d/{result['formId']}/edit"


def get_forms_service():
    creds_json = json.loads(os.getenv("GOOGLE_CREDENTIALS_JSON"))
    credentials = service_account.Credentials.from_service_account_info(
        creds_json, scopes=["https://www.googleapis.com/auth/forms.body"]
    )
    return build("forms", "v1", credentials=credentials)


st.set_page_config(page_title="AI-Powered Job Matching System", layout="wide")
st.title("AI-Powered Job Matching System")

with st.expander("📥 Upload Requirements", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.header("Job Requirements")
        job_text = st.text_area(
            "Enter job requirements (or upload PDF below):", height=200
        )
        job_pdf = st.file_uploader("Upload Job PDF", type="pdf", key="job_pdf")
        if job_pdf:
            job_text = extract_text_from_pdf(job_pdf)

    with col2:
        st.header("Candidate Profiles")
        candidate_text = st.text_area(
            "Enter candidate profiles (or upload PDF below):", height=200
        )
        candidate_pdf = st.file_uploader(
            "Upload Candidate PDF", type="pdf", key="candidate_pdf"
        )
        if candidate_pdf:
            candidate_text = extract_text_from_pdf(candidate_pdf)

weights = create_weight_sliders()

if st.button("🚀 Rank Candidates", type="primary"):
    if not job_text or not candidate_text:
        st.error("Please provide both job requirements and candidate profiles.")
    else:
        with st.spinner("Analyzing matches..."):
            scrape_task = f"""
                Extract additional candidate insights from provided URLs.
                **Candidate Profiles:**
                {candidate_text}

                Your task:
                1. Identify candidate URLs in the provided profiles.
                2. Scrape relevant data from those URLs.
                3. Summarize the additional insights gained from each candidate.
            """
            scraped_data = scraper_agent.run(scrape_task)

            match_task = f"""
                Determine candidate-job role matches based on provided job requirements and profiles.

                **Job Requirements:**
                {job_text}

                **Candidate Profiles:**
                {candidate_text}

                Your task:
                1. Analyze the provided job requirements and candidate profiles.
                2. Match each candidate to the most suitable job role based on:
                - **Skills**
                - **Experience**
                - **Salary Fit**
                - **Location**
                - **Education**
                - **Weight Priorities:** {weights}
                3. Generate a ranked list of candidates for each job role.
            """
            matched_candidates = matcher_agent.run(match_task)

            final_task = f"""
                Generate a structured ranking based on the insights provided by both agents.

                **Scraped Data from Candidate URLs:**
                {scraped_data}

                **Matched Candidates:**
                {matched_candidates}

                Your task:
                1. Combine insights from the scraped data and the candidate-job matching results.
                2. Generate a final ranked list of candidates for each job role in the following format:

                ## Top Candidates for Each Job Role
                - **Job Role 1:**
                | Rank | Name | Score | Reasoning |
                |------|------|-------|-----------|

                - **Job Role 2:**
                | Rank | Name | Score | Reasoning |
                |------|------|-------|-----------|

                3. Ensure the output is clear and actionable for HR decision-makers.
            """
            response = finalizer_agent.run(final_task)

            try:
                st.session_state.ranking_data = json.loads(response.content)
            except:
                st.session_state.ranking_data = None

            st.subheader("🔍 Matching Insights")
            st.markdown(response.content)
            st.download_button(
                "📥 Download Full Report", "ranked_candidates.md", "text/markdown"
            )

if "ranking_data" in st.session_state:
    st.subheader("📝 Generate Assessment Tests")

    for job in st.session_state.ranking_data.get("job_roles", []):
        with st.expander(f"Tests for {job['role_name']}"):
            if st.button(
                f"Create Test for {job['role_name']}", key=f"test_{job['role_name']}"
            ):
                with st.spinner("Generating assessment..."):
                    try:
                        questions = generate_test_questions(job_text)
                        service = get_forms_service()
                        form_url = create_google_form(
                            service, job["role_name"], questions
                        )
                        st.success(f"Test created: [Google Form Link]({form_url})")
                    except Exception as e:
                        st.error(f"Error creating form: {str(e)}")

# Footer
st.markdown("---")
st.caption(
    "💡 Tip: Adjust the weight sliders to prioritize different matching factors based on your needs!"
)
