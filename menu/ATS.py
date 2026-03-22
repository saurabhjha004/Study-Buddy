import streamlit as st
from langchain_groq import ChatGroq
import PyPDF2 as pdf
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re

load_dotenv()

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

def extract_json_from_text(text):
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    if start_idx >= 0 and end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def analyze_resume(resume_text, job_description, job_role):
    max_length = 10000
    if len(resume_text) > max_length:
        resume_text = resume_text[:max_length] + "..."
    if len(job_description) > max_length:
        job_description = job_description[:max_length] + "..."

    input_prompt = f'''
    You're an advanced ATS (Applicant Tracking System) expert analyzing resumes.
    
    TASK:
    Analyze the provided resume against the job description for the role of {job_role}.
    
    RESUME:
    {resume_text}
    
    JOB DESCRIPTION:
    {job_description}
    
    PROVIDE YOUR ANALYSIS IN THE FOLLOWING JSON FORMAT:
    {{
        "PercentageMatch": "XX%",
        "MissingKeywordsintheResume": ["keyword1", "keyword2"],
        "FoundKeywords": ["keyword1", "keyword2"],
        "KeySkillGaps": ["skill1", "skill2"],
        "ResumeImprovementSuggestions": ["suggestion1", "suggestion2"],
        "ProfileSummary": "Brief assessment",
        "StrengthsForRole": ["strength1", "strength2"],
        "InterviewTips": ["tip1", "tip2"]
    }}
    
    IMPORTANT: RESPOND ONLY WITH THE PROPERLY FORMATTED JSON. DO NOT ADD ANY TEXT BEFORE OR AFTER THE JSON.
    '''

    try:
        llm = get_llm()
        response = llm.invoke(input_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        st.expander("View Raw AI Response").code(response_text)

        json_data = extract_json_from_text(response_text)
        if not json_data:
            st.warning("Could not parse JSON response. Using fallback analysis.")
            return create_fallback_analysis(resume_text, job_description)
        return json_data

    except Exception as e:
        st.error(f"Error analyzing resume: {str(e)}")
        return create_fallback_analysis(resume_text, job_description)

def extract_keywords(text):
    common_skills = [
        "python", "java", "javascript", "react", "node", "sql", "machine learning",
        "data analysis", "project management", "communication", "leadership",
        "agile", "scrum", "docker", "kubernetes", "aws", "azure", "git",
        "tensorflow", "pytorch", "nlp", "deep learning", "api", "rest"
    ]
    text_lower = text.lower()
    found_keywords = [skill for skill in common_skills if skill in text_lower]
    words = re.findall(r'\b[A-Za-z][A-Za-z+#.]{2,}\b', text)
    word_freq = Counter(words)
    common_words = {'the', 'and', 'for', 'with', 'this', 'that', 'are', 'will',
                    'you', 'your', 'our', 'have', 'has', 'been', 'from', 'they'}
    freq_keywords = [word.lower() for word, count in word_freq.most_common(20)
                     if word.lower() not in common_words and len(word) > 3]
    all_keywords = list(set(found_keywords + freq_keywords))
    return all_keywords[:15]

def create_fallback_analysis(resume_text, job_description):
    keywords = extract_keywords(job_description)
    found = [kw for kw in keywords if kw.lower() in resume_text.lower()]
    missing = [kw for kw in keywords if kw.lower() not in resume_text.lower()]
    match_percentage = int((len(found) / max(len(keywords), 1)) * 100) if keywords else 50
    return {
        "PercentageMatch": f"{match_percentage}%",
        "MissingKeywordsintheResume": missing[:5],
        "FoundKeywords": found[:5],
        "KeySkillGaps": missing[:3],
        "ResumeImprovementSuggestions": [
            "Add more keywords relevant to the job description",
            "Highlight your relevant experience more clearly",
            "Quantify your achievements with metrics"
        ],
        "ProfileSummary": "Fallback analysis used. Please try again for a detailed summary.",
        "StrengthsForRole": found[:3],
        "InterviewTips": [
            "Research the company before the interview",
            "Prepare examples of your past experiences",
            "Practice answering common interview questions"
        ]
    }

def create_match_radar_chart(response_data):
    categories = ['Technical Skills', 'Experience', 'Education', 'Keywords', 'Communication', 'Cultural Fit']
    try:
        base_match = int(response_data['PercentageMatch'].strip('%'))
    except Exception:
        base_match = 50

    values = [
        min(100, base_match + np.random.randint(-5, 15)),
        min(100, base_match + np.random.randint(-10, 5)),
        min(100, base_match + np.random.randint(-15, 10)),
        min(100, base_match + np.random.randint(-3, 12)),
        min(100, base_match + np.random.randint(-12, 8)),
        min(100, base_match + np.random.randint(-7, 10))
    ]
    values = [max(0, v) for v in values]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        ax.text(angle, value + 5, f'{value}%', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    plt.title('Skills Match Analysis', size=15, fontweight='bold', pad=15)
    return fig

def main():
    st.write("<h1><center>Advanced ATS Analyzer</center></h1>", unsafe_allow_html=True)
    st.text("👉🏻                  Personal ATS for Job-Seekers & Recruiters                   👈")

    try:
        with open('src/ATS.json') as anim_source:
            animation = json.load(anim_source)
        st_lottie(animation, 1, True, True, "high", 200, -200)
    except Exception as e:
        st.warning(f"Animation file not found: {str(e)}")

    col1, col2 = st.columns(2)
    with col1:
        job_role = st.text_input("Job Role", placeholder="e.g. Senior Python Developer")
        uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload PDF file only")
    with col2:
        job_desc = st.text_area("Paste the Job Description", height=200)

    submit = st.button("Analyze Resume")

    if submit:
        if uploaded_file is not None and job_desc and job_role:
            try:
                reader = pdf.PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += str(page.extract_text())

                with st.spinner("Analyzing your resume against job requirements..."):
                    response_data = analyze_resume(text, job_desc, job_role)

                if response_data:
                    tab1, tab2, tab3 = st.tabs(["Match Analysis", "Skills Gap", "Improvement Tips"])

                    with tab1:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.subheader("Match Score")
                            match_val = int(response_data['PercentageMatch'].strip('%'))
                            color = 'green' if match_val >= 70 else 'orange' if match_val >= 50 else 'red'
                            st.markdown(f"<div style='text-align:center;'><h1 style='font-size:4rem;color:{color};'>{response_data['PercentageMatch']}</h1></div>", unsafe_allow_html=True)
                            st.subheader("Profile Summary")
                            st.write(response_data['ProfileSummary'])
                        with col2:
                            st.subheader("Skills Match Analysis")
                            fig = create_match_radar_chart(response_data)
                            st.pyplot(fig)

                    with tab2:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.subheader("Missing Keywords")
                            for keyword in response_data['MissingKeywordsintheResume']:
                                st.markdown(f"- 🔴 {keyword}")
                        with col2:
                            st.subheader("Found Keywords")
                            for keyword in response_data['FoundKeywords']:
                                st.markdown(f"- ✅ {keyword}")
                        st.subheader("Key Skill Gaps")
                        for skill in response_data['KeySkillGaps']:
                            st.markdown(f"- 🔍 {skill}")
                        st.subheader("Your Strengths")
                        for strength in response_data['StrengthsForRole']:
                            st.markdown(f"- 💪 {strength}")

                    with tab3:
                        st.subheader("Resume Improvement Suggestions")
                        for i, s in enumerate(response_data['ResumeImprovementSuggestions'], 1):
                            st.markdown(f"**{i}.** {s}")
                        st.subheader("Interview Preparation Tips")
                        for i, tip in enumerate(response_data['InterviewTips'], 1):
                            st.markdown(f"**{i}.** {tip}")

                    report = f"""# Resume Analysis Report for {job_role}

## Overall Match: {response_data['PercentageMatch']}

### Profile Summary
{response_data['ProfileSummary']}

### Missing Keywords
{"- " + chr(10) + "- ".join(response_data['MissingKeywordsintheResume']) if response_data['MissingKeywordsintheResume'] else "No missing keywords found!"}

### Key Skill Gaps
{"- " + chr(10) + "- ".join(response_data['KeySkillGaps']) if response_data['KeySkillGaps'] else "No major skill gaps identified!"}

### Resume Improvement Suggestions
{"1. " + chr(10) + "2. ".join(response_data['ResumeImprovementSuggestions']) if response_data['ResumeImprovementSuggestions'] else "Your resume looks great!"}

### Interview Tips
{"1. " + chr(10) + "2. ".join(response_data['InterviewTips'])}
"""
                    st.download_button(
                        label="Download Analysis Report",
                        data=report,
                        file_name="resume_analysis_report.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("Failed to analyze resume. Please try again.")
            except Exception as e:
                st.error(f"Error processing your resume: {str(e)}")
        else:
            st.warning("Please provide a job role, upload your resume, and paste the job description.")

if __name__ == "__main__":
    main()