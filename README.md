Here’s a sample **README** file for your project:

---

# Study Buddy - AI-Powered Educational Platform

Study Buddy is an innovative, AI-powered educational platform that provides a set of powerful tools designed to enhance the learning experience. Whether you're preparing for a coding interview or just looking to optimize your resume for job applications, Study Buddy has got you covered with real-time MCQ generation, PDF querying, YouTube transcript-based note generation, contest tracking, and ATS resume optimization. Powered by advanced AI models like Google Gemini Pro and LangChain, the platform tailors resources to individual needs.

##Demo
https://www.youtube.com/watch?v=HD2UCnCWKJ8&t=31s

## Features

### 1. **Real-Time MCQ Generator**
Automatically generates multiple-choice questions (MCQs) from any uploaded text file (PDF or TXT). This feature uses Google Gemini Pro to create personalized quizzes based on complexity and subject preference.
- Upload a PDF or text file.
- Specify the number of MCQs, subject, and complexity level.
- Get a quiz with multiple-choice questions for study and review.

### 2. **Ask to PDF**
Query multiple PDFs and receive answers in real-time. This feature allows users to upload PDFs and ask questions related to the content, with the answers derived from the documents.
- Upload multiple PDFs.
- Ask questions related to the uploaded files.
- Get relevant answers by analyzing the content of the PDFs.

### 3. **YouTube Notes Maker**
Generates detailed notes from any YouTube video by analyzing the transcript, making it easier to study video content.
- Enter the YouTube video link.
- Extract the transcript and summarize it into notes.
- Review and export the notes for later study.

### 4. **Contest Calendar**
Stay up-to-date with upcoming coding contests across various competitive programming platforms such as Codeforces, LeetCode, CodeChef, and more. This feature automatically fetches and displays a calendar with the latest contests.
- View and track upcoming coding contests.
- Set reminders for contests you're interested in.

### 5. **ATS Resume Analyzer**
Optimize your resume for Applicant Tracking Systems (ATS). This feature analyzes your resume against a given job description, provides a score, and highlights missing keywords for better visibility.
- Upload your resume and job description.
- Get a detailed ATS score.
- Find missing keywords and get a profile summary.

## Tech Stack

- **Backend**: Python, LangChain, Google Gemini Pro
- **Frontend**: Streamlit
- **APIs & Tools**:
  - PyPDF2 (for PDF handling)
  - YouTubeTranscriptApi (for YouTube video transcript extraction)
  - OpenAI GPT (for AI-driven content generation)
  - Pandas (for data handling)
  - dotenv (for environment configuration)
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/study-buddy.git
   cd study-buddy
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add environment variables for APIs (e.g., Google Gemini Pro, LangChain) in a `.env` file.

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Navigate to the homepage where you can select from the various features like MCQ Generator, Ask to PDF, YouTube Notes Maker, Contest Calendar, and ATS Resume Analyzer.
- Follow the instructions for each feature to upload your files, input job descriptions, or track coding contests.

## Diagrams

Below is a flowchart showing the structure of the project:

![Flowchart](path/to/flowchart.png)

## Contributing

We welcome contributions! If you'd like to contribute, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

This **README** file provides detailed information about your project, explaining the purpose and how to use each feature, with proper guidance for installing and running the application.
