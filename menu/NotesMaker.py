from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import re
from langchain_groq import ChatGroq
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from streamlit_lottie import st_lottie
import json
import PyPDF2
import docx2txt
from PIL import Image
import httpx

# Try to import pytesseract, but make it optional
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Constants for prompts
DETAILED_PROMPT = """
You are an expert educational content summarizer. Create comprehensive notes including:
1. TITLE: Create an appropriate title based on content
2. SUMMARY: A concise 2-3 sentence summary
3. KEY CONCEPTS: 4-6 main ideas with explanations
4. DETAILED NOTES: Organized structure with headings and bullet points
5. IMPORTANT TERMINOLOGY: List and define key terms
6. APPLICATION: 2-3 ways to apply this information
7. CONNECTIONS: How this connects to related topics

Format using proper Markdown with headings, bullet points, emphasis, and code blocks as needed.
"""

SIMPLE_PROMPT = """
Summarize this content into concise notes with:
- Key points and main takeaways
- Bullet points for easy reading
- Under 250 words total
"""

def load_animation():
    try:
        with open('src/Notes.json', encoding='utf-8') as anim_source:
            animation = json.load(anim_source)
        st_lottie(animation, 1, True, True, "high", 100, -200)
    except Exception:
        pass

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

def extract_video_id(youtube_url):
    if not youtube_url:
        return None
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com') and parsed_url.path.startswith('/embed/'):
        return parsed_url.path.split('/')[2]
    if re.match(r'^[A-Za-z0-9_-]{11}$', youtube_url):
        return youtube_url
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    return match.group(1) if match else None

def get_video_details(video_id):
    title = "YouTube Video"
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.text.replace(' - YouTube', '')
    except Exception:
        pass
    return {
        "title": title,
        "thumbnail": f"http://img.youtube.com/vi/{video_id}/0.jpg"
    }

def get_transcript_from_youtube_api(video_id):
    try:
        import yt_dlp
        url = f"https://www.youtube.com/watch?v={video_id}"

        ydl_opts = {
            'skip_download': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Try manual subtitles first, then auto-generated
            subtitles = info.get('subtitles', {})
            auto_captions = info.get('automatic_captions', {})

            captions = subtitles.get('en') or auto_captions.get('en')

            if captions:
                for fmt in captions:
                    if fmt.get('ext') == 'json3':
                        resp = requests.get(fmt['url'])
                        data = resp.json()
                        text = ' '.join(
                            event.get('segs', [{}])[0].get('utf8', '')
                            for event in data.get('events', [])
                            if event.get('segs')
                        )
                        return text.strip()
    except Exception as e:
        print(f"yt-dlp error: {e}")
        return None

def get_transcript_from_alternative_apis(video_id):
    try:
        response = httpx.get(
            f"https://yt-downloader-six.vercel.app/transcript?id={video_id}",
            timeout=10.0
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("transcript"):
                return " ".join(item["text"] for item in data["transcript"])
    except Exception:
        pass
    rapid_api_key = os.getenv("RAPID_API_KEY")
    if rapid_api_key:
        try:
            url = "https://youtube-transcriptor.p.rapidapi.com/transcript"
            headers = {
                "X-RapidAPI-Key": rapid_api_key,
                "X-RapidAPI-Host": "youtube-transcriptor.p.rapidapi.com"
            }
            params = {"video_id": video_id}
            response = requests.get(url, headers=headers, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                transcript = data.get("transcript", "")
                if transcript:
                    return transcript
        except Exception:
            pass
    return None

def get_transcript(video_id):
    transcript = get_transcript_from_youtube_api(video_id)
    if transcript:
        return transcript
    return get_transcript_from_alternative_apis(video_id)

def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        elif file_extension in ['docx', 'doc']:
            return docx2txt.process(uploaded_file)
        elif file_extension in ['txt', 'md']:
            return uploaded_file.read().decode('utf-8', errors='replace')
        elif file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
            if PYTESSERACT_AVAILABLE:
                image = Image.open(uploaded_file)
                return pytesseract.image_to_string(image)
            else:
                st.error("Image text extraction requires pytesseract.")
                return None
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
    except Exception as e:
        st.error(f"Error extracting text from file: {str(e)}")
        return None

def generate_notes(text, is_detailed=True):
    from langchain_core.messages import SystemMessage, HumanMessage
    
    llm = get_llm()
    max_tokens = 25000
    if len(text) > max_tokens:
        text = text[:max_tokens] + "...[text truncated due to length]"

    system = SystemMessage(content="""You are an expert educational content summarizer.
You ONLY output structured notes. You NEVER repeat or quote the input text.
Your response MUST start with '## ' heading directly. No preamble, no quoting source material.""")

    prompt = DETAILED_PROMPT if is_detailed else SIMPLE_PROMPT
    human = HumanMessage(content=f"{prompt}\n\nHere is the content to summarize:\n\n{text}")

    try:
        response = llm.invoke([system, human])
        result = response.content if hasattr(response, 'content') else str(response)
        
        # Find where the actual notes start (first ## heading)
        if '##' in result:
            result = result[result.index('##'):]
        
        return result
    except Exception as e:
        st.error(f"Error generating notes: {str(e)}")
        if is_detailed:
            st.info("Trying simplified notes generation...")
            return generate_notes(text[:5000], False)
        return None

def display_notes(notes, title="Content"):
    st.markdown("## Generated Notes")
    with st.expander("View Full Notes", expanded=True):
        st.markdown(notes)
    safe_title = re.sub(r'[^\w\s-]', '', title).strip()
    safe_title = re.sub(r'[-\s]+', '_', safe_title)
    st.download_button(
        label="Download Notes",
        data=notes,
        file_name=f"{safe_title[:50]}_notes.md",
        mime="text/markdown"
    )

def main():
    st.write("<h1><center>Notes Generator</center></h1>", unsafe_allow_html=True)
    load_animation()

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found in .env file.")
        return

    st.subheader("Transform Content into Comprehensive Study Notes")
    tab1, tab2 = st.tabs(["YouTube Video", "Upload File"])

    with tab1:
        st.markdown("### Generate Notes from YouTube Video")
        youtube_link = st.text_input("Enter YouTube video link", key="youtube_input")
        note_style_yt = st.selectbox("Note Style", ["Comprehensive", "Summary"], key="note_style_yt")

        if st.button("Generate Notes from Video", key="generate_yt"):
            if not youtube_link:
                st.warning("Please enter a YouTube URL")
            else:
                with st.spinner("Processing YouTube video..."):
                    try:
                        video_id = extract_video_id(youtube_link)
                        if not video_id:
                            st.error("Invalid YouTube URL format.")
                        else:
                            video = get_video_details(video_id)
                            st.markdown(f"### {video['title']}")
                            st.image(video['thumbnail'], use_container_width=True)
                            with st.spinner("Fetching transcript..."):
                                transcript = get_transcript(video_id)
                                if not transcript:
                                    st.warning("Could not fetch transcript automatically.")
                                    st.session_state.show_manual_input = True
                                else:
                                    st.session_state.show_manual_input = False
                                    st.success(f"Transcript obtained ({len(transcript.split())} words)")
                                    with st.spinner("Generating notes..."):
                                        notes = generate_notes(transcript, note_style_yt == "Comprehensive")
                                        if notes:
                                            display_notes(notes, video['title'])
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")

        # Manual transcript fallback
        if st.session_state.get("show_manual_input", False):
            st.markdown("---")
            st.info("""
            **How to get YouTube transcript manually:**
            1. Open the YouTube video
            2. Click **'...'** (three dots) below the video
            3. Click **'Show transcript'**
            4. Select all text and copy it
            5. Paste below
            """)
            manual_transcript = st.text_area(
                "Paste transcript here",
                height=200,
                key="manual_transcript"
            )
            if st.button("Generate Notes from Transcript", key="generate_manual"):
                if manual_transcript:
                    with st.spinner("Generating notes..."):
                        notes = generate_notes(manual_transcript, note_style_yt == "Comprehensive")
                        if notes:
                            st.session_state.show_manual_input = False
                            display_notes(notes, "YouTube Video")
                else:
                    st.warning("Please paste the transcript first.")

    with tab2:
        st.markdown("### Generate Notes from Uploaded File")
        supported_types = ["pdf", "docx", "doc", "txt", "md"]
        if PYTESSERACT_AVAILABLE:
            supported_types.extend(["jpg", "jpeg", "png"])
        uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=supported_types)
        note_style_file = st.selectbox("Note Style", ["Comprehensive", "Summary"], key="note_style_file")

        if st.button("Generate Notes from File", key="generate_file"):
            if uploaded_file is None:
                st.warning("Please upload a file first")
            else:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        text = extract_text_from_file(uploaded_file)
                        if text:
                            st.success(f"Text extracted: {len(text.split())} words")
                            with st.spinner("Generating notes..."):
                                notes = generate_notes(text, note_style_file == "Comprehensive")
                                if notes:
                                    display_notes(notes, uploaded_file.name)
                        else:
                            st.error("Could not extract text from the uploaded file.")
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()