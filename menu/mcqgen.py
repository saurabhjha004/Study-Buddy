import os
import json
import PyPDF2
import pandas as pd
import traceback
from dotenv import load_dotenv
import streamlit as st
from src.mcqgenerator.utils import read_file
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

# Load Response JSON
with open("Response.json", 'r') as file:
    RESPONSE_JSON = json.load(file)

def main():
    st.title("Real-Time MCQ Creator with LangChain & Google Gemini Pro")
    st.title("QuizCraft: The MCQ Genie")

    # Initialize session state
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'show_error' not in st.session_state:
        st.session_state.show_error = False

    def process_quiz_data(quiz_json):
        """Convert the nested JSON structure to a list of questions."""
        processed_data = []
        try:
            quiz_dict = json.loads(quiz_json) if isinstance(quiz_json, str) else quiz_json
            for question_num, question_data in quiz_dict.items():
                processed_data.append({
                    'question_num': question_num,
                    'mcq': question_data['mcq'],
                    'options': question_data['options'],
                    'correct': question_data['correct']
                })
        except Exception as e:
            st.error(f"Error processing quiz data: {e}")
        return processed_data

    def calculate_score():
        correct_answers = sum(1 for i, question in enumerate(st.session_state.quiz_data)
                              if st.session_state.user_answers.get(i, '')[:1] == question['correct'])
        return correct_answers, len(st.session_state.quiz_data)

    def check_answers_complete():
        """Check if all questions have been answered."""
        return all(ans != 'Select an option' for ans in st.session_state.user_answers.values())

    # File Upload and Quiz Generation
    if not st.session_state.quiz_data:
        with st.form("user_inputs"):
            uploaded_file = st.file_uploader("Upload a PDF or TXT file")
            mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50, value=5)
            subject = st.text_input("Insert Subject", max_chars=20)
            tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")
            button = st.form_submit_button("Create MCQs")

            if button and uploaded_file and subject and tone:
                with st.spinner("Generating MCQs..."):
                    try:
                        text = read_file(uploaded_file)
                        
                        # Adding debugging info
                        st.write("Processing text from file...")
                        
                        # Create input for the model
                        input_data = {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                        
                        # Get response from the model
                        try:
                            response = generate_evaluate_chain.invoke(input_data)
                            
                            # Debug response type
                            st.write(f"Response type: {type(response)}")
                            
                            # Extract content from AIMessage object - Multiple approaches for robustness
                            quiz_text = None
                            
                            # Try different methods to extract content
                            if hasattr(response, 'content'):
                                quiz_text = response.content
                            elif hasattr(response, 'text'):
                                quiz_text = response.text
                            elif hasattr(response, 'response'):
                                quiz_text = response.response
                            elif isinstance(response, dict) and 'content' in response:
                                quiz_text = response['content']
                            elif isinstance(response, dict) and 'text' in response:
                                quiz_text = response['text']
                            else:
                                # Last resort: convert to string
                                quiz_text = str(response)
                                st.write("Used string conversion as fallback")
                            
                            # Show extracted text for debugging
                            st.write("Extracted quiz text from model response")
                            
                            # Extract JSON from the response text
                            try:
                                # Find JSON in the text
                                if quiz_text:
                                    # Try direct JSON parsing first
                                    try:
                                        json_data = json.loads(quiz_text)
                                        quiz_json = json.dumps(json_data) # Convert back to ensure valid JSON
                                        st.write("Successfully parsed JSON directly")
                                    except json.JSONDecodeError:
                                        # Fallback: extract JSON using string methods
                                        st.write("Direct JSON parsing failed, trying string extraction")
                                        start_idx = quiz_text.find('{')
                                        end_idx = quiz_text.rfind('}') + 1
                                        
                                        if start_idx >= 0 and end_idx > start_idx:
                                            quiz_json = quiz_text[start_idx:end_idx]
                                            st.write(f"Extracted JSON from indexes {start_idx} to {end_idx}")
                                        else:
                                            st.error("No JSON delimiters found in the response")
                                            st.code(quiz_text)
                                            raise ValueError("Could not find JSON in response")
                                    
                                    # Process the extracted JSON
                                    st.session_state.quiz_data = process_quiz_data(quiz_json)
                                    st.success(f"Successfully created {len(st.session_state.quiz_data)} MCQs!")
                                else:
                                    st.error("No text extracted from the model response")
                            except Exception as e:
                                st.error(f"Error extracting JSON from response: {str(e)}")
                                st.code(quiz_text)  # Show the raw response for debugging
                        except Exception as e:
                            st.error(f"Error invoking model: {str(e)}")
                            st.error(traceback.format_exc())
                                
                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")
                        st.error(traceback.format_exc())

    # Quiz Display
    if st.session_state.quiz_data and not st.session_state.quiz_submitted:
        st.subheader("Answer the following questions:")
        
        # Display error message if needed
        if st.session_state.show_error:
            st.error("Please answer all questions before submitting.")
            st.session_state.show_error = False
        
        with st.form("quiz_form"):
            for i, question in enumerate(st.session_state.quiz_data):
                st.markdown(f"**Q{i+1}. {question['mcq']}**")
                
                options = ['Select an option'] + [f"{key}) {val}" for key, val in question['options'].items()]
                selected_option = st.radio(f"Select your answer for Q{i+1}:", options, key=f"q_{i}", index=0)
                st.session_state.user_answers[i] = selected_option
            
            submit_quiz = st.form_submit_button("Submit Quiz")
            if submit_quiz:
                if check_answers_complete():
                    st.session_state.quiz_submitted = True
                else:
                    st.session_state.show_error = True
                    st.experimental_rerun()

    # Quiz Results
    if st.session_state.quiz_submitted:
        correct_answers, total_questions = calculate_score()
        st.session_state.score = (correct_answers / total_questions) * 100

        st.subheader("Quiz Results")
        st.write(f"Your Score: {st.session_state.score:.2f}%")
        st.write(f"Correct Answers: {correct_answers}/{total_questions}")

        # Show Correct Answers
        st.subheader("Detailed Review")
        for i, question in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}. {question['mcq']}**")
            for opt_key, opt_value in question['options'].items():
                if opt_key == question['correct']:
                    st.markdown(f"- {opt_key}) {opt_value} ✅ (Correct)")
                elif opt_key == st.session_state.user_answers[i][:1]:  # Compare first character
                    st.markdown(f"- {opt_key}) {opt_value} ❌ (Your Answer)")
                else:
                    st.markdown(f"- {opt_key}) {opt_value}")
            st.markdown("---")

        # Reset Button
        def reset_quiz():
            st.session_state.quiz_data = None
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.score = 0
            st.session_state.show_error = False

        if st.button("Start New Quiz"):
            reset_quiz()
            st.experimental_rerun()

if __name__ == "__main__":
    main()
