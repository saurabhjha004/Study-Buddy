from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Create a quiz generation prompt template
quiz_generation_template = """
You are an expert quiz creator specializing in creating multiple-choice questions (MCQs) for {subject} students at {tone} level.

Based on the following text, create {number} multiple-choice questions that test the understanding of key concepts.

TEXT: {text}

For each question:
1. Frame a clear question
2. Provide 4 options labeled as A, B, C, and D with only one correct answer
3. Indicate the correct answer

Format your response as a valid JSON object matching this exact structure:
{response_json}

IMPORTANT: Your response should ONLY contain the valid JSON object and nothing else - no introduction, no explanations. Just the JSON.
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=quiz_generation_template
)

# Initialize the LLM using Groq
def get_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    print("Using Groq LLM: llama-3.3-70b-versatile")
    return llm

def generate_mcq_with_fallback(input_data):
    llm = get_llm()
    response = llm.invoke(quiz_generation_prompt.format(**input_data))

    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)

class MCQChain:
    def invoke(self, input_data):
        return generate_mcq_with_fallback(input_data)

generate_evaluate_chain = MCQChain()