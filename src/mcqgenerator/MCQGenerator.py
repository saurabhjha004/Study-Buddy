from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import json

load_dotenv()

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

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

def generate_mcq_with_fallback(input_data):
    llm = get_llm()

    # Truncate text to prevent model ignoring JSON instructions on long PDFs
    input_data = dict(input_data)
    input_data['text'] = input_data['text'][:3000]

    system = SystemMessage(content=(
        "You are a JSON-only MCQ generator. "
        "Your entire response must be a single valid JSON object. "
        "Do NOT repeat or quote the input text. "
        "Start with { and end with }. Nothing else."
    ))

    human = HumanMessage(content=quiz_generation_prompt.format(**input_data))
    response = llm.invoke([system, human])

    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)

class MCQChain:
    def invoke(self, input_data):
        return generate_mcq_with_fallback(input_data)

generate_evaluate_chain = MCQChain()