import google.generativeai as genai
import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Load API Key (Replace with your actual API key in production)
# Replace the entire line with:
GEMINI_API_KEY = "AIzaSyBl2fvOFG0xDiJN1KzYRbqTqOEa8MOSB_k"  # Paste between the quotes

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Project and App Information
PROJECT_INFO = {
    "name": "NeuroVision AI",
    "version": "2.1.0",
    "description": "An advanced medical imaging analysis platform for diabetic retinopathy and brain tumor detection.",
    "features": [
        "Diabetic retinopathy classification (5 stages)",
        "Brain tumor detection (4 types)",
        "Comprehensive diagnostic reports",
        "AI-powered analysis with expert validation",
        "Secure data handling with HIPAA compliance"
    ],
    "team": "NeuroVision Medical Technologies",
    "contact": "support@neurovision.ai"
}

# Define comprehensive medical information database
MEDICAL_INFO = {
    # [Previous medical info dictionary remains exactly the same...]
}

# FAQ database for common questions
FAQ_DATABASE = {
    # [Previous FAQ database remains exactly the same...]
}

# General knowledge topics for fallback responses
GENERAL_KNOWLEDGE = {
    "weather": "I can't provide real-time weather data, but I can explain weather patterns or meteorological concepts.",
    "news": "I don't have live news access, but I can discuss historical events or analyze news topics you specify.",
    "sports": "I can provide information about sports rules, history, or famous athletes, but not live scores.",
    "entertainment": "I can discuss movies, books, or music in general terms, but don't have access to current releases.",
    "technology": "I can explain technology concepts, compare devices, or discuss tech history and trends."
}

# Cache for frequent responses to improve performance
response_cache = {}

# Initialize the AI Model with optimized parameters
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,  # Increased for more comprehensive responses
    }
)

def generate_system_prompt() -> str:
    """Generate a comprehensive system prompt combining project info and medical knowledge."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return f"""
    # NeuroVision AI Assistant - System Prompt
    ## Project Information
    Name: {PROJECT_INFO['name']}
    Version: {PROJECT_INFO['version']}
    Description: {PROJECT_INFO['description']}
    Features: {", ".join(PROJECT_INFO['features'])}
    Team: {PROJECT_INFO['team']}
    Contact: {PROJECT_INFO['contact']}
    Current Date: {current_date}

    ## Medical Knowledge Base
    You have access to detailed information about:
    - Diabetic Retinopathy (5 classifications)
    - Brain Tumors (4 classifications)
    
    ## Role and Guidelines
    You are a compassionate and knowledgeable AI assistant for NeuroVision with these capabilities:
    1. Provide medically accurate yet easy-to-understand explanations
    2. Offer reassurance and support for health concerns
    3. Clearly emphasize that results are preliminary and must be reviewed by professionals
    4. Use non-diagnostic language ("the analysis suggests" rather than definitive statements)
    5. Handle general knowledge queries when not medical-related
    6. Provide information about the NeuroVision platform and its features
    
    ## Response Formatting Guidelines
    - Use Markdown for formatting (headers, bold, lists)
    - Include disclaimers when discussing medical topics
    - Break long responses into readable paragraphs
    - Highlight important warnings or recommendations
    
    ## General Knowledge Handling
    For non-medical queries:
    - Acknowledge the query type
    - Provide general information if possible
    - Redirect to appropriate resources when needed
    - Be clear about limitations
    
    Always maintain a professional yet approachable tone.
    """

# Enhanced system prompt
SYSTEM_PROMPT = generate_system_prompt()

# Start a chat session to maintain context
chat = model.start_chat(
    history=[
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": [f"I am {PROJECT_INFO['name']} (v{PROJECT_INFO['version']}), your medical diagnosis and general knowledge assistant. How can I help you today?"]}
    ]
)

def get_detailed_condition_info(condition_key: str) -> str:
    """[Previous implementation remains exactly the same...]"""

def extract_condition_keywords(message: str) -> List[str]:
    """[Previous implementation remains exactly the same...]"""

def match_faq(message: str) -> Optional[str]:
    """[Previous implementation remains exactly the same...]"""

def get_app_functionality_info(query: str) -> Optional[str]:
    """[Previous implementation remains exactly the same...]"""

def is_general_knowledge_query(message: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if the query is a general knowledge question and return appropriate response.
    
    Args:
        message: User input message
        
    Returns:
        Tuple: (is_general_knowledge, response_or_none)
    """
    message = message.lower()
    
    for topic, response in GENERAL_KNOWLEDGE.items():
        if re.search(rf'\b{topic}\b', message):
            return (True, response)
    
    # Common general questions
    general_phrases = {
        "who created you": f"I was developed by {PROJECT_INFO['team']} as part of the {PROJECT_INFO['name']} project.",
        "what can you do": f"I can help with:\n- Medical image analysis information\n- {PROJECT_INFO['description']}\n- General knowledge questions\n\nMy features include: {', '.join(PROJECT_INFO['features'])}",
        "how old are you": f"I'm an AI assistant for {PROJECT_INFO['name']} version {PROJECT_INFO['version']}, first released in 2024.",
        "tell me about yourself": f"I'm {PROJECT_INFO['name']}, an AI assistant designed to provide information about medical diagnostics and general knowledge. {PROJECT_INFO['description']}"
    }
    
    for phrase, response in general_phrases.items():
        if phrase in message:
            return (True, response)
    
    return (False, None)

def enhance_response(original_response: str, is_medical: bool = True) -> str:
    """
    Enhance AI response with improved formatting and additional context.
    
    Args:
        original_response: Original AI response
        is_medical: Whether the response is medical-related
        
    Returns:
        Enhanced response with better formatting
    """
    # Basic cleaning
    enhanced = original_response.strip()
    
    # Add proper Markdown formatting
    if not enhanced.startswith("#"):
        enhanced = f"## Response\n{enhanced}"
    
    # Medical-specific enhancements
    if is_medical:
        # Add section breaks for better readability
        enhanced = re.sub(r"(\n)([A-Z][a-z]+:)", r"\n\n**\2**", enhanced)
        
        # Ensure disclaimer is present if not already there
        if "Disclaimer:" not in enhanced:
            enhanced += "\n\n**Disclaimer:** This information is for educational purposes only and not a substitute for professional medical advice. Always consult with qualified healthcare providers."
    
    # General formatting improvements
    enhanced = re.sub(r"(\n\s*\n)", r"\n\n", enhanced)  # Remove extra newlines
    enhanced = re.sub(r"\.(\s)([A-Z])", r".\n\n\2", enhanced)  # Paragraph breaks
    
    # Add project footer for longer responses
    if len(enhanced.split()) > 50:  # Only for substantial responses
        enhanced += f"\n\n---\n*{PROJECT_INFO['name']} v{PROJECT_INFO['version']}*"
    
    return enhanced

def generate_combined_prompt(user_query: str, context: Dict[str, Any] = None) -> str:
    """
    Generate a comprehensive prompt combining user query with system context.
    
    Args:
        user_query: The user's input message
        context: Additional context dictionary (e.g., current diagnosis)
        
    Returns:
        Combined prompt string for the AI model
    """
    prompt_parts = []
    
    # 1. Current context
    if context:
        prompt_parts.append(f"## Current Context\n{json.dumps(context, indent=2)}")
    
    # 2. User query
    prompt_parts.append(f"## User Query\n{user_query}")
    
    # 3. Instructions
    instructions = """
    Please provide a comprehensive response considering:
    - The user's specific query
    - Our medical knowledge base (when relevant)
    - Project information (when relevant)
    - Appropriate disclaimers
    - Clear, formatted output
    """
    prompt_parts.append(f"## Response Guidelines\n{instructions}")
    
    return "\n\n".join(prompt_parts)

def chatbot_response(message: str) -> str:
    """
    Process user input and return enhanced chatbot response.
    
    Args:
        message: User input message
        
    Returns:
        Formatted AI response with appropriate context
    """
    try:
        # Check cache first for performance
        cache_key = message.lower().strip()
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        # Check for general knowledge queries first
        is_general, general_response = is_general_knowledge_query(message)
        if is_general and general_response:
            enhanced = enhance_response(general_response, is_medical=False)
            response_cache[cache_key] = enhanced
            return enhanced
        
        # Extract condition keywords from message
        condition_keywords = extract_condition_keywords(message)
        
        # If specific condition mentioned, provide detailed information
        if condition_keywords:
            condition = condition_keywords[0]
            response = get_detailed_condition_info(condition)
            response_cache[cache_key] = response
            return response
        
        # Check for FAQ matches
        faq_response = match_faq(message)
        if faq_response:
            response_cache[cache_key] = faq_response
            return faq_response
        
        # Check for app functionality questions
        app_info = get_app_functionality_info(message)
        if app_info:
            response_cache[cache_key] = app_info
            return app_info
        
        # Get any relevant context (e.g., current diagnosis)
        context = {
            "latest_diagnosis": get_latest_diagnosis_result(),
            "query_time": datetime.now().isoformat()
        }
        
        # Generate comprehensive prompt
        combined_prompt = generate_combined_prompt(message, context)
        
        # Get AI response
        response = chat.send_message(combined_prompt)
        
        # Enhance the response
        enhanced_response = enhance_response(response.text)
        
        # Cache the response
        if len(response_cache) > 100:  # Limit cache size
            response_cache.popitem()  # Remove oldest entry
        response_cache[cache_key] = enhanced_response
        
        return enhanced_response
        
    except Exception as e:
        print(f"Chatbot Error: {str(e)}")
        error_msg = f"I'm experiencing technical difficulties. Please try again later. \n\n*Error details: {str(e)}*"
        return enhance_response(error_msg, is_medical=False)

def get_latest_diagnosis_result() -> Dict[str, Any]:
    """[Previous implementation remains exactly the same...]"""

# Example usage
if __name__ == "__main__":
    print("NeuroVision AI Assistant - Interactive Mode")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = chatbot_response(user_input)
        print("\nAssistant:")
        print(response)
        print("\n" + "-"*50 + "\n")