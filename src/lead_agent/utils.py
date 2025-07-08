import os
import json
from typing import Tuple, List, Dict, Any, Optional # Added Optional here as it's used later

# Import the main 'google.generativeai' module and alias it
# import google.generativeai as genai # Standard import commented out
import importlib
try:
    genai = importlib.import_module("google.generativeai")
except ImportError:
    print("ERROR: Failed to import 'google.generativeai' using importlib.")
    raise

# Remove: from google.generativeai import GenerativeModel
# Remove: from google.genai import types as genai_types
# Remove: from agent.utils import genai_client

from .configuration import LeadIdentificationConfiguration


def parse_gemini_json_response(response: Any) -> Tuple[List[Dict], str]:
    """
    Parses Gemini response, expecting a JSON string within the text part.
    Extracts the JSON and the raw text.
    """
    raw_text = ""
    if response.parts:
        for part in response.parts:
            if hasattr(part, 'text'):
                raw_text += part.text + "\n"
            # Handle potential function calls if the model tries to use tools for structured output
            if hasattr(part, 'function_call') and part.function_call:
                # This is a simplified example. Real handling might be more complex.
                # Assuming the function call args contain the JSON we want.
                args = part.function_call.args
                if args and 'leads_json' in args: # Example: expecting JSON under 'leads_json' key
                    try:
                        # If args['leads_json'] is already a dict/list
                        if isinstance(args['leads_json'], (list, dict)):
                             return args['leads_json'], raw_text.strip()
                        # If it's a string that needs parsing
                        return json.loads(args['leads_json']), raw_text.strip()
                    except json.JSONDecodeError as e:
                        print(f"JSON Decode Error in function call: {e}")
                        print(f"Problematic JSON string: {args['leads_json']}")
                        return [], f"Error: Could not parse JSON from function call. {raw_text.strip()}"

    # Fallback: try to find JSON in the raw text
    # This is a common pattern: Gemini might output markdown with a json block
    try:
        # Look for ```json ... ```
        json_block_start = raw_text.find("```json")
        if json_block_start != -1:
            json_block_end = raw_text.find("```", json_block_start + 6)
            if json_block_end != -1:
                json_str = raw_text[json_block_start + 6 : json_block_end].strip()
                return json.loads(json_str), raw_text.strip()

        # If no markdown block, try to parse the whole text (less reliable)
        return json.loads(raw_text.strip()), raw_text.strip()
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error in raw text: {e}")
        print(f"Problematic text: {raw_text.strip()}")
        return [], f"Error: Could not parse JSON from response. Raw text: {raw_text.strip()}"
    except Exception as e:
        print(f"Generic error parsing Gemini response: {e}")
        return [], f"Error: An unexpected error occurred during parsing. Raw text: {raw_text.strip()}"


def generate_lead_identification_prompt(company_name: str, lead_generation_area: str, titles: List[str]) -> str:
    """Generates the prompt for the lead identification Gemini call."""
    titles_str = ", ".join(f"'{title}'" for title in titles)
    prompt = f"""
Identify potential leads at the company "{company_name}" based on the following criteria:
Area/Department: "{lead_generation_area}"
Titles: {titles_str}

Please search for individuals matching these roles. For each lead, provide the following information in a structured JSON format within a list, where each item is an object:
- name (string, full name)
- title (string, current title at the company)
- email (string, professional email if available, otherwise null)
- linkedin_url (string, LinkedIn profile URL if available, otherwise null)
- summary (string, a brief summary of why they are a relevant lead based on the criteria)

Example of expected JSON output format:
```json
[
  {{
    "name": "Jane Doe",
    "title": "Senior Software Engineer",
    "email": "jane.doe@example.com",
    "linkedin_url": "https://linkedin.com/in/janedoe",
    "summary": "Matches title 'Senior Software Engineer' in the software engineering area."
  }},
  {{
    "name": "John Smith",
    "title": "Engineering Manager",
    "email": null,
    "linkedin_url": "https://linkedin.com/in/johnsmith",
    "summary": "Matches title 'Engineering Manager' and oversees teams relevant to the lead generation area."
  }}
]
```
Ensure the output is only the JSON list. Do not include any other explanatory text outside the JSON structure itself.
Use Google Search to find this information.
"""
    return prompt

def generate_summary_report_prompt(company_name: str, lead_generation_area: str, titles: List[str], leads: List[Dict]) -> str:
    """Generates the prompt for the summary report Gemini call."""
    titles_str = ", ".join(f"'{title}'" for title in titles)
    leads_summary = []
    if leads:
        for lead in leads:
            leads_summary.append(f"- {lead.get('name', 'N/A')} ({lead.get('title', 'N/A')})")
    else:
        leads_summary.append("No leads were identified.")

    leads_str = "\n".join(leads_summary)

    prompt = f"""
Generate a brief summary report for the lead identification task with the following parameters:
Company Searched: "{company_name}"
Lead Generation Area: "{lead_generation_area}"
Titles Sought: {titles_str}

Identified Leads:
{leads_str}

Based on this information, provide a concise report (2-3 paragraphs) summarizing the findings.
If leads were found, comment on the quality or relevance if possible (based on the data provided).
If no leads were found, suggest potential reasons or next steps.
"""
    return prompt

# Placeholder for Gemini client initialization if not using the global one
# Ensure GEMINI_API_KEY is set in your environment
# genai_client = GenerativeModel() # Or however it's initialized in the project

# Example of how genai_client is used in the research agent (for reference):
# from google.genai import GenerativeModel as GenAIModel # In actual code it's from agent.utils import genai_client
# genai_client.models.generate_content(...)
# For this new agent, we'll assume `genai_client` is available similarly.
# If it's `genai.configure(api_key=...)` and then `model = genai.GenerativeModel(...)`,
# we'd wrap that. The current research agent uses `genai_client.models...` which is a bit unusual.
# Let's assume `genai_client` is an instance of `google.generativeai.GenerativeModel` or a compatible client.
# The research agent's genai_client seems to be the top-level module, so `genai_client.GenerativeModel()` is used.
# For simplicity and consistency with the existing agent, we will use the imported `genai_client` which refers to `google.generativeai` module.

def get_gemini_model(model_name: str) -> genai.GenerativeModel:
    """Helper to get a Gemini model instance."""
    # Use the 'genai' alias
    return genai.GenerativeModel(model_name)

def call_gemini_api(model: genai.GenerativeModel, prompt: str, temperature: float, tools: Optional[List[Any]] = None) -> Any:
    """Generic function to call Gemini API and return the response object."""
    # Use the 'genai' alias for types
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        # response_mime_type="application/json" # Could be useful if model consistently supports it
    )

    safety_settings = [ # More permissive safety settings for this use case
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    gemini_tools_list = None # Renamed for clarity from gemini_tools to avoid conflict if genai_types.Tool was used
    if tools:
        # Assuming tools is a list of dicts like [{"google_search":{}}],
        # which the Gemini API can sometimes handle directly for pre-defined tools.
        # For explicitly declared functions, one would construct genai.types.Tool objects.
        # For google_search, this simple structure often works.
        gemini_tools_list = tools


    # The existing agent uses `genai_client.models.generate_content`
    # This implies `genai_client.models` is the actual model instance, which is confusing.
    # Let's assume `model` passed here is the correct `GenerativeModel` instance.
    # The research agent uses `genai_client.models.generate_content(model=configuration.search_model, ...)`
    # This means `genai_client.models` is NOT a model instance but something that can dispatch to one.
    # Let's try to align with that pattern if `genai_client` is indeed the `google.generativeai` module.

    # If `genai_client` is the module `google.generativeai`:
    # response = genai_client.generate_content( # This is not a method of the module
    # model=model, # This would be the model name string
    # contents=prompt,
    # generation_config=generation_config,
    # safety_settings=safety_settings,
    # tools=gemini_tools
    # )

    # If `model` is an actual `GenerativeModel` instance:
    response = model.generate_content(
        contents=prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        tools=gemini_tools_list # Corrected variable name
    )
    return response
