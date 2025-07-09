import os
import wave
import json # Import json module
import datetime # For signed URL expiration
from typing import Optional, List, Dict, Any # Added for type hints
from google.genai import Client, types
from google.cloud import storage # For GCS operations
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()

# Initialize client
gemini_api_key_value = os.getenv("GEMINI_API_KEY")

# !!! WARNING: DEBUGGING CODE - RE-ADDED - REMOVE AFTER TESTING !!!
if gemini_api_key_value:
    print(f"DEBUG (utils.py - Initialization): GEMINI_API_KEY loaded. Length: {len(gemini_api_key_value)}, First 5 chars: {gemini_api_key_value[:5]}", flush=True)
else:
    print("DEBUG (utils.py - Initialization): GEMINI_API_KEY IS NOT SET or is empty! Client initialization might fail or use default credentials if available.", flush=True)
# !!! END OF DEBUGGING CODE !!!

genai_client = Client(api_key=gemini_api_key_value)


def display_gemini_response(response):
    """Extract text from Gemini response and display as markdown with references"""
    console = Console()
    
    # Extract main content
    text = response.candidates[0].content.parts[0].text
    md = Markdown(text)
    console.print(md)
    
    # Get candidate for grounding metadata
    candidate = response.candidates[0]
    
    # Build sources text block
    sources_text = ""
    
    # Display grounding metadata if available
    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
        console.print("\n" + "="*50)
        console.print("[bold blue]References & Sources[/bold blue]")
        console.print("="*50)
        
        # Display and collect source URLs
        if candidate.grounding_metadata.grounding_chunks:
            console.print(f"\n[bold]Sources ({len(candidate.grounding_metadata.grounding_chunks)}):[/bold]")
            sources_list = []
            for i, chunk in enumerate(candidate.grounding_metadata.grounding_chunks, 1):
                if hasattr(chunk, 'web') and chunk.web:
                    title = getattr(chunk.web, 'title', 'No title') or "No title"
                    uri = getattr(chunk.web, 'uri', 'No URI') or "No URI"
                    console.print(f"{i}. {title}")
                    console.print(f"   [dim]{uri}[/dim]")
                    sources_list.append(f"{i}. {title}\n   {uri}")
            
            sources_text = "\n".join(sources_list)
        
        # Display grounding supports (which text is backed by which sources)
        if candidate.grounding_metadata.grounding_supports:
            console.print(f"\n[bold]Text segments with source backing:[/bold]")
            for support in candidate.grounding_metadata.grounding_supports[:5]:  # Show first 5
                if hasattr(support, 'segment') and support.segment:
                    snippet = support.segment.text[:100] + "..." if len(support.segment.text) > 100 else support.segment.text
                    source_nums = [str(i+1) for i in support.grounding_chunk_indices]
                    console.print(f"• \"{snippet}\" [dim](sources: {', '.join(source_nums)})[/dim]")
    
    return text, sources_text


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save PCM data to a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def create_podcast_discussion(topic, search_text, video_text, search_sources_text, video_url, filename="research_podcast.wav", configuration=None):
    """Create a 2-speaker podcast discussion explaining the research topic"""
    
    # Use default values if no configuration provided
    if configuration is None:
        from .configuration import Configuration # Relative import
        configuration = Configuration()
    
    # Step 1: Generate podcast script
    # --- DEBUG: Runtime API Key Check ---
    gemini_api_key_runtime_pd_script = os.getenv("GEMINI_API_KEY")
    if gemini_api_key_runtime_pd_script:
        print(f"DEBUG (create_podcast_discussion - Script Gen - Runtime): About to call Gemini. Key starts with: {gemini_api_key_runtime_pd_script[:5]}", flush=True)
    else:
        print("DEBUG (create_podcast_discussion - Script Gen - Runtime): GEMINI_API_KEY not found in env at runtime!", flush=True)
    # --- END DEBUG ---
    script_prompt = f"""
    Create a natural, engaging podcast conversation between Dr. Sarah (research expert) and Mike (curious interviewer) about "{topic}".
    
    Use this research content:
    
    SEARCH FINDINGS:
    {search_text}
    
    VIDEO INSIGHTS:
    {video_text}
    
    Format as a dialogue with:
    - Mike introducing the topic and asking questions
    - Dr. Sarah explaining key concepts and insights
    - Natural back-and-forth discussion (5-7 exchanges)
    - Mike asking follow-up questions
    - Dr. Sarah synthesizing the main takeaways
    - Keep it conversational and accessible (3-4 minutes when spoken)
    
    Format exactly like this:
    Mike: [opening question]
    Dr. Sarah: [expert response]
    Mike: [follow-up]
    Dr. Sarah: [explanation]
    [continue...]
    """
    
    script_response = genai_client.models.generate_content(
        model=configuration.synthesis_model,
        contents=script_prompt,
        config={"temperature": configuration.podcast_script_temperature}
    )
    
    podcast_script = script_response.candidates[0].content.parts[0].text
    
    # Step 2: Generate TTS audio
    # --- DEBUG: Runtime API Key Check ---
    gemini_api_key_runtime_pd_tts = os.getenv("GEMINI_API_KEY")
    if gemini_api_key_runtime_pd_tts:
        print(f"DEBUG (create_podcast_discussion - TTS Gen - Runtime): About to call Gemini for TTS. Key starts with: {gemini_api_key_runtime_pd_tts[:5]}", flush=True)
    else:
        print("DEBUG (create_podcast_discussion - TTS Gen - Runtime): GEMINI_API_KEY not found in env at runtime for TTS!", flush=True)
    # --- END DEBUG ---
    tts_prompt = f"TTS the following conversation between Mike and Dr. Sarah:\n{podcast_script}"
    
    response = genai_client.models.generate_content(
        model=configuration.tts_model,
        contents=tts_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker='Mike',
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=configuration.mike_voice,
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker='Dr. Sarah',
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=configuration.sarah_voice,
                                )
                            )
                        ),
                    ]
                )
            )
        )
    )
    
    # Step 3: Save audio file
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    wave_file(filename, audio_data, configuration.tts_channels, configuration.tts_rate, configuration.tts_sample_width)
    print(f"Podcast saved locally as: {filename}")

    # Step 4: Upload to GCS and generate signed URL
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        print("GCS_BUCKET_NAME environment variable not set. Skipping GCS upload.")
        # Fallback: In a real scenario, you might want to handle this more gracefully
        # or make GCS upload mandatory. For now, we'll return None for the URL.
        # Alternatively, if running locally without GCS, one might want to serve the local file.
        # However, for Cloud Run, GCS is the way for persistent, accessible files.
        return podcast_script, None # Or raise an error

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)

        # Sanitize filename for GCS path if necessary, though current filename is likely fine
        # The filename already includes topic, making it somewhat unique.
        # Adding a timestamp or UUID could make it more robustly unique if needed.
        blob_name = f"podcasts/{filename}"
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(filename)
        print(f"Uploaded {filename} to gs://{gcs_bucket_name}/{blob_name}")

        # Generate a signed URL for the blob, valid for 1 hour
        expiration_time = datetime.timedelta(hours=1)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET",
        )
        print(f"Generated signed URL: {signed_url}")

        # Clean up local file after upload (optional, good for stateless environments)
        try:
            os.remove(filename)
            print(f"Removed local file: {filename}")
        except OSError as e:
            print(f"Error removing local file {filename}: {e}")

        return podcast_script, signed_url
    except Exception as e:
        print(f"Error during GCS upload or signed URL generation: {e}")
        # Fallback or error handling
        return podcast_script, None # Or re-raise the error after logging


def create_research_report(
    topic: str,
    research_approach: str, # New parameter: "Topic Only" or "Topic Company Leads"
    search_text: Optional[str],
    video_text: Optional[str],
    search_sources_text: Optional[str],
    video_url: Optional[str],
    company_name: Optional[str] = None,
    company_specific_topic_research_text: Optional[str] = None,
    company_info_text: Optional[str] = None,
    identified_leads_data: Optional[List[Dict]] = None,
    configuration=None
):
    """Create a comprehensive research report by synthesizing available content based on research approach."""
    
    if configuration is None:
        from .configuration import Configuration # Relative import
        configuration = Configuration()

    # Step 1: Construct the synthesis prompt based on research_approach
    prompt_title = f"Research Synthesis on '{topic}'"
    if research_approach == "Topic Company Leads" and company_name:
        prompt_title += f" in Relation to {company_name}"

    input_materials_sections = []
    if search_text: # This would be general topic search for "Topic Only"
        input_materials_sections.append(f"GENERAL TOPIC SEARCH RESULTS:\n{search_text}\n")
    if company_specific_topic_research_text:
        input_materials_sections.append(f"COMPANY-SPECIFIC TOPIC RESEARCH ({company_name}):\n{company_specific_topic_research_text}\n")
    if company_info_text:
        input_materials_sections.append(f"GENERAL COMPANY INFORMATION ({company_name}):\n{company_info_text}\n")
    if video_text:
        input_materials_sections.append(f"VIDEO CONTENT INSIGHTS:\n{video_text}\n")

    if identified_leads_data:
        leads_summary_for_prompt = []
        for idx, lead in enumerate(identified_leads_data[:5]): # Show details for up to 5 leads in prompt
            lead_str = f"  Lead {idx+1}: {lead.get('lead_name', 'N/A')} ({lead.get('lead_title', 'N/A')})\n"
            lead_str += f"    Department: {lead.get('lead_department', 'N/A')}\n"
            lead_str += f"    Relevance: {lead.get('summary_of_relevance', 'N/A')}\n"
            if lead.get('named_buyers'):
                lead_str += f"    Named Buyers:\n"
                for buyer_idx, buyer in enumerate(lead.get('named_buyers', [])):
                    lead_str += f"      - {buyer.get('buyer_name', 'N/A')} ({buyer.get('buyer_title', 'N/A')}): {buyer.get('buyer_rationale', 'N/A')}\n"
            leads_summary_for_prompt.append(lead_str)
        input_materials_sections.append(f"IDENTIFIED LEADS AT {company_name}:\n" + "\n".join(leads_summary_for_prompt) + "\n")

    all_input_text = "\n---\n".join(input_materials_sections)

    synthesis_prompt = f"""
You are tasked with producing a high-quality, comprehensive research report.
The report should synthesize information from the various INPUT MATERIALS provided below.
Do not invent external information or sources.

Report Title: {prompt_title}

Please structure your report as follows:

1.  **Introduction (1-2 paragraphs):**
    *   Briefly introduce the main subject: "{topic}".
    *   If applicable (i.e., if company information is provided), introduce the company "{company_name}" and its relevance to the topic.
    *   State the purpose of this report (to synthesize and analyze the provided input materials).

2.  **Key Findings and Thematic Analysis (Multiple Paragraphs):**
    *   Identify and discuss the major themes, concepts, and findings from the INPUT MATERIALS.
    *   If company-specific research or general company information is present, integrate these insights smoothly.
    *   If video content is present, incorporate its key takeaways.
    *   If lead information is provided, briefly summarize the types of leads identified and their general relevance in a dedicated sub-section or integrated into the discussion of the company's role. Do not just list them; synthesize the findings.
    *   Ensure a logical flow and use transition sentences.

3.  **Discussion (1-2 paragraphs):**
    *   Provide an overall discussion based on all synthesized information.
    *   Highlight significant patterns, trends, or consistencies.
    *   If the materials suggest any limitations or gaps (based *only* on what's given), mention them.

4.  **Conclusion (1 paragraph):**
    *   Summarize the main findings of the report.
    *   Offer a final concluding thought.

Tone and Style: Formal, objective, analytical, and clear.
Length: Aim for a comprehensive review appropriate to the provided materials (e.g., 6-8 paragraphs or more).

INPUT MATERIALS:
{all_input_text}
---
Begin the report now, starting with the Introduction (the title is already defined above).
"""
    # --- DEBUG: Runtime API Key Check ---
    gemini_api_key_runtime_report = os.getenv("GEMINI_API_KEY")
    if gemini_api_key_runtime_report:
        print(f"DEBUG (create_research_report - Synthesis - Runtime): About to call Gemini. Key starts with: {gemini_api_key_runtime_report[:5]}", flush=True)
    else:
        print("DEBUG (create_research_report - Synthesis - Runtime): GEMINI_API_KEY not found in env at runtime!", flush=True)
    # --- END DEBUG ---
    synthesis_response = genai_client.models.generate_content(
        model=configuration.synthesis_model,
        contents=synthesis_prompt,
        config={"temperature": configuration.synthesis_temperature}
    )
    
    synthesis_text = synthesis_response.candidates[0].content.parts[0].text
    
    # Step 2: Create markdown report string
    report_sections = [f"# {prompt_title}\n\n{synthesis_text}"]

    if research_approach == "Topic Company Leads" and identified_leads_data:
        leads_md_section = ["\n\n## Identified Leads Summary\n"]
        if not identified_leads_data:
            leads_md_section.append("No specific leads were identified or provided for this report section.")
        else:
            for i, lead in enumerate(identified_leads_data):
                leads_md_section.append(f"### Lead {i+1}: {lead.get('lead_name', 'N/A')} - {lead.get('lead_title', 'N/A')}")
                leads_md_section.append(f"-   **Department:** {lead.get('lead_department', 'N/A')}")
                leads_md_section.append(f"-   **LinkedIn:** {lead.get('linkedin_url', 'N/A') if lead.get('linkedin_url') else 'Not available'}")
                leads_md_section.append(f"-   **Relevance:** {lead.get('summary_of_relevance', 'N/A')}")
                if lead.get('named_buyers'):
                    leads_md_section.append("-   **Potential Named Buyers:**")
                    for buyer in lead.get('named_buyers', []):
                        leads_md_section.append(f"    -   {buyer.get('buyer_name', 'N/A')} ({buyer.get('buyer_title', 'N/A')}): {buyer.get('buyer_rationale', 'N/A')}")
                leads_md_section.append("\n")
        report_sections.append("\n".join(leads_md_section))

    if video_url:
        report_sections.append(f"\n\n## Video Source\n- **URL**: {video_url if video_url else 'Not provided'}")

    if search_sources_text: # General search sources
        report_sections.append(f"\n\n## Additional Research Sources\n{search_sources_text if search_sources_text else 'None available'}")

    report_sections.append("\n\n---\n*Report generated using multi-modal AI research.*")
    report_content = "\n".join(report_sections)

    # Step 3: Upload report to GCS
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        print("GCS_BUCKET_NAME environment variable not set. Skipping GCS upload for report.")
        # In a real scenario, decide how to handle this. For now, returning content and None URL.
        return report_content, synthesis_text # Or return None, synthesis_text if URL is mandatory

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)

        # Create a unique filename for the report
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        report_filename = f"research_report_{safe_topic.replace(' ', '_')}.md"
        blob_name = f"reports/{report_filename}"
        blob = bucket.blob(blob_name)

        # Upload the report content
        blob.upload_from_string(report_content, content_type='text/markdown')
        print(f"Uploaded report to gs://{gcs_bucket_name}/{blob_name}")

        # Generate a signed URL for the blob, valid for 1 hour
        expiration_time = datetime.timedelta(hours=1)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET",
        )
        print(f"Generated signed URL for report: {signed_url}")

        return signed_url, synthesis_text
    except Exception as e:
        print(f"Error during GCS upload or signed URL generation for report: {e}")
        # Fallback or error handling
        return report_content, synthesis_text # Or None, synthesis_text


def generate_company_topic_research_prompt(topic: str, company_name: str) -> str:
    """Generates a prompt to research a topic in the context of a specific company."""
    return f"""
Conduct detailed research on the topic "{topic}" specifically as it relates to the company "{company_name}".
Additionally, gather general information about "{company_name}", including:
- Its primary business, industry, and market position.
- Key products, services, or initiatives relevant to "{topic}".
- Any publicly available information about its organizational structure or key departments related to "{topic}".

Provide a comprehensive overview based on publicly available information. Focus on factual data and established knowledge.
Structure your response into two main sections:
1.  **Topic Research in Company Context:** Detailed findings about "{topic}" pertaining to "{company_name}".
2.  **General Company Information:** Overview of "{company_name}" relevant to the research.

Please ensure the information is well-organized and clearly presented.
"""

def generate_lead_identification_prompt(company_name: str, title_areas: List[str], company_topic_context: str) -> str:
    """
    Generates a prompt for identifying leads, their departments, and named buyers.
    """
    titles_str = ", ".join(f"'{title}'" for title in title_areas)
    prompt = f"""
You are a specialized Lead Identification and Market Research AI.
Your task is to identify up to 5 key individuals (leads) at the company "{company_name}" who match the specified title areas: {titles_str}.
The research should be informed by the following context about the company's activities related to a specific topic:
Context: "{company_topic_context[:1500]}" (Context is provided for background, focus on identifying people based on titles and company)

For each of the (up to) 5 leads identified, provide the following information in a structured JSON format.
The output should be a single JSON list, where each item is an object representing a lead:
{{
  "lead_name": "string (Full name of the lead)",
  "lead_title": "string (Exact job title of the lead at {company_name})",
  "lead_department": "string (Department the lead likely belongs to, e.g., 'Marketing', 'Engineering', 'Product Management')",
  "linkedin_url": "string (Full LinkedIn profile URL if available, otherwise null)",
  "summary_of_relevance": "string (Brief 1-2 sentence summary explaining why this person is a relevant lead based on their title and potential connection to the topic context)",
  "named_buyers": [ // Up to 3 potential named buyers associated with this lead or their area of influence
    {{
      "buyer_name": "string (Full name of the named buyer)",
      "buyer_title": "string (Job title of the named buyer)",
      "buyer_rationale": "string (Brief rationale why this person is considered a potential buyer/influencer for solutions related to the topic, in context of the lead or company)"
    }},
    // ... more buyers if applicable (up to 3)
  ]
}}

Example of a single lead object in the list:
```json
{{
  "lead_name": "Dr. Eleanor Vance",
  "lead_title": "VP of AI Research",
  "lead_department": "Research and Development",
  "linkedin_url": "https://linkedin.com/in/eleanorvance",
  "summary_of_relevance": "As VP of AI Research, Dr. Vance is directly involved in the company's strategic direction for AI, making her a key contact for understanding {company_name}'s needs in this area.",
  "named_buyers": [
    {{
      "buyer_name": "Mr. Samuel Green",
      "buyer_title": "Chief Technology Officer (CTO)",
      "buyer_rationale": "The CTO typically has budget authority and strategic oversight for technology adoption, including AI initiatives led by Dr. Vance's department."
    }},
    {{
      "buyer_name": "Ms. Olivia Chen",
      "buyer_title": "Director of Innovation Strategy",
      "buyer_rationale": "Works closely with R&D on implementing new technologies and would likely be involved in evaluating solutions related to the topic."
    }}
  ]
}}
```

IMPORTANT:
- Return *only* the JSON list. Do not include any introductory text, explanations, or markdown formatting like ```json ... ``` outside the JSON list itself.
- If no leads are found, return an empty JSON list `[]`.
- Ensure all string fields are properly escaped within the JSON.
- Use Google Search to find this information. Prioritize publicly available, professional information.
"""
    return prompt

def parse_leads_from_gemini_response(gemini_response: Any) -> List[Dict]:
    """
    Parses the Gemini response, expecting a JSON string containing a list of leads.
    """
    # Assuming the response object structure is similar to what display_gemini_response handles
    # or that the relevant text part containing JSON is directly accessible.
    # This part might need to be made more robust based on actual Gemini API response structure
    # for generate_content calls that are expected to return JSON.

    raw_text = ""
    if hasattr(gemini_response, 'text') and gemini_response.text: # Direct text attribute
        raw_text = gemini_response.text
    elif hasattr(gemini_response, 'candidates') and gemini_response.candidates:
        # Standard path for generate_content responses
        if gemini_response.candidates[0].content and gemini_response.candidates[0].content.parts:
            raw_text = gemini_response.candidates[0].content.parts[0].text
    else:
        print("WARN (parse_leads_from_gemini_response): Gemini response structure not recognized or empty.")
        return []

    raw_text = raw_text.strip()

    # Attempt to extract JSON from markdown code block if present
    if raw_text.startswith("```json"):
        json_block_start = raw_text.find("```json") + 7 # Length of "```json\n"
        json_block_end = raw_text.rfind("```")
        if json_block_start != -1 and json_block_end != -1 and json_block_end > json_block_start :
            json_str = raw_text[json_block_start:json_block_end].strip()
        else: # Fallback if markdown block is malformed but starts with ```json
            json_str = raw_text.replace("```json", "").replace("```", "").strip()
    else:
        json_str = raw_text

    try:
        leads_data = json.loads(json_str)
        if isinstance(leads_data, list):
            return leads_data
        else:
            print(f"ERROR (parse_leads_from_gemini_response): Parsed JSON is not a list. Got: {type(leads_data)}")
            return []
    except json.JSONDecodeError as e:
        print(f"ERROR (parse_leads_from_gemini_response): JSONDecodeError - {e}. Raw text was: '{json_str[:500]}...'")
        return []
    except Exception as e:
        print(f"ERROR (parse_leads_from_gemini_response): An unexpected error occurred during JSON parsing - {e}")
        return []
