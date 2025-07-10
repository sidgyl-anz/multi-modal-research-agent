import os
import wave
import json # Import json module
import requests # For CSE API calls
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
    linkedin_cse_contacts: Optional[List[Dict]] = None, # New parameter
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

    if identified_leads_data: # This now contains B2B Opportunity objects
        opportunity_summary_for_prompt = ["\n\nSUMMARY OF IDENTIFIED B2B OPPORTUNITIES:"]
        for idx, opp in enumerate(identified_leads_data[:3]): # Summarize first 3 opportunities for prompt context
            opportunity_summary_for_prompt.append(f"  Opportunity {idx+1}: {opp.get('opportunity_name', 'N/A')}")
            opportunity_summary_for_prompt.append(f"    Description: {opp.get('opportunity_description', 'N/A')[:200]}...") # Snippet
            if opp.get('contact_points'):
                opportunity_summary_for_prompt.append(f"    Key Contact: {opp['contact_points'][0].get('contact_name', 'N/A')} ({opp['contact_points'][0].get('contact_title', 'N/A')})")
        input_materials_sections.append("\n".join(opportunity_summary_for_prompt))

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
    report_sections = [f"# {prompt_title}\n\n{synthesis_text}"] # This is the main LLM-generated synthesis

    # Section for Identified B2B Opportunities (from Gemini)
    if research_approach == "Topic Company Leads" and identified_leads_data: # identified_leads_data now opportunity objects
        opportunities_md_section = ["\n\n## Identified B2B Opportunities\n"]
        if not identified_leads_data:
            opportunities_md_section.append("No specific B2B opportunities were identified by the AI.")
        else:
            for i, opp in enumerate(identified_leads_data):
                opportunities_md_section.append(f"### Opportunity {i+1}: {opp.get('opportunity_name', 'N/A')}")
                opportunities_md_section.append(f"**Description:** {opp.get('opportunity_description', 'N/A')}")
                opportunities_md_section.append(f"**Relevant Departments:** {', '.join(opp.get('relevant_departments', [])) if opp.get('relevant_departments') else 'N/A'}")

                opportunities_md_section.append("\n**Contact Points for this Opportunity:**")
                if opp.get('contact_points'):
                    for cp_idx, cp in enumerate(opp.get('contact_points', [])):
                        opportunities_md_section.append(f"-   **Contact {cp_idx+1}:** {cp.get('contact_name', 'N/A')} ({cp.get('contact_title', 'N/A')})")
                        opportunities_md_section.append(f"    -   Department: {cp.get('contact_department', 'N/A')}")
                        opportunities_md_section.append(f"    -   LinkedIn: {cp.get('contact_linkedin_url', 'N/A') if cp.get('contact_linkedin_url') else 'Not available'}")
                        opportunities_md_section.append(f"    -   Relevance: {cp.get('contact_relevance_to_opportunity', 'N/A')}")
                else:
                    opportunities_md_section.append("  No specific contact points identified for this opportunity.")

                opportunities_md_section.append("\n**Potential Decision-Makers for this Opportunity Type:**")
                if opp.get('potential_decision_makers_for_opportunity'):
                    for dm_idx, dm in enumerate(opp.get('potential_decision_makers_for_opportunity', [])):
                        opportunities_md_section.append(f"-   **Decision Maker {dm_idx+1}:** {dm.get('dm_name', 'N/A')} ({dm.get('dm_title', 'N/A')})")
                        opportunities_md_section.append(f"    -   Rationale: {dm.get('dm_rationale', 'N/A')}")
                else:
                    opportunities_md_section.append("  No specific decision-makers identified for this opportunity type.")
                opportunities_md_section.append("\n---\n") # Separator for each opportunity
        report_sections.append("\n".join(opportunities_md_section))

    if video_url:
        report_sections.append(f"\n\n## Video Source\n- **URL**: {video_url if video_url else 'Not provided'}")

    if search_sources_text: # General search sources
        report_sections.append(f"\n\n## Additional Research Sources\n{search_sources_text if search_sources_text else 'None available'}")

    if linkedin_cse_contacts:
        cse_contacts_md_section = ["\n\n## LinkedIn Contacts (via Custom Search)\n"]
        if not linkedin_cse_contacts:
            cse_contacts_md_section.append("No additional LinkedIn contacts found via Custom Search Engine.")
        else:
            for i, contact in enumerate(linkedin_cse_contacts):
                cse_contacts_md_section.append(f"### CSE Contact {i+1}: {contact.get('title', 'N/A')}")
                cse_contacts_md_section.append(f"-   **Link:** {contact.get('link', 'N/A')}")
                cse_contacts_md_section.append(f"-   **Snippet:** {contact.get('snippet', 'N/A')}")
                cse_contacts_md_section.append("\n")
        report_sections.append("\n".join(cse_contacts_md_section))

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
    Generates a prompt for identifying B2B sales opportunities and associated contacts/decision-makers.
    """
    # title_areas will be used to guide the identification of contact_points for each opportunity.
    titles_hint = ""
    if title_areas:
        titles_list_str = ", ".join(f"'{t}'" for t in title_areas)
        titles_hint = f"When identifying contact points for each opportunity, pay special attention to individuals with titles such as {titles_list_str} or similar roles relevant to the opportunity."

    prompt = f"""
You are a B2B Sales Opportunity Identification AI. Your goal is to identify potential sales opportunities or projects.
Based on the company "{company_name}", the topic "{company_topic_context.splitlines()[0] if company_topic_context else 'the provided context'}",
and the following detailed context about the company's activities:
Context: "{company_topic_context[:2000]}"

Identify up to 3-5 potential B2B sales opportunities or projects within "{company_name}" that are relevant to the topic and context.

For each identified opportunity, provide the following information in a structured JSON format.
The output MUST be a single JSON list, where each item is an object representing an opportunity:
{{
  "opportunity_name": "string (Descriptive name of the potential sales opportunity/project, e.g., 'AI-Powered Upgrade for CRM Platform', 'New Data Analytics Initiative for Marketing Dept', 'Development of [TOPIC]-based Solution for [COMPANY_NAME]'s specific need X')",
  "opportunity_description": "string (Detailed 2-3 sentence description: What is the opportunity? Why is it relevant for [COMPANY_NAME] in relation to the topic? What kind of solution or service might they need? What problem does it solve for them?)",
  "relevant_departments": ["string", "string", ...], // List of departments within [COMPANY_NAME] most likely involved or benefiting from this opportunity.
  "contact_points": [ // Up to 3-5 key individuals at {company_name} who would be relevant points of contact for initial discussions about THIS SPECIFIC OPPORTUNITY. {titles_hint}
    {{
      "contact_name": "string (Full name of the contact person)",
      "contact_title": "string (Exact job title of the contact at [COMPANY_NAME])",
      "contact_department": "string (Their department, if known or inferable)",
      "contact_linkedin_url": "string (Full LinkedIn profile URL if available, otherwise null)",
      "contact_relevance_to_opportunity": "string (Briefly explain why this person is a relevant contact for THIS specific opportunity, considering their role and the opportunity's nature)"
    }}
  ],
  "potential_decision_makers_for_opportunity": [ // Up to 3 key individuals at {company_name} likely to be decision-makers or budget-holders for THIS TYPE of opportunity.
    {{
      "dm_name": "string (Full name of the decision-maker)",
      "dm_title": "string (Job title of the decision-maker at {company_name})",
      "dm_rationale": "string (Briefly explain why this person is likely a decision-maker or budget-holder for this type of opportunity, e.g., based on their seniority, role scope, or typical responsibilities)"
    }}
  ]
}}

Example of a single opportunity object in the list:
```json
{{
  "opportunity_name": "AI-Driven Predictive Maintenance for Manufacturing Lines",
  "opportunity_description": "[COMPANY_NAME] could significantly reduce downtime and maintenance costs in their manufacturing division by implementing an AI-driven predictive maintenance solution. This aligns with their stated goals of increasing operational efficiency and leveraging new technologies for [TOPIC].",
  "relevant_departments": ["Manufacturing", "Operations", "IT", "Data Science"],
  "contact_points": [
    {{
      "contact_name": "Sarah Miller",
      "contact_title": "Director of Plant Operations",
      "contact_department": "Operations",
      "contact_linkedin_url": "https://linkedin.com/in/sarahmillerops",
      "contact_relevance_to_opportunity": "Directly responsible for the efficiency and uptime of manufacturing lines; would be a key stakeholder and initial contact for discussing operational improvements through predictive maintenance."
    }},
    {{
      "contact_name": "James Lee",
      "contact_title": "Senior Manager, Data Analytics",
      "contact_department": "IT / Data Science",
      "contact_linkedin_url": "https://linkedin.com/in/jamesleedata",
      "contact_relevance_to_opportunity": "Likely involved in evaluating and implementing the data infrastructure and AI models required for such a solution."
    }}
  ],
  "potential_decision_makers_for_opportunity": [
    {{
      "dm_name": "Robert Green",
      "dm_title": "Chief Operating Officer (COO) at [COMPANY_NAME]",
      "dm_rationale": "The COO typically oversees large operational expenditures and strategic initiatives aimed at improving efficiency and would likely approve such a project."
    }},
    {{
      "dm_name": "Maria Rodriguez",
      "dm_title": "VP of Manufacturing at [COMPANY_NAME]",
      "dm_rationale": "Directly responsible for the manufacturing budget and would be a key approver for solutions impacting production lines."
    }}
  ]
}}
```

IMPORTANT INSTRUCTIONS:
- The entire response MUST be a single valid JSON list. Do not include any text or explanation before or after the JSON list.
- If no specific B2B opportunities are identified, return an empty JSON list: `[]`.
- Ensure all string values within the JSON are properly escaped.
- Use Google Search to find information to support your identifications. Focus on publicly available professional information.
- The "contact_points" should be people relevant for *initial discussions* about the specific opportunity you've identified.
- The "potential_decision_makers_for_opportunity" should be higher-level individuals likely responsible for approving or funding such an initiative.
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

    json_str = raw_text # Default to trying to parse the whole raw_text

    # Try to find and extract a JSON markdown block if present
    block_start_marker = "```json"
    block_end_marker = "```"

    start_index = raw_text.find(block_start_marker)
    if start_index != -1:
        # Found the start of a JSON block
        content_start_index = start_index + len(block_start_marker)
        # Handle optional newline after ```json
        if content_start_index < len(raw_text) and raw_text[content_start_index] == '\n':
            content_start_index += 1

        end_index = raw_text.find(block_end_marker, content_start_index)
        if end_index != -1:
            # Found a complete block
            json_str_candidate = raw_text[content_start_index:end_index].strip()
            # Basic validation: if it looks like JSON, use it. Otherwise, stick with raw_text.
            if (json_str_candidate.startswith("[") and json_str_candidate.endswith("]")) or \
               (json_str_candidate.startswith("{") and json_str_candidate.endswith("}")):
                json_str = json_str_candidate
            # else: malformed block or content doesn't look like JSON, will try to parse original json_str (raw_text)
        # else: no proper end marker found after start marker, try to parse original json_str (raw_text)

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

def build_linkedin_cse_query(company_name: str, title_areas: List[str]) -> str:
    """Builds a Google Custom Search query for LinkedIn profiles."""
    # Sanitize company name and titles for query (simple quotes for now)
    safe_company_name = f'"{company_name}"'
    safe_title_queries = [f'"{title}"' for title in title_areas]
    titles_query_part = " OR ".join(safe_title_queries)

    # Construct the query
    # Example: site:linkedin.com/in ("Some Company") ("VP of Engineering" OR "Chief Architect")
    query = f'site:linkedin.com/in ({safe_company_name}) ({titles_query_part})'
    return query

def fetch_linkedin_contacts_via_cse(query: str, api_key: str, cse_id: str, num_results: int = 10) -> List[Dict]:
    """
    Fetches LinkedIn contacts using Google Custom Search API.
    Returns a list of dicts, each with 'title', 'link', 'snippet'.
    """
    print(f"INFO (fetch_linkedin_contacts_via_cse): Performing CSE search with query: {query}", flush=True)
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id,
        'num': num_results, # Number of search results to return
        # 'gl': 'us',  # Optional: Geolocation bias (country)
        # 'lr': 'lang_en',  # Optional: Language restriction
    }

    contacts_found = []
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        search_results = response.json()

        items = search_results.get('items', [])
        for item in items:
            contacts_found.append({
                "title": item.get("title", "N/A"),
                "link": item.get("link", "N/A"),
                "snippet": item.get("snippet", "N/A")
            })
        print(f"INFO (fetch_linkedin_contacts_via_cse): Found {len(contacts_found)} contacts via CSE.", flush=True)
    except requests.exceptions.HTTPError as http_err:
        print(f"ERROR (fetch_linkedin_contacts_via_cse): HTTP error occurred: {http_err} - {response.text}", flush=True)
    except requests.exceptions.RequestException as req_err:
        print(f"ERROR (fetch_linkedin_contacts_via_cse): Request error occurred: {req_err}", flush=True)
    except json.JSONDecodeError as json_err:
        print(f"ERROR (fetch_linkedin_contacts_via_cse): JSON decode error: {json_err} - Response was: {response.text}", flush=True)
    except Exception as e:
        print(f"ERROR (fetch_linkedin_contacts_via_cse): An unexpected error occurred: {e}", flush=True)

    return contacts_found
