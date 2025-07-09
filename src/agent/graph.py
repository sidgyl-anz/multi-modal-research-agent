"""LangGraph implementation of the research and podcast generation workflow"""

import os # Added for runtime debug
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from google.genai import types
import json # For debug printing of lead data

from .state import ResearchState, ResearchStateInput, ResearchStateOutput, ResearchApproach # Ensure ResearchApproach is imported if used explicitly
from .utils import (
    display_gemini_response,
    create_podcast_discussion,
    create_research_report,
    genai_client,
    generate_company_topic_research_prompt,
    generate_lead_identification_prompt,
    parse_leads_from_gemini_response,
    build_linkedin_cse_query,               # New
    fetch_linkedin_contacts_via_cse         # New
)
from .configuration import Configuration
from langsmith import traceable

# Define the Google Search tool for Gemini, used by multiple nodes
GOOGLE_SEARCH_TOOL = [{"google_search": {}}]

@traceable(run_type="llm", name="Web Research", project_name="multi-modal-researcher")
def search_research_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that performs web search research on the topic"""
    # --- DEBUG: Print received state ---
    print(f"DEBUG (search_research_node): Received state: {state}", flush=True)
    # --- END DEBUG ---
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]

    # --- DEBUG: Runtime API Key Check ---
    gemini_api_key_runtime = os.getenv("GEMINI_API_KEY")
    if gemini_api_key_runtime:
        print(f"DEBUG (search_research_node - Runtime): About to call Gemini. Key starts with: {gemini_api_key_runtime[:5]}", flush=True)
    else:
        print("DEBUG (search_research_node - Runtime): GEMINI_API_KEY not found in env at runtime!", flush=True)
    # --- END DEBUG ---
    
    search_response = genai_client.models.generate_content(
        model=configuration.search_model,
        contents=f"Research this topic and give me an overview: {topic}",
        config={
            "tools": [{"google_search": {}}],
            "temperature": configuration.search_temperature,
        },
    )
    
    search_text, search_sources_text = display_gemini_response(search_response)
    
    return {
        "search_text": search_text,
        "search_sources_text": search_sources_text
    }

@traceable(run_type="llm", name="Company Topic Research", project_name="multi-modal-researcher")
def company_topic_research_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that performs web search on a topic related to a specific company and gathers company info."""
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]
    company_name = state.get("company_name")

    if not company_name:
        # This node should only be called if company_name is available and approach is Topic Company Leads
        print("ERROR (company_topic_research_node): Company name not provided.")
        return {
            "company_specific_topic_research_text": "Error: Company name required.",
            "company_info_text": "Error: Company name required."
        }

    prompt = generate_company_topic_research_prompt(topic, company_name)

    # Using search_model and search_temperature for this, can be configured separately if needed
    response = genai_client.models.generate_content(
        model=configuration.search_model,
        contents=prompt,
        config={
            "tools": GOOGLE_SEARCH_TOOL,
            "temperature": configuration.search_temperature,
        },
    )

    # display_gemini_response extracts main text and source text.
    # For this node, the prompt asks for two sections. We might need more sophisticated parsing
    # or trust the LLM to structure it well enough that the full text is usable.
    # For now, let's assume the full text contains both parts and can be used as context.
    # display_gemini_response also prints to console, which might be too verbose here.
    # Let's directly get the text.

    full_research_text = ""
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        full_research_text = response.candidates[0].content.parts[0].text

    # Heuristic to split: Assume "General Company Information:" is a good marker if LLM follows prompt.
    # This is a simplification. A more robust way might be two separate LLM calls or more structured output.
    company_info_marker = "General Company Information:"
    company_specific_text = full_research_text
    company_general_text = ""

    marker_idx = full_research_text.lower().find(company_info_marker.lower())
    if marker_idx != -1:
        company_specific_text = full_research_text[:marker_idx].strip()
        company_general_text = full_research_text[marker_idx + len(company_info_marker):].strip()
    else: # Fallback if marker not found
        print("WARN (company_topic_research_node): Could not clearly separate company-specific topic research from general company info.")
        # Assign a portion, or all to specific, and leave general empty or with a note
        # For now, assign all to company_specific_topic_research_text for simplicity if not split
        company_specific_text = full_research_text


    return {
        "company_specific_topic_research_text": company_specific_text,
        "company_info_text": company_general_text,
        # We might also want to capture sources here if display_gemini_response isn't used.
        # For now, this is simplified.
    }

@traceable(run_type="llm", name="Identify Leads", project_name="multi-modal-researcher")
def identify_leads_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that identifies leads at a company based on titles and topic context."""
    configuration = Configuration.from_runnable_config(config)
    company_name = state.get("company_name")
    title_areas = state.get("title_areas")
    # Use company_specific_topic_research_text as context for better lead relevance
    company_topic_context = state.get("company_specific_topic_research_text", "")

    if not company_name or not title_areas:
        print("ERROR (identify_leads_node): Company name and title areas are required.")
        return {"identified_leads_data": [], "identified_leads": []}

    prompt = generate_lead_identification_prompt(company_name, title_areas, company_topic_context)

    response = genai_client.models.generate_content(
        model=configuration.lead_identification_model, # Use dedicated model
        contents=prompt,
        config={ # Pass tools and temperature within the config dictionary
            "tools": [types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())],
            "temperature": configuration.lead_identification_temperature,
            # "response_mime_type": "application/json", # If needed for structured output
        }
    )

    identified_leads = parse_leads_from_gemini_response(response)
    print(f"DEBUG (identify_leads_node): Identified {len(identified_leads)} leads. Data: {json.dumps(identified_leads[:1], indent=2)} (first lead example)", flush=True)

    return {
        "identified_leads_data": identified_leads, # Intermediate state for report generation
        "identified_leads": identified_leads # Final output state
    }

@traceable(run_type="llm", name="YouTube Video Analysis", project_name="multi-modal-researcher")
def analyze_video_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that analyzes video content if video URL is provided"""
    configuration = Configuration.from_runnable_config(config)
    video_url = state.get("video_url")
    topic = state["topic"]
    
    if not video_url:
        return {"video_text": "No video provided for analysis."}
    
    video_response = genai_client.models.generate_content(
        model=configuration.video_model,
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=video_url)
                ),
                types.Part(text=f'Based on the video content, give me an overview of this topic: {topic}')
            ]
        )
    )
    
    video_text, _ = display_gemini_response(video_response)
    
    return {"video_text": video_text}

@traceable(run_type="llm", name="Create Report", project_name="multi-modal-researcher")
def create_report_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a comprehensive research report"""
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]
    research_approach = state["research_approach"] # New

    # Data for "Topic Only" or common data
    search_text = state.get("search_text") # This will be populated by search_research_node
    search_sources_text = state.get("search_sources_text") # From search_research_node
    video_text = state.get("video_text") # Common, from analyze_video_node if run
    video_url = state.get("video_url")   # Common input

    # Data for "Topic Company Leads"
    company_name = state.get("company_name") # Input
    company_specific_topic_research_text = state.get("company_specific_topic_research_text") # From company_topic_research_node
    company_info_text = state.get("company_info_text") # From company_topic_research_node
    identified_leads_data = state.get("identified_leads_data") # From identify_leads_node

    # Call the modified utility function from utils.py
    # It now handles the different research approaches internally.
    report_url_or_text, synthesis_text = create_research_report(
        topic=topic,
        research_approach=research_approach,
        search_text=search_text, # Will be None if Topic Company Leads path didn't run search_research_node
        video_text=video_text,
        search_sources_text=search_sources_text, # Will be None if Topic Company Leads path didn't run search_research_node
        video_url=video_url,
        company_name=company_name,
        company_specific_topic_research_text=company_specific_topic_research_text,
        company_info_text=company_info_text,
        identified_leads_data=identified_leads_data,
        configuration=configuration
    )
    
    # The 'report' field in state should be the GCS URL or the report text itself if GCS fails.
    # 'identified_leads' is already set directly by identify_leads_node if that path was taken.
    # The 'create_research_report' utility also formats the lead data into the report text.
    return {
        "report": report_url_or_text,
        "synthesis_text": synthesis_text
        # 'identified_leads' output field is populated by identify_leads_node directly
    # 'linkedin_cse_contacts' will be populated by the new CSE search node.
    }

@traceable(run_type="tool", name="Search LinkedIn via CSE", project_name="multi-modal-researcher")
def search_linkedin_via_cse_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that searches LinkedIn for contacts using Google Custom Search Engine."""
    # Configuration is not directly used here for CSE keys, as they come from env vars.
    # However, if we added CSE config (like num_results) to Configuration, we'd load it.
    # configuration = Configuration.from_runnable_config(config)

    company_name = state.get("company_name")
    title_areas = state.get("title_areas")

    if not company_name or not title_areas:
        print("WARN (search_linkedin_via_cse_node): Company name or title areas missing. Skipping CSE LinkedIn search.")
        return {"linkedin_cse_contacts": []}

    # Retrieve API keys from environment variables
    # These must be set in the environment where the agent is running.
    cse_api_key = os.getenv("GOOGLE_API_KEY_FOR_CSE")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not cse_api_key or not cse_id:
        print("WARN (search_linkedin_via_cse_node): GOOGLE_API_KEY_FOR_CSE or GOOGLE_CSE_ID not set in environment. Skipping CSE LinkedIn search.")
        return {"linkedin_cse_contacts": []}

    query = build_linkedin_cse_query(company_name, title_areas)

    # Potentially get num_results from config if we add it there, e.g., configuration.cse_num_results
    num_results_to_fetch = 10 # Default or could be from config

    linkedin_contacts = fetch_linkedin_contacts_via_cse(query, cse_api_key, cse_id, num_results=num_results_to_fetch)

    print(f"DEBUG (search_linkedin_via_cse_node): Found {len(linkedin_contacts)} LinkedIn contacts via CSE.", flush=True)

    return {"linkedin_cse_contacts": linkedin_contacts}


@traceable(run_type="llm", name="Create Podcast", project_name="multi-modal-researcher")
def create_podcast_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a podcast discussion"""
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]
    research_approach = state["research_approach"]

    # Common fields
    video_text = state.get("video_text", "")
    video_url = state.get("video_url", "")
    # search_sources_text might be from general search or company search, or empty.
    # The podcast utility doesn't directly use it for script, mainly for GCS naming or metadata.
    search_sources_text = state.get("search_sources_text", "")
                                     # Or state.get("company_research_sources_text") if we add that.
                                     # For now, using existing search_sources_text.

    # Determine the primary research text for the podcast based on the approach
    primary_research_text_for_podcast: str
    if research_approach == "Topic Company Leads":
        primary_research_text_for_podcast = state.get("company_specific_topic_research_text", "") \
                                           or state.get("company_info_text", "") # Use company research
    else: # "Topic Only"
        primary_research_text_for_podcast = state.get("search_text", "") # Use general topic research

    # Fallback to synthesis_text if primary research texts are empty but synthesis exists
    if not primary_research_text_for_podcast and state.get("synthesis_text"):
        # This case implies the podcast is more of a summary of the report itself.
        # The current create_podcast_discussion prompt is geared towards discussing raw findings.
        # Using synthesis_text here might make the podcast a "reading of the summary".
        # This might be an area for future improvement in the podcast script generation prompt.
        primary_research_text_for_podcast = state.get("synthesis_text", "No detailed research text available for podcast.")


    # Create unique filename based on topic and optionally company
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    company_name_safe_segment = ""
    if research_approach == "Topic Company Leads" and state.get("company_name"):
        company_name = state.get("company_name", "")
        company_name_safe = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        if company_name_safe:
             company_name_safe_segment = f"_{company_name_safe}"
    
    filename = f"research_podcast_{safe_topic}{company_name_safe_segment}.wav"

    # Call the utility function
    # Note: create_podcast_discussion's internal prompt uses its 'search_text' param as the main research content.
    podcast_script, podcast_url = create_podcast_discussion(
        topic=topic, # Topic remains the core subject
        search_text=primary_research_text_for_podcast, # This is now context-dependent
        video_text=video_text,
        search_sources_text=search_sources_text, # This is less critical for script, more for context/metadata
        video_url=video_url,
        filename=filename,
        configuration=configuration
    )
    
    return {
        "podcast_script": podcast_script,
        "podcast_url": podcast_url
    }

def should_analyze_video(state: ResearchState) -> str:
    """Conditional edge to determine if video analysis should be performed"""
    if state.get("video_url"):
        return "analyze_video"
    else:
        return "create_report"

def should_create_podcast(state: ResearchState) -> str:
    """Conditional edge to determine if podcast creation should be performed"""
    if state.get("create_podcast", False): # Default to False if not set, though it should be
        return "create_podcast"
    else:
        return END

# New conditional edge function
def should_perform_company_research(state: ResearchState) -> str:
    """Determines the initial research path based on research_approach."""
    if state.get("research_approach") == "Topic Company Leads":
        # Validate that company_name and title_areas are provided if this approach is chosen
        if not state.get("company_name") or not state.get("title_areas"):
            # This ideally should be validated at input, but as a safeguard:
            print("WARN (should_perform_company_research): 'Topic Company Leads' chosen, but company_name or title_areas missing. Defaulting to 'Topic Only' path.")
            return "topic_only_path" # Or raise an error / go to an error handling node
        return "company_leads_path"
    return "topic_only_path"


def create_research_graph() -> StateGraph:
    """Create and return the research workflow graph"""
    
    graph = StateGraph(
        ResearchState, 
        input=ResearchStateInput, 
        output=ResearchStateOutput,
        config_schema=Configuration
    )
    
    # Add all nodes, including new ones
    graph.add_node("search_research", search_research_node) # For "Topic Only"
    graph.add_node("company_topic_research", company_topic_research_node)
    graph.add_node("identify_leads", identify_leads_node)
    graph.add_node("search_linkedin_via_cse", search_linkedin_via_cse_node) # New CSE node
    graph.add_node("analyze_video", analyze_video_node)
    graph.add_node("create_report", create_report_node)
    graph.add_node("create_podcast", create_podcast_node)

    # 1. Start: Conditional routing based on research_approach
    graph.add_conditional_edges(
        START,
        should_perform_company_research,
        {
            "topic_only_path": "search_research",
            "company_leads_path": "company_topic_research"
        }
    )
    
    # 2. "Topic Only" Path
    # search_research -> (optional video) -> create_report
    graph.add_conditional_edges(
        "search_research",
        should_analyze_video, # Existing conditional
        {
            "analyze_video": "analyze_video",
            "create_report": "create_report" # If no video, go directly to report
        }
    )

    # 3. "Topic Company Leads" Path
    # company_topic_research -> identify_leads -> search_linkedin_via_cse -> (optional video) -> create_report
    graph.add_edge("company_topic_research", "identify_leads")
    graph.add_edge("identify_leads", "search_linkedin_via_cse") # New edge
    graph.add_conditional_edges(
        "search_linkedin_via_cse", # From new CSE node
        should_analyze_video,
        {
            "analyze_video": "analyze_video",
            "create_report": "create_report"
        }
    )

    # 4. Video Analysis Path (common for both main paths if video_url is provided)
    # analyze_video -> create_report
    graph.add_edge("analyze_video", "create_report")

    # 5. Report to Podcast/End
    # create_report -> (optional podcast) -> END
    graph.add_conditional_edges(
        "create_report",
        should_create_podcast, # Existing conditional
        {
            "create_podcast": "create_podcast",
            END: END
        }
    )
    graph.add_edge("create_podcast", END)
    
    return graph


def create_compiled_graph():
    """Create and compile the research graph"""
    graph = create_research_graph()
    return graph.compile()

if __name__ == "__main__":
    # This block allows direct execution for manual testing.
    # Ensure GEMINI_API_KEY is set in your environment.
    # For CSE LinkedIn search, also set GOOGLE_API_KEY_FOR_CSE and GOOGLE_CSE_ID.
    # You might also need GOOGLE_APPLICATION_CREDENTIALS for GCS if testing podcast/report GCS upload.

    print("Attempting to run research graph for manual testing...", flush=True)
    print("Required Env Vars for full test: GEMINI_API_KEY, GOOGLE_API_KEY_FOR_CSE, GOOGLE_CSE_ID, (optional for GCS: GCS_BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS)", flush=True)

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Gemini related calls will fail.", flush=True)
    else:
        print(f"GEMINI_API_KEY found, starting with: {os.getenv('GEMINI_API_KEY')[:5]}", flush=True)

    if not os.getenv("GOOGLE_API_KEY_FOR_CSE") or not os.getenv("GOOGLE_CSE_ID"):
        print("GOOGLE_API_KEY_FOR_CSE or GOOGLE_CSE_ID not set. CSE LinkedIn search will be skipped or fail.", flush=True)
    else:
        print(f"GOOGLE_API_KEY_FOR_CSE found, starting with: {os.getenv('GOOGLE_API_KEY_FOR_CSE')[:5]}", flush=True)
        print(f"GOOGLE_CSE_ID found, starting with: {os.getenv('GOOGLE_CSE_ID')[:5]}", flush=True)

    compiled_graph = create_compiled_graph()
    if compiled_graph: # Ensure compilation was successful (e.g. GEMINI_API_KEY check inside create_compiled_graph)

        # Test Case 1: Topic Only Research
        print("\n--- Test Case 1: Topic Only ---", flush=True)
        topic_only_input = ResearchStateInput(
            topic="The future of serverless computing",
            research_approach="Topic Only",
            company_name=None,
            title_areas=None,
            video_url=None, # Or add a sample YouTube URL for video analysis part
            create_podcast=False # Or True to test podcast generation
        )
        print(f"Input: {topic_only_input}", flush=True)

        try:
            # Using stream to see progress, then extract final output
            events_topic_only = compiled_graph.stream(topic_only_input, {"recursion_limit": 100})
            final_output_topic_only = None
            for event in events_topic_only:
                # print(f"Event: {event['event']}, Data: {event['data']}", flush=True) # Detailed stream
                if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                    final_output_topic_only = event['data']['output']
                    break

            if final_output_topic_only:
                print("\nFinal Output (Topic Only):", flush=True)
                print(f"  Report: {final_output_topic_only.get('report')}", flush=True)
                print(f"  Podcast Script: {'Generated' if final_output_topic_only.get('podcast_script') else 'Not generated'}", flush=True)
                print(f"  Podcast URL: {final_output_topic_only.get('podcast_url')}", flush=True)
                print(f"  Identified Leads (Gemini): {final_output_topic_only.get('identified_leads')}", flush=True)
                print(f"  LinkedIn CSE Contacts: {final_output_topic_only.get('linkedin_cse_contacts')}", flush=True)
            else:
                print("No final output captured for Topic Only case.", flush=True)
        except Exception as e:
            print(f"Error running Topic Only case: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Test Case 2: Topic Company Leads Research
        print("\n--- Test Case 2: Topic Company Leads ---", flush=True)
        topic_company_leads_input = ResearchStateInput(
            topic="AI in Customer Relationship Management",
            research_approach="Topic Company Leads",
            company_name="Salesforce",
            title_areas=["VP of AI Product", "Director of CRM Innovation", "Lead AI Strategist"],
            video_url=None,
            create_podcast=False
        )
        print(f"Input: {topic_company_leads_input}", flush=True)

        try:
            events_company_leads = compiled_graph.stream(topic_company_leads_input, {"recursion_limit": 100})
            final_output_company_leads = None
            for event in events_company_leads:
                # print(f"Event: {event['event']}, Data: {event['data']}", flush=True) # Detailed stream
                if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                    final_output_company_leads = event['data']['output']
                    break

            if final_output_company_leads:
                print("\nFinal Output (Topic Company Leads):", flush=True)
                print(f"  Report: {final_output_company_leads.get('report')}", flush=True)
                print(f"  Podcast Script: {'Generated' if final_output_company_leads.get('podcast_script') else 'Not generated'}", flush=True)
                print(f"  Podcast URL: {final_output_company_leads.get('podcast_url')}", flush=True)
                print(f"  Identified Leads (Gemini) Count: {len(final_output_company_leads.get('identified_leads', []))}", flush=True)
                if final_output_company_leads.get('identified_leads'):
                    print(f"  First Gemini Lead Example: {json.dumps(final_output_company_leads.get('identified_leads')[0], indent=2)}", flush=True)
                print(f"  LinkedIn CSE Contacts Count: {len(final_output_company_leads.get('linkedin_cse_contacts', []))}", flush=True)
                if final_output_company_leads.get('linkedin_cse_contacts'):
                    print(f"  First LinkedIn CSE Contact Example: {json.dumps(final_output_company_leads.get('linkedin_cse_contacts')[0], indent=2)}", flush=True)
            else:
                print("No final output captured for Topic Company Leads case.", flush=True)

        except Exception as e:
            print(f"Error running Topic Company Leads case: {e}", flush=True)
            import traceback
            traceback.print_exc()