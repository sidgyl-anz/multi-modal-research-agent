"""LangGraph implementation of the lead identification workflow"""

import os
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from .state import LeadIdentificationState, LeadIdentificationStateInput, LeadIdentificationStateOutput
from .configuration import LeadIdentificationConfiguration
from .utils import (
    generate_lead_identification_prompt,
    generate_summary_report_prompt,
    parse_gemini_json_response,
    get_gemini_model,
    call_gemini_api
)

# Define the Google Search tool for Gemini
# This matches the structure used in the research agent
GOOGLE_SEARCH_TOOL = [{"google_search": {}}]

@traceable(run_type="llm", name="Identify Leads", project_name="lead-identification-agent")
def identify_leads_node(state: LeadIdentificationState, config: RunnableConfig) -> dict:
    """Node that uses Gemini to identify leads based on input criteria."""
    configuration = LeadIdentificationConfiguration.from_runnable_config(config)

    company_name = state["company_name"]
    lead_generation_area = state["lead_generation_area"]
    titles = state["titles"]

    prompt = generate_lead_identification_prompt(company_name, lead_generation_area, titles)

    gemini_model = get_gemini_model(configuration.lead_search_model, configuration)

    print(f"DEBUG (identify_leads_node): Calling Gemini for lead identification. Model: {configuration.lead_search_model}, Temp: {configuration.lead_search_temperature}", flush=True)

    try:
        response = call_gemini_api(
            model=gemini_model,
            prompt=prompt,
            temperature=configuration.lead_search_temperature,
            is_json_output_expected=True, # Hint to Gemini to output JSON
            tools=GOOGLE_SEARCH_TOOL # Enable Google Search tool
        )

        # --- DEBUG: Print Gemini Raw Response ---
        # This helps understand what Gemini is actually returning.
        # The content of 'response' depends on the Gemini library version and how call_gemini_api structures it.
        # Assuming response has a 'text' attribute or can be iterated for parts.
        if hasattr(response, 'text'):
            print(f"DEBUG (identify_leads_node): Gemini raw response text: {response.text}", flush=True)
        elif hasattr(response, 'parts'):
            full_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            print(f"DEBUG (identify_leads_node): Gemini raw response parts text: {full_response_text}", flush=True)
            # Also print if there are function calls, which might contain the data
            for part in response.parts:
                if part.function_call:
                    print(f"DEBUG (identify_leads_node): Gemini function call: {part.function_call}", flush=True)

        else:
            print(f"DEBUG (identify_leads_node): Gemini response object type: {type(response)}, content: {response}", flush=True)


        # The parse_gemini_json_response function expects the Gemini response object
        processed_leads, raw_response_text = parse_gemini_json_response(response)

        if not processed_leads and "Error:" in raw_response_text:
            print(f"ERROR (identify_leads_node): Failed to parse leads. Error: {raw_response_text}", flush=True)
            # Return empty list and raw text to allow flow to continue to reporting if needed
            return {
                "raw_search_results": raw_response_text,
                "processed_leads": [],
                "leads": [] # Ensure leads is also empty
            }

        print(f"DEBUG (identify_leads_node): Successfully parsed {len(processed_leads)} leads.", flush=True)
        return {
            "raw_search_results": raw_response_text, # Storing the raw text for audit/debug
            "processed_leads": processed_leads,
            "leads": processed_leads # Directly use processed leads as final leads
        }

    except Exception as e:
        print(f"ERROR (identify_leads_node): Exception during Gemini call or parsing: {e}", flush=True)
        # Fallback in case of unexpected error
        return {
            "raw_search_results": f"Exception: {str(e)}",
            "processed_leads": [],
            "leads": []
        }


@traceable(run_type="llm", name="Create Summary Report", project_name="lead-identification-agent")
def create_summary_report_node(state: LeadIdentificationState, config: RunnableConfig) -> dict:
    """Node that creates a summary report of the lead identification process."""
    configuration = LeadIdentificationConfiguration.from_runnable_config(config)

    company_name = state["company_name"]
    lead_generation_area = state["lead_generation_area"]
    titles = state["titles"]
    processed_leads = state.get("processed_leads") # This should be populated by identify_leads_node

    if processed_leads is None: # Should not happen if graph is correct
        print("WARN (create_summary_report_node): processed_leads is None, using empty list for report.", flush=True)
        processed_leads = []

    prompt = generate_summary_report_prompt(company_name, lead_generation_area, titles, processed_leads)

    gemini_model = get_gemini_model(configuration.report_generation_model, configuration)

    print(f"DEBUG (create_summary_report_node): Calling Gemini for summary report. Model: {configuration.report_generation_model}, Temp: {configuration.report_generation_temperature}", flush=True)

    try:
        response = call_gemini_api(
            model=gemini_model,
            prompt=prompt,
            temperature=configuration.report_generation_temperature,
            is_json_output_expected=False # Report is text/markdown
        )

        report_text = ""
        if hasattr(response, 'text') and response.text:
            report_text = response.text
        elif response.parts:
            for part in response.parts:
                if hasattr(part, 'text'):
                    report_text += part.text

        report_text = report_text.strip()

        if not report_text:
            report_text = "Failed to generate report text from Gemini response."
            print(f"ERROR (create_summary_report_node): {report_text}", flush=True)

        print(f"DEBUG (create_summary_report_node): Report generated successfully.", flush=True)
        return {"report": report_text}

    except Exception as e:
        print(f"ERROR (create_summary_report_node): Exception during Gemini call for report: {e}", flush=True)
        return {"report": f"Exception during report generation: {str(e)}"}


def create_lead_identification_graph() -> StateGraph:
    """Create and return the lead identification workflow graph."""

    graph = StateGraph(
        LeadIdentificationState,
        input=LeadIdentificationStateInput,
        output=LeadIdentificationStateOutput,
        config_schema=LeadIdentificationConfiguration
    )

    # Add nodes
    # Removed extract_company_info_node for simplicity, direct to lead identification
    graph.add_node("identify_leads", identify_leads_node)
    graph.add_node("create_summary_report", create_summary_report_node)

    # Add edges
    graph.add_edge(START, "identify_leads")
    graph.add_edge("identify_leads", "create_summary_report")
    graph.add_edge("create_summary_report", END)

    return graph

def create_compiled_lead_identification_graph():
    """Create and compile the lead identification graph."""
    graph = create_lead_identification_graph()
    # It's good practice to ensure the API key is configured before compilation
    # or at least before the graph is run. utils.py now handles genai.configure()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set. Cannot compile graph.")
    return graph.compile()

# Example of how to run (for testing purposes, not part of the library code)
if __name__ == '__main__':
    # This section is for local testing if you run this file directly.
    # Ensure GEMINI_API_KEY is set in your environment.
    print("Attempting to run lead identification graph example...", flush=True)

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Please set it to run the example.", flush=True)
    else:
        print(f"GEMINI_API_KEY found, starting with: {os.getenv('GEMINI_API_KEY')[:5]}", flush=True)

        compiled_graph = create_compiled_lead_identification_graph()

        # Example input
        inputs = LeadIdentificationStateInput(
            company_name="Google",
            lead_generation_area="AI Research",
            titles=["Research Scientist", "AI Ethicist"]
        )

        # LangGraph configuration (optional, defaults can be used)
        # This config is passed to each node via the `config` parameter in the node function
        run_config = RunnableConfig(
            configurable={
                # "lead_search_model": "gemini-1.5-pro-latest", # Example override
                # "report_generation_model": "gemini-1.5-pro-latest"
            }
        )

        print(f"Invoking graph with inputs: {inputs}", flush=True)

        try:
            # Use .stream() for observing intermediate states or .invoke() for final result
            # output = compiled_graph.invoke(inputs, config=run_config)

            # Using stream to see progress:
            events = compiled_graph.stream(inputs, config=run_config)
            final_output = None
            for event in events:
                # event is a dictionary, event_type is event['event']
                # The actual data payload is often in event['data']['chunk'] for streaming updates
                # or event['data']['output'] for node outputs when not streaming node internals.
                # LangGraph's stream events can be a bit complex.
                # We're interested in the final output or outputs of specific nodes.

                # Print all events to understand the structure
                # print(f"Event: {event}\n", flush=True)

                if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                     # The final output is in event['data']['output']
                     final_output = event['data']['output']
                     break # Found the final output

            if final_output:
                print("\n--- Final Output ---", flush=True)
                print(f"Leads: {json.dumps(final_output.get('leads'), indent=2)}", flush=True)
                print(f"Report:\n{final_output.get('report')}", flush=True)
            else:
                print("No final output captured from stream. Check event structure.", flush=True)

        except Exception as e:
            print(f"Error running the graph: {e}", flush=True)
            import traceback
            traceback.print_exc()
