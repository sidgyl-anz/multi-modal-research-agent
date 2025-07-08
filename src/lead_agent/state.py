from typing_extensions import TypedDict
from typing import Optional, List, Dict

class LeadIdentificationStateInput(TypedDict):
    """Input state for the lead identification workflow"""
    company_name: str
    lead_generation_area: str
    titles: List[str]

class LeadIdentificationStateOutput(TypedDict):
    """Output state for the lead identification workflow"""
    leads: List[Dict]
    report: Optional[str]

class LeadIdentificationState(TypedDict):
    """Full state for the lead identification workflow"""
    # Input fields
    company_name: str
    lead_generation_area: str
    titles: List[str]

    # Intermediate results
    raw_search_results: Optional[str]
    processed_leads: Optional[List[Dict]]

    # Final outputs
    leads: List[Dict]
    report: Optional[str]
