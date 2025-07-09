from typing_extensions import TypedDict, Literal
from typing import Optional, List, Dict

# Define the research approach type
ResearchApproach = Literal["Topic Only", "Topic Company Leads"]

class ResearchStateInput(TypedDict):
    """Input state for the research and podcast generation workflow"""
    # Input fields
    topic: str # Mandatory
    research_approach: ResearchApproach
    company_name: Optional[str]
    title_areas: Optional[List[str]]
    video_url: Optional[str]
    create_podcast: bool # User preference to create a podcast or not

class ResearchStateOutput(TypedDict):
    """Output state for the research and podcast generation workflow"""
    # Final outputs
    report: Optional[str] # Comprehensive report text
    podcast_script: Optional[str] # Text script of the podcast
    podcast_url: Optional[str] # URL to the podcast audio file
    identified_leads: Optional[List[Dict]] # List of identified leads with details

class ResearchState(TypedDict):
    """Full state for the research and podcast generation workflow"""
    # Input fields (mirrored from ResearchStateInput for internal use)
    topic: str
    research_approach: ResearchApproach
    company_name: Optional[str]
    title_areas: Optional[List[str]]
    video_url: Optional[str]
    create_podcast: bool
    
    # Intermediate results for topic research
    search_text: Optional[str]
    search_sources_text: Optional[str]

    # Intermediate results for video analysis
    video_text: Optional[str]

    # Intermediate results for company & lead research
    company_specific_topic_research_text: Optional[str]
    company_info_text: Optional[str]
    identified_leads_data: Optional[List[Dict]] # Raw data from lead identification step

    # Common intermediate result for synthesis
    synthesis_text: Optional[str] # Text used for report/podcast generation
    
    # Final outputs (mirrored from ResearchStateOutput)
    report: Optional[str]
    podcast_script: Optional[str]
    podcast_url: Optional[str]
    identified_leads: Optional[List[Dict]]
