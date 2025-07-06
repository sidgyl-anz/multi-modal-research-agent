from typing_extensions import TypedDict
from typing import Optional

class ResearchStateInput(TypedDict):
    """Input state for the research and podcast generation workflow"""
    # Input fields
    topic: str
    video_url: Optional[str]
    create_podcast: bool # User preference to create a podcast or not

class ResearchStateOutput(TypedDict):
    """Output state for the research and podcast generation workflow"""
    # Final outputs
    report: Optional[str] # URL to the research report
    podcast_script: Optional[str] # Text script of the podcast
    podcast_url: Optional[str] # URL to the podcast audio file

class ResearchState(TypedDict):
    """Full state for the research and podcast generation workflow"""
    # Input fields (mirrored from ResearchStateInput for internal use)
    topic: str
    video_url: Optional[str]
    create_podcast: bool
    
    # Intermediate results
    search_text: Optional[str]
    search_sources_text: Optional[str]
    video_text: Optional[str]
    synthesis_text: Optional[str] # Text used for report/podcast generation
    
    # Final outputs (mirrored from ResearchStateOutput)
    report: Optional[str]
    podcast_script: Optional[str]
    podcast_url: Optional[str]