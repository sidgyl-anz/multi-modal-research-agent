"""Configuration settings for the lead identification app"""

import os
from dataclasses import dataclass, fields
from typing import Optional, Any
from langchain_core.runnables import RunnableConfig

@dataclass(kw_only=True)
class LeadIdentificationConfiguration:
    """LangGraph Configuration for the lead identification agent."""

    # Model settings
    lead_search_model: str = "gemini-1.5-flash-latest"
    report_generation_model: str = "gemini-1.5-flash-latest"

    # Temperature settings
    lead_search_temperature: float = 0.2  # For more factual, structured output
    report_generation_temperature: float = 0.5 # For a bit more creativity in reporting

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "LeadIdentificationConfiguration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
