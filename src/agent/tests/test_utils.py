import unittest
import json
from unittest.mock import MagicMock, patch

# Need to adjust import paths based on how tests are run.
# Using relative imports assuming tests are run as part of the agent package.
from ..utils import (
    generate_company_topic_research_prompt,
    generate_lead_identification_prompt,
    parse_leads_from_gemini_response,
    create_research_report
    # display_gemini_response # if we want to test its parsing logic for simple text
)
from ..configuration import Configuration # For testing create_research_report with config

class TestAgentUtils(unittest.TestCase):

    def test_generate_company_topic_research_prompt(self):
        topic = "AI in Healthcare"
        company_name = "FutureHealth Corp"
        prompt = generate_company_topic_research_prompt(topic, company_name)
        self.assertIn(topic, prompt)
        self.assertIn(company_name, prompt)
        self.assertIn("Topic Research in Company Context", prompt)
        self.assertIn("General Company Information", prompt)

    def test_generate_lead_identification_prompt(self):
        company_name = "Innovatech Ltd"
        title_areas = ["VP of Engineering", "Chief Architect"]
        company_topic_context = "Innovatech is expanding its cloud services division."
        prompt = generate_lead_identification_prompt(company_name, title_areas, company_topic_context)

        self.assertIn(company_name, prompt)
        self.assertIn("VP of Engineering", prompt)
        self.assertIn("Chief Architect", prompt)
        self.assertIn(company_topic_context[:100], prompt) # Check if context is included
        self.assertIn("lead_name", prompt)
        self.assertIn("lead_title", prompt)
        self.assertIn("lead_department", prompt)
        self.assertIn("named_buyers", prompt)
        self.assertIn("buyer_name", prompt)
        self.assertIn("buyer_title", prompt)
        self.assertIn("Use Google Search", prompt)
        self.assertIn("Return *only* the JSON list", prompt)

    def test_parse_leads_from_gemini_response_correct_json(self):
        mock_response = MagicMock()
        lead_data = [{"lead_name": "John Doe", "lead_title": "CEO"}]
        # Simulate Gemini response structure (candidates[0].content.parts[0].text)
        mock_part = MagicMock()
        mock_part.text = json.dumps(lead_data)
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_response.text = None # Ensure it doesn't use a direct .text if .candidates is present

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["lead_name"], "John Doe")

    def test_parse_leads_from_gemini_response_direct_text_json(self):
        mock_response = MagicMock()
        lead_data = [{"lead_name": "Jane Alex", "lead_title": "CTO"}]
        mock_response.text = json.dumps(lead_data)
        mock_response.candidates = None # Ensure it uses direct .text

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["lead_name"], "Jane Alex")


    def test_parse_leads_from_gemini_response_json_in_markdown(self):
        mock_response = MagicMock()
        lead_data = [{"lead_name": "Mark Down"}]
        mock_part = MagicMock()
        mock_part.text = f"Some text before\n```json\n{json.dumps(lead_data)}\n```\nSome text after"
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["lead_name"], "Mark Down")

    def test_parse_leads_from_gemini_response_malformed_json(self):
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "[{'name': 'Lead1'}," # Malformed
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0)
        # Add assertion for logged error if possible, or ensure it doesn't raise unhandled exception

    def test_parse_leads_from_gemini_response_empty_list(self):
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "[]"
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0)

    def test_parse_leads_from_gemini_response_not_a_list(self):
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = json.dumps({"error": "not a list"})
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0) # Expect empty list due to type mismatch

    def test_parse_leads_from_gemini_response_empty_response_text(self):
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = ""
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0)

    def test_parse_leads_from_gemini_response_no_candidates(self):
        mock_response = MagicMock()
        mock_response.candidates = [] # No candidates
        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0)

    def test_parse_leads_from_gemini_response_no_parts(self):
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.parts = [] # No parts
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0)

    # Test for create_research_report (focus on prompt construction and data inclusion)
    # This is more of an integration test for the utility function itself.
    # We'll mock the actual genai_client.models.generate_content call within it.
    @patch('agent.utils.genai_client') # Mock the global genai_client used in utils
    def test_create_research_report_topic_only(self, mock_genai_client):
        # Mock Gemini's response for synthesis
        mock_gemini_synthesis_response = MagicMock()
        mock_synthesis_part = MagicMock()
        mock_synthesis_part.text = "Synthesized topic research."
        mock_synthesis_content = MagicMock()
        mock_synthesis_content.parts = [mock_synthesis_part]
        mock_synthesis_candidate = MagicMock()
        mock_synthesis_candidate.content = mock_synthesis_content
        mock_gemini_synthesis_response.candidates = [mock_synthesis_candidate]
        mock_genai_client.models.generate_content.return_value = mock_gemini_synthesis_response

        config = Configuration()
        report_url_or_text, synthesis_text = create_research_report(
            topic="Test Topic",
            research_approach="Topic Only",
            search_text="Some search text.",
            video_text="Some video text.",
            search_sources_text="Source 1",
            video_url="http://example.com/video",
            configuration=config
        )
        self.assertIn("Synthesized topic research.", synthesis_text)
        self.assertIn("# Research Synthesis on 'Test Topic'", report_url_or_text)
        self.assertNotIn("Identified Leads Summary", report_url_or_text)
        self.assertNotIn("COMPANY-SPECIFIC", mock_genai_client.models.generate_content.call_args[1]['contents'])


    @patch('agent.utils.genai_client')
    @patch('agent.utils.storage.Client') # Mock GCS client
    def test_create_research_report_topic_company_leads(self, mock_gcs_client, mock_genai_client):
        # Mock Gemini's response for synthesis
        mock_gemini_synthesis_response = MagicMock()
        mock_synthesis_part = MagicMock()
        mock_synthesis_part.text = "Synthesized company and lead research."
        mock_synthesis_content = MagicMock()
        mock_synthesis_content.parts = [mock_synthesis_part]
        mock_synthesis_candidate = MagicMock()
        mock_synthesis_candidate.content = mock_synthesis_content
        mock_gemini_synthesis_response.candidates = [mock_synthesis_candidate]
        mock_genai_client.models.generate_content.return_value = mock_gemini_synthesis_response

        # Mock GCS
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_gcs_client.return_value.bucket.return_value = mock_bucket
        mock_blob.generate_signed_url.return_value = "http://gcs.example.com/report.md"


        config = Configuration()
        identified_leads = [
            {"lead_name": "Lead Alpha", "lead_title": "Big Boss", "lead_department": "Strategy",
             "linkedin_url": "http://linkedin.com/alpha", "summary_of_relevance": "Very relevant",
             "named_buyers": [{"buyer_name": "Buyer One", "buyer_title": "Money Bags", "buyer_rationale": "Signs checks"}]}
        ]

        with patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}): # Mock GCS bucket env var
            report_url_or_text, synthesis_text = create_research_report(
                topic="AI Solutions",
                research_approach="Topic Company Leads",
                search_text=None, # No general search text for this path usually
                video_text="Video about AI.",
                search_sources_text=None, # No general search sources
                video_url="http://example.com/ai_video",
                company_name="LeadGen Corp",
                company_specific_topic_research_text="LeadGen Corp is exploring AI.",
                company_info_text="LeadGen Corp is a leader.",
                identified_leads_data=identified_leads,
                configuration=config
            )

        self.assertIn("Synthesized company and lead research.", synthesis_text)
        self.assertEqual("http://gcs.example.com/report.md", report_url_or_text) # Expecting GCS URL

        # Check that the prompt to Gemini contained company/lead related keywords
        synthesis_prompt_sent_to_gemini = mock_genai_client.models.generate_content.call_args[1]['contents']
        self.assertIn("COMPANY-SPECIFIC TOPIC RESEARCH (LeadGen Corp)", synthesis_prompt_sent_to_gemini)
        self.assertIn("GENERAL COMPANY INFORMATION (LeadGen Corp)", synthesis_prompt_sent_to_gemini)
        self.assertIn("IDENTIFIED LEADS AT LeadGen Corp", synthesis_prompt_sent_to_gemini)
        self.assertIn("Lead Alpha", synthesis_prompt_sent_to_gemini) # Check if lead data was in prompt
        self.assertIn("Buyer One", synthesis_prompt_sent_to_gemini)


if __name__ == '__main__':
    unittest.main()
