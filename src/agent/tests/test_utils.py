import unittest
import json
from unittest.mock import MagicMock, patch

import os # Ensure os is imported for patch.dict(os.environ, ...)
# Need to adjust import paths based on how tests are run.
# Using relative imports assuming tests are run as part of the agent package.
from ..utils import (
    generate_company_topic_research_prompt,
    generate_lead_identification_prompt,
    parse_leads_from_gemini_response,
    create_research_report,
    build_linkedin_cse_query,          # New import
    fetch_linkedin_contacts_via_cse    # New import
    # display_gemini_response # if we want to test its parsing logic for simple text
)
from ..configuration import Configuration # For testing create_research_report with config

# Mock requests for fetch_linkedin_contacts_via_cse
class MockRequestsResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data) if json_data is not None else ""

    def json(self):
        if self.json_data is None:
            raise json.JSONDecodeError("No JSON data", "", 0)
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            # Simplified error, actual requests.HTTPError is more complex
            raise Exception(f"HTTP Error {self.status_code}")

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
        mock_response.text = None

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
        mock_response.text = None # Ensure path A is skipped

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
        mock_response.text = None # Ensure path A is skipped

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
        mock_response.text = None # Ensure path A is skipped

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
        mock_response.text = None # Ensure path A is skipped

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0) # Expect empty list due to type mismatch

    def test_parse_leads_from_gemini_response_empty_response_text(self):
        mock_response = MagicMock()
        mock_response.text = "" # This is what we want to test, Path A with empty string
        # mock_part = MagicMock() # Not needed for this test
        # mock_part.text = ""      # Not needed
        # mock_content = MagicMock() # Not needed
        # mock_content.parts = [mock_part] # Not needed
        # mock_candidate = MagicMock() # Not needed
        # mock_candidate.content = mock_content # Not needed
        mock_response.candidates = None # Explicitly ensure Path B is not taken

        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0)

    def test_parse_leads_from_gemini_response_no_candidates(self):
        mock_response = MagicMock()
        mock_response.text = None # Ensure path A is skipped
        mock_response.candidates = [] # No candidates, so path B is skipped
        # Should hit path C and return []
        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0)

    def test_parse_leads_from_gemini_response_no_parts(self):
        mock_response = MagicMock()
        mock_response.text = None # Ensure path A is skipped
        mock_content = MagicMock()
        mock_content.parts = [] # No parts, so raw_text remains "" from Path B's default assignment
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        # Path B will be taken, but inner if '...parts[0].text' will effectively not run if parts is empty
        # raw_text remains "" as initialized inside the function (or from the part before .text if parts not empty but .text is not there)
        # The function should then try to parse "" which results in an empty list.
        leads = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(leads), 0)

    # Test for create_research_report (focus on prompt construction and data inclusion)
    # This is more of an integration test for the utility function itself.
    # We'll mock the actual genai_client.models.generate_content call within it.
    @patch('src.agent.utils.genai_client') # Corrected patch target
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


    @patch('src.agent.utils.genai_client') # Corrected patch target
    @patch('src.agent.utils.storage.Client') # Corrected patch target
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

    def test_build_linkedin_cse_query(self):
        company_name = "Test Inc."
        title_areas = ["Software Engineer", "Product Manager"]
        expected_query = 'site:linkedin.com/in ("Test Inc.") ("Software Engineer" OR "Product Manager")'
        query = build_linkedin_cse_query(company_name, title_areas)
        self.assertEqual(query, expected_query)

    @patch('src.agent.utils.requests.get')
    def test_fetch_linkedin_contacts_via_cse_success(self, mock_requests_get):
        mock_api_key = "test_cse_api_key"
        mock_cse_id = "test_cse_id"
        query = "test_query"

        # Mock successful API response
        mock_response_data = {
            "items": [
                {"title": "Profile 1 - Test Inc.", "link": "http://linkedin.com/in/profile1", "snippet": "Snippet 1"},
                {"title": "Profile 2 - Test Inc.", "link": "http://linkedin.com/in/profile2", "snippet": "Snippet 2"}
            ]
        }
        mock_requests_get.return_value = MockRequestsResponse(json_data=mock_response_data, status_code=200)

        contacts = fetch_linkedin_contacts_via_cse(query, mock_api_key, mock_cse_id)

        self.assertEqual(len(contacts), 2)
        self.assertEqual(contacts[0]['title'], "Profile 1 - Test Inc.")
        mock_requests_get.assert_called_once()
        called_url = mock_requests_get.call_args[0][0]
        called_params = mock_requests_get.call_args[1]['params']
        self.assertEqual(called_url, "https://www.googleapis.com/customsearch/v1")
        self.assertEqual(called_params['q'], query)
        self.assertEqual(called_params['key'], mock_api_key)
        self.assertEqual(called_params['cx'], mock_cse_id)

    @patch('src.agent.utils.requests.get')
    def test_fetch_linkedin_contacts_via_cse_http_error(self, mock_requests_get):
        mock_api_key = "test_cse_api_key"
        mock_cse_id = "test_cse_id"
        query = "test_query_http_error"

        mock_requests_get.return_value = MockRequestsResponse(json_data={"error": "bad request"}, status_code=400)

        contacts = fetch_linkedin_contacts_via_cse(query, mock_api_key, mock_cse_id)
        self.assertEqual(len(contacts), 0) # Expect empty list on error

    @patch('src.agent.utils.requests.get')
    def test_fetch_linkedin_contacts_via_cse_no_items(self, mock_requests_get):
        mock_api_key = "test_cse_api_key"
        mock_cse_id = "test_cse_id"
        query = "test_query_no_items"

        mock_requests_get.return_value = MockRequestsResponse(json_data={"items": []}, status_code=200)

        contacts = fetch_linkedin_contacts_via_cse(query, mock_api_key, mock_cse_id)
        self.assertEqual(len(contacts), 0)

    @patch('src.agent.utils.requests.get')
    def test_fetch_linkedin_contacts_via_cse_request_exception(self, mock_requests_get):
        mock_api_key = "test_cse_api_key"
        mock_cse_id = "test_cse_id"
        query = "test_query_request_exception"

        # Simulate a requests.exceptions.RequestException (e.g., network error)
        mock_requests_get.side_effect = Exception("Network Error") # requests.exceptions.RequestException("Network Error")

        contacts = fetch_linkedin_contacts_via_cse(query, mock_api_key, mock_cse_id)
        self.assertEqual(len(contacts), 0)


if __name__ == '__main__':
    unittest.main()
