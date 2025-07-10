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
        self.assertIn("VP of Engineering", prompt) # title_areas are still in the prompt as a hint
        self.assertIn("Chief Architect", prompt)   # title_areas are still in the prompt as a hint
        self.assertIn(company_topic_context[:100], prompt) # Check if context is included

        # Check for new B2B opportunity keys
        self.assertIn("opportunity_name", prompt)
        self.assertIn("opportunity_description", prompt)
        self.assertIn("relevant_departments", prompt)
        self.assertIn("contact_points", prompt)
        self.assertIn("contact_name", prompt)
        self.assertIn("contact_title", prompt)
        self.assertIn("contact_department", prompt)
        self.assertIn("contact_linkedin_url", prompt)
        self.assertIn("contact_relevance_to_opportunity", prompt)
        self.assertIn("potential_decision_makers_for_opportunity", prompt)
        self.assertIn("dm_name", prompt)
        self.assertIn("dm_title", prompt)
        self.assertIn("dm_rationale", prompt)

        self.assertIn("Use Google Search", prompt)
        self.assertIn("The entire response MUST be a single valid JSON list.", prompt) # Corrected assertion
        self.assertIn("B2B Sales Opportunity Identification AI", prompt) # Check role play

    def test_parse_leads_from_gemini_response_correct_json(self):
        mock_response = MagicMock()
        # New B2B Opportunity structure
        opportunity_data = [
            {
                "opportunity_name": "AI CRM Upgrade",
                "opportunity_description": "Upgrade existing CRM with AI.",
                "relevant_departments": ["Sales", "IT"],
                "contact_points": [
                    {"contact_name": "John Doe", "contact_title": "Sales Manager", "contact_department": "Sales", "contact_linkedin_url": "linkedin.com/johndoe", "contact_relevance_to_opportunity": "Manages sales team"}
                ],
                "potential_decision_makers_for_opportunity": [
                    {"dm_name": "Jane CEO", "dm_title": "CEO", "dm_rationale": "Overall budget holder"}
                ]
            }
        ]
        mock_part = MagicMock()
        # Correctly assign opportunity_data to mock_part.text ONCE
        mock_part.text = json.dumps(opportunity_data)
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_response.text = None

        opportunities = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0]["opportunity_name"], "AI CRM Upgrade")
        self.assertEqual(opportunities[0]["contact_points"][0]["contact_name"], "John Doe")

    def test_parse_leads_from_gemini_response_direct_text_json(self):
        mock_response = MagicMock()
        opportunity_data = [
            {
                "opportunity_name": "Direct AI CRM Upgrade",
                "contact_points": [{"contact_name": "Alex Direct"}]
            }
        ]
        mock_response.text = json.dumps(opportunity_data)
        mock_response.candidates = None # Ensure it uses direct .text

        opportunities = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0]["opportunity_name"], "Direct AI CRM Upgrade")
        self.assertEqual(opportunities[0]["contact_points"][0]["contact_name"], "Alex Direct")


    def test_parse_leads_from_gemini_response_json_in_markdown(self):
        mock_response = MagicMock()
        opportunity_data = [{"opportunity_name": "Markdown Opp"}]
        mock_part = MagicMock()
        mock_part.text = f"Some text before\n```json\n{json.dumps(opportunity_data)}\n```\nSome text after"
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        mock_response.text = None # Ensure path A is skipped

        opportunities = parse_leads_from_gemini_response(mock_response)
        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0]["opportunity_name"], "Markdown Opp")

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
        # New B2B Opportunity structure for identified_leads_data
        b2b_opportunities_data = [
            {
                "opportunity_name": "AI CRM Upgrade for LeadGen Corp",
                "opportunity_description": "Upgrade existing CRM with AI capabilities to improve sales forecasting.",
                "relevant_departments": ["Sales", "IT", "Product"],
                "contact_points": [
                    {"contact_name": "John Contact", "contact_title": "Sales Operations Lead", "contact_department": "Sales", "contact_linkedin_url": "linkedin.com/johncontact", "contact_relevance_to_opportunity": "Manages CRM usage and sales process efficiency."}
                ],
                "potential_decision_makers_for_opportunity": [
                    {"dm_name": "Alice Approver", "dm_title": "VP of Sales", "dm_rationale": "Budget holder for sales tools and CRM."},
                    {"dm_name": "Bob Budget", "dm_title": "CTO", "dm_rationale": "Oversees technology stack including CRM infrastructure."}
                ]
            }
        ]

        # Mock CSE contacts data as well, since create_research_report now accepts it
        mock_cse_contacts = [
            {"title": "CSE Contact 1", "link": "http://linkedin.com/cse1", "snippet": "Found via CSE."}
        ]

        with patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}): # Mock GCS bucket env var
            report_url_or_text, synthesis_text = create_research_report(
                topic="AI Solutions",
                research_approach="Topic Company Leads",
                search_text=None,
                video_text="Video about AI.",
                search_sources_text=None,
                video_url="http://example.com/ai_video",
                company_name="LeadGen Corp",
                company_specific_topic_research_text="LeadGen Corp is exploring AI for CRM.",
                company_info_text="LeadGen Corp is a leader in innovative sales solutions.",
                identified_leads_data=b2b_opportunities_data, # Pass new structure
                linkedin_cse_contacts=mock_cse_contacts,     # Pass mock CSE contacts
                configuration=config
            )

        self.assertIn("Synthesized company and lead research.", synthesis_text)
        self.assertEqual("http://gcs.example.com/report.md", report_url_or_text)

        # Check prompt to Gemini for synthesis
        synthesis_prompt_sent_to_gemini = mock_genai_client.models.generate_content.call_args[1]['contents']
        self.assertIn("COMPANY-SPECIFIC TOPIC RESEARCH (LeadGen Corp)", synthesis_prompt_sent_to_gemini)
        self.assertIn("GENERAL COMPANY INFORMATION (LeadGen Corp)", synthesis_prompt_sent_to_gemini)
        self.assertIn("SUMMARY OF IDENTIFIED B2B OPPORTUNITIES", synthesis_prompt_sent_to_gemini) # Check for new summary type
        self.assertIn("AI CRM Upgrade for LeadGen Corp", synthesis_prompt_sent_to_gemini)
        self.assertIn("John Contact", synthesis_prompt_sent_to_gemini) # Check contact point in prompt

        # Check final report content for new sections and details
        # This part of the test might need access to the actual report_content if GCS fails or is not used.
        # For now, assuming GCS works and report_url_or_text is the URL.
        # To test the actual content, we'd need to modify create_research_report to return text if GCS_BUCKET_NAME is not set.
        # Let's assume the GCS part is mocked correctly and focus on the prompt.
        # If we had the report_content string, we would:
        # self.assertIn("## Identified B2B Opportunities", report_content_string)
        # self.assertIn("Opportunity 1: AI CRM Upgrade for LeadGen Corp", report_content_string)
        # self.assertIn("John Contact", report_content_string)
        # self.assertIn("Alice Approver", report_content_string)
        # self.assertIn("## LinkedIn Contacts (via Custom Search)", report_content_string)
        # self.assertIn("CSE Contact 1", report_content_string)


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
