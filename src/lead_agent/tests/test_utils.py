import unittest
import json
from unittest.mock import MagicMock

from ..utils import (
    generate_lead_identification_prompt,
    generate_summary_report_prompt,
    parse_gemini_json_response
)

class TestLeadAgentUtils(unittest.TestCase):

    def test_generate_lead_identification_prompt(self):
        company_name = "TestCorp"
        lead_generation_area = "Software Development"
        titles = ["Software Engineer", "Senior Developer"]
        prompt = generate_lead_identification_prompt(company_name, lead_generation_area, titles)

        self.assertIn(company_name, prompt)
        self.assertIn(lead_generation_area, prompt)
        self.assertIn("Software Engineer", prompt)
        self.assertIn("Senior Developer", prompt)
        self.assertIn("JSON list format", prompt) # Check for JSON instruction
        self.assertIn("Use Google Search", prompt) # Check for search instruction

    def test_generate_summary_report_prompt(self):
        company_name = "TestCorp"
        lead_generation_area = "Marketing"
        titles = ["Marketing Manager"]
        leads = [
            {"name": "John Doe", "title": "Marketing Manager", "summary": "A good lead."}
        ]
        prompt = generate_summary_report_prompt(company_name, lead_generation_area, titles, leads)

        self.assertIn(company_name, prompt)
        self.assertIn(lead_generation_area, prompt)
        self.assertIn("Marketing Manager", prompt)
        self.assertIn("John Doe", prompt)
        self.assertIn("A good lead.", prompt)
        self.assertIn("Markdown format", prompt)

    def test_generate_summary_report_prompt_no_leads(self):
        company_name = "TestCorp"
        lead_generation_area = "Sales"
        titles = ["Sales Executive"]
        leads = []
        prompt = generate_summary_report_prompt(company_name, lead_generation_area, titles, leads)

        self.assertIn(company_name, prompt)
        self.assertIn("No specific leads were identified", prompt)

    def test_parse_gemini_json_response_correct_json_in_text(self):
        mock_response = MagicMock()
        mock_response.parts = None # Test the direct .text attribute path first
        mock_response.text = json.dumps([{"name": "Lead1"}, {"name": "Lead2"}])

        leads, raw_text = parse_gemini_json_response(mock_response)
        self.assertEqual(len(leads), 2)
        self.assertEqual(leads[0]["name"], "Lead1")
        self.assertEqual(raw_text, mock_response.text)

    def test_parse_gemini_json_response_correct_json_in_parts(self):
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = json.dumps([{"name": "Lead1"}, {"name": "Lead2"}])
        mock_part.function_call = None # Ensure no function call
        mock_response.parts = [mock_part]
        mock_response.text = None # Ensure it uses parts

        leads, raw_text = parse_gemini_json_response(mock_response)
        self.assertEqual(len(leads), 2)
        self.assertEqual(leads[0]["name"], "Lead1")
        self.assertEqual(raw_text.strip(), mock_part.text)


    def test_parse_gemini_json_response_json_in_markdown_block(self):
        mock_response = MagicMock()
        json_data = [{"name": "Lead Markdown"}]
        mock_response.text = f"Some introductory text.\n```json\n{json.dumps(json_data)}\n```\nSome trailing text."
        mock_response.parts = None

        leads, raw_text = parse_gemini_json_response(mock_response)
        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["name"], "Lead Markdown")
        self.assertEqual(raw_text, mock_response.text)

    def test_parse_gemini_json_response_malformed_json(self):
        mock_response = MagicMock()
        mock_response.text = "[{'name': 'Lead1'}," # Malformed JSON
        mock_response.parts = None

        leads, raw_text = parse_gemini_json_response(mock_response)
        self.assertEqual(len(leads), 0)
        self.assertIn("Error: Could not parse JSON from response", raw_text)

    def test_parse_gemini_json_response_empty_text(self):
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.parts = None

        leads, raw_text = parse_gemini_json_response(mock_response)
        self.assertEqual(len(leads), 0)
        self.assertIn("Error: Could not parse JSON from response", raw_text) # As it tries json.loads("")

    def test_parse_gemini_json_response_no_text_no_parts(self):
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.parts = [] # Empty parts list

        leads, raw_text = parse_gemini_json_response(mock_response)
        self.assertEqual(len(leads), 0)
        self.assertIn("Error: Could not parse JSON from response", raw_text)

    def test_parse_gemini_json_response_not_a_list(self):
        mock_response = MagicMock()
        # Gemini returns a single JSON object instead of a list
        mock_response.text = json.dumps({"name": "Lead1", "title": "CEO"})
        mock_response.parts = None

        leads, raw_text = parse_gemini_json_response(mock_response)
        self.assertEqual(len(leads), 0)
        self.assertIn("Error: Parsed JSON is not in the expected list format.", raw_text)


if __name__ == '__main__':
    unittest.main()
