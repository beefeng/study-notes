"""Unit checks for environment loading and SQL AI output validation."""
import os
import unittest
from unittest.mock import patch

from i3_ai_functions_lab import _required, validate_summary


class ExperimentConfigTests(unittest.TestCase):
    """Verify safe configuration errors without making model requests."""

    def test_required_setting_fails_with_actionable_name(self):
        with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
            _required({}, "LLM_API_KEY")

    def test_summary_contract_rejects_invalid_output(self):
        with self.assertRaises(ValueError):
            validate_summary("not-json")
        with self.assertRaises(ValueError):
            validate_summary('{"summary":"x","sentiment":"unknown","review_count":1}')

    def test_summary_contract_accepts_valid_output(self):
        result = validate_summary('{"summary":"good","sentiment":"positive","review_count":2}')
        self.assertEqual(result["review_count"], 2)


if __name__ == "__main__":
    unittest.main()
