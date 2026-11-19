import unittest
from core import calculate_score, validate_http_url

class ScoreTests(unittest.TestCase):
    def test_all_full(self):
        result = calculate_score(["충분히 확인됨"] * 5)
        self.assertEqual(result["score"], 5.0)
        self.assertEqual(result["status"], "평가 완료")

    def test_partial_and_zero(self):
        result = calculate_score(["충분히 확인됨", "일부 확인됨", "확인되지 않거나 기준 미충족", "평가 불가", "평가 불가"])
        self.assertEqual(result["score"], 2.5)
        self.assertEqual(result["evaluated"], 3)

    def test_too_few_evaluable(self):
        result = calculate_score(["평가 불가", "충분히 확인됨", "평가 불가", "평가 불가", "충분히 확인됨"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "평가 보류")

    def test_localhost_rejected(self):
        ok, _ = validate_http_url("http://localhost:8501")
        self.assertFalse(ok)

    def test_https_accepted(self):
        ok, _ = validate_http_url("https://example.com/article")
        self.assertTrue(ok)

if __name__ == "__main__":
    unittest.main()
