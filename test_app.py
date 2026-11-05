import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Study_Hours": 6,
                "Attendance": 90,
                "Assignment_Score": 88,
                "Internal_Marks": 85,
                "Previous_Score": 80
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "PASS"
        )

    def test_low_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Study_Hours": 2,
                "Attendance": 55,
                "Assignment_Score": 40,
                "Internal_Marks": 30,
                "Previous_Score": 35
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "FAIL"
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
