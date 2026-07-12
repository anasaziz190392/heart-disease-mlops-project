"""
Load testing scenarios for the Heart Disease Prediction API using Locust.
Tests authentication, predictions, and SHAP explanations under load.
"""
from locust import HttpUser, task, between, events
import json
import random
import logging

logger = logging.getLogger(__name__)


class HeartDiseaseAPIUser(HttpUser):
    """Simulates a user interacting with the Heart Disease API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts. Authenticate and get JWT token."""
        response = self.client.post(
            "/token",
            json={"username": "doctor", "password": "doctor123"},
            name="POST /token"
        )
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
            logger.info("User authenticated successfully")
        else:
            logger.error(f"Authentication failed: {response.status_code}")
            self.token = None
    
    @task(4)
    def predict_low_risk(self):
        """Make predictions for low-risk patients (weighted 4x)."""
        patient_data = {
            "age": random.randint(30, 50),
            "sex": random.choice([0, 1]),
            "cp": random.randint(1, 2),
            "trestbps": random.randint(100, 130),
            "chol": random.randint(150, 200),
            "fbs": 0,
            "restecg": random.randint(0, 1),
            "thalach": random.randint(140, 170),
            "exang": 0,
            "oldpeak": random.uniform(0, 1),
            "slope": 2,
            "ca": 0,
            "thal": 3,
        }
        
        response = self.client.post(
            "/predict",
            json=patient_data,
            headers=self.headers,
            name="POST /predict (low-risk)"
        )
        
        if response.status_code != 200:
            logger.warning(f"Prediction failed: {response.status_code}")
    
    @task(2)
    def predict_high_risk(self):
        """Make predictions for high-risk patients (weighted 2x)."""
        patient_data = {
            "age": random.randint(55, 80),
            "sex": random.choice([0, 1]),
            "cp": random.randint(3, 4),
            "trestbps": random.randint(140, 180),
            "chol": random.randint(240, 400),
            "fbs": random.choice([0, 1]),
            "restecg": random.randint(1, 2),
            "thalach": random.randint(90, 130),
            "exang": random.choice([0, 1]),
            "oldpeak": random.uniform(2, 6),
            "slope": random.randint(1, 2),
            "ca": random.randint(1, 3),
            "thal": random.choice([6, 7]),
        }
        
        response = self.client.post(
            "/predict",
            json=patient_data,
            headers=self.headers,
            name="POST /predict (high-risk)"
        )
        
        if response.status_code != 200:
            logger.warning(f"Prediction failed: {response.status_code}")
    
    @task(1)
    def get_explanation(self):
        """Request SHAP explanation for a prediction."""
        params = {
            "age": random.randint(40, 70),
            "sex": random.choice([0, 1]),
            "cp": random.randint(1, 4),
            "trestbps": random.randint(100, 160),
            "chol": random.randint(150, 350),
            "fbs": random.choice([0, 1]),
            "restecg": random.randint(0, 2),
            "thalach": random.randint(100, 180),
            "exang": random.choice([0, 1]),
            "oldpeak": random.uniform(0, 5),
            "slope": random.randint(1, 3),
            "ca": random.randint(0, 3),
            "thal": random.randint(3, 7),
        }
        
        response = self.client.get(
            "/explain",
            params=params,
            headers=self.headers,
            name="GET /explain"
        )
        
        if response.status_code != 200:
            logger.warning(f"Explanation failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Periodic health check."""
        response = self.client.get("/health", name="GET /health")
        if response.status_code != 200:
            logger.warning(f"Health check failed: {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("Load test started")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("Load test finished")
    logger.info(f"Total requests: {environment.stats.total.num_requests}")
    logger.info(f"Total failures: {environment.stats.total.num_failures}")
