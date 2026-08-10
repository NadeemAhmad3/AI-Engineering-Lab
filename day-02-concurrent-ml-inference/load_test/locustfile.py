import random
from locust import HttpUser, task, between

class MLInferenceUser(HttpUser):
    """
    Locust load testing user simulating concurrent ML prediction requests.
    """
    wait_time = between(0.01, 0.05)  # Short pause between consecutive requests

    @task
    def predict(self):
        # Generate 50 random features
        features = [random.uniform(-1.0, 1.0) for _ in range(50)]
        payload = {"features": features}
        
        self.client.post(
            "/predict",
            json=payload,
            name="/predict"
        )
