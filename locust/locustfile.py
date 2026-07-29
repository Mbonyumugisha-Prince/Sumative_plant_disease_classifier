import os
import random

from locust import HttpUser, task, between

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
SAMPLE_IMAGES = [
    os.path.join(SAMPLE_DIR, f)
    for f in os.listdir(SAMPLE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
] if os.path.isdir(SAMPLE_DIR) else []


class PlantDiseaseUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def predict(self):
        if not SAMPLE_IMAGES:
            return
        path = random.choice(SAMPLE_IMAGES)
        with open(path, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": (os.path.basename(path), f, "image/png")},
                name="/predict",
            )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
