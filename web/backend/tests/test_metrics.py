from fastapi.testclient import TestClient
from app.main import app


def test_metrics_endpoint_exposes_prometheus_format():
    client = TestClient(app)
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert "agricnxedge_predict_requests_total" in response.text
