const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").trim();
// Empty string => same-origin. Works for local dev via localhost:8000,
// single-container Docker/Cloud Run (FastAPI serves / + /api), and split hosting via env.

export async function getHealth() {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new Error("API health check failed.");
  return response.json();
}

export async function predictImage(file) {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body,
  });
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "Prediction request failed.");
  }

  return data;
}
