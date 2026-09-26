import sys
import os
import subprocess
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS_DIR = os.path.join(ROOT, "models", "artifacts")


@pytest.fixture(scope="module", autouse=True)
def ensure_trained_artifacts():
    """The API loads trained artifacts at import time, so make sure they
    exist before running this module (regenerates the dataset + trains
    the ensemble if artifacts are missing, e.g. on a fresh CI checkout)."""
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "meta_classifier.joblib")):
        subprocess.run([sys.executable, "data/generate_dataset.py"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "models/train_ensemble.py"], cwd=ROOT, check=True)
    yield


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, ROOT)
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_flags_ai_style_text(client):
    text = (
        "In recent developments regarding the bridge collapse, it is important to note "
        "that several factors must be considered. Furthermore, experts suggest that the "
        "outcome may potentially affect residents in the area. In conclusion, this "
        "highlights the importance of continued vigilance."
    )
    r = client.post("/analyze", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["ai_probability"] > 0.5
    assert "signal_breakdown" in body
    assert set(body["signal_breakdown"].keys()) == {
        "perplexity_engine", "stylometric_engine", "learned_classifier"
    }


def test_analyze_flags_human_style_text(client):
    text = (
        "Okay so here's the thing about the bridge collapse: nobody actually saw it "
        "coming. I got there late and honestly it was already chaos on the block. "
        "We'll see what happens next."
    )
    r = client.post("/analyze", json={"text": text})
    assert r.status_code == 200
    assert r.json()["ai_probability"] < 0.5


def test_analyze_rejects_too_short_text(client):
    r = client.post("/analyze", json={"text": "hi"})
    assert r.status_code == 422
