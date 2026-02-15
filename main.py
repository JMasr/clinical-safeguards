"""
Entrypoint for running the Clinical Safeguard Middleware.

Production:
    uvicorn clinical_safeguard.main:app --host 0.0.0.0 --port 8080 --workers 4

Development:
    uvicorn clinical_safeguard.main:app --reload
"""
from clinical_safeguard.api.app import create_app

app = create_app()
