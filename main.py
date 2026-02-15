"""
Entrypoint for running the Clinical Safeguard Middleware.

Production:
    uvicorn src.main:app --host 0.0.0.0 --port 8080 --workers 4

Development:
    uvicorn src.main:app --reload
"""
from src.api.app import create_app

app = create_app()
