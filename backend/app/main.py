
from fastapi import FastAPI

from backend.app.api.routes import router

app = FastAPI(
    title="E-Shop Support Copilot API",
    description=(
        "API for analyzing customer support tickets, "
        "retrieving relevant policies, and generating "
        "grounded replies."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.get("/health", tags=["Health"])
def health_check():
    """Check whether the API is running."""
    return {"status": "ok"}
