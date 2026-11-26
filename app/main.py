from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.services.analysis_service import router as analysis_router

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(
    title="Link Analyser",
    docs_url="/docs",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The only API endpoint.
app.include_router(analysis_router)


@app.get("/", include_in_schema=False)
def frontend_page():
    return FileResponse(FRONTEND_DIR / "main.html")


# Serves style.css and main.js referenced by the frontend page.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=4000, reload=True)
