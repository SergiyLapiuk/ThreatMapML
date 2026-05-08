from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import threat

app = FastAPI(title="ThreatMap LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(threat.router, prefix="/api/threat")


@app.get("/")
async def root():
    return {"message": "ThreatMap LLM API працює!"}