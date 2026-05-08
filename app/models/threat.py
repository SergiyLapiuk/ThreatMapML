from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from pydantic import Field

class Threat(BaseModel):

    ThreatType: str
    CurrentLocation: str
    Direction: Optional[str]
    Count: int = 1

    StartLatitude: Optional[float]
    StartLongitude: Optional[float]

    EndLatitude: Optional[float]
    EndLongitude: Optional[float]

    DetectedAt: datetime = Field(default_factory=datetime.utcnow)