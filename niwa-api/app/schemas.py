# from pydantic import BaseModel, Field
# from typing import List, Optional, Literal

# class Anomaly(BaseModel):
#     box_2d: List[int] = Field(description="[ymin, xmin, ymax, xmax] coordinates of the anomaly")
#     label: str
#     confidence: float
#     reasoning: str

# class ForensicResponse(BaseModel):
#     frame_id: str
#     verdict: Literal["AUTHENTIC", "SUSPICIOUS", "SYNTHETIC_CONFIRMED"]
#     confidence_score: float
#     anomalies: List[Anomaly] = []
#     thought_signature: Optional[str] = None
#     text_response: Optional[str] = None # For conversational fallback
#     audio_data: Optional[str] = None    # For spoken verdict
from pydantic import BaseModel
from typing import List, Optional, Literal

class Anomaly(BaseModel):
    box_2d: List[int] # [ymin, xmin, ymax, xmax] normalized 0-1000
    label: str
    confidence: float
    reasoning: str

class ForensicResponse(BaseModel):
    frame_id: str
    source_model: Literal["LIVE_HUD", "DEEP_AUDIT"]
    verdict: Literal["AUTHENTIC", "SUSPICIOUS", "SYNTHETIC_CONFIRMED"]
    anomalies: List[Anomaly]
    thought_signature: Optional[str] = None
    audit_notes: Optional[str] = None # For Gemini 3 deep thinking output