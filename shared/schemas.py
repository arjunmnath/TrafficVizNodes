from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class Attributes(BaseModel):
    color: str
    type: Optional[str] = None  # None for person, string for vehicle

class TrackEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    camera_id: str
    track_id: int
    timestamp: float
    bbox: List[float] = Field(min_length=4, max_length=4)
    class_label: str = Field(alias="class")  # "person" or "vehicle"
    embedding: List[float]
    attributes: Attributes
