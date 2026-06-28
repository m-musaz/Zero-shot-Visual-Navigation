from typing import TypedDict, List, Optional


class DatasetEntry(TypedDict):
    distance: float
    scan: str
    path_id: int
    path: List[str]
    heading: float
    instructions: List[str]
    goals: Optional[any]  # You can replace `any` with a specific type if known

