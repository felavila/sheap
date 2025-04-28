from dataclasses import dataclass
from typing import Optional

@dataclass
class SpectralLine:
    center: float
    line_name: str
    kind: str
    component: int
    amplitude: float = 0.0            # default amplitude
    how: Optional[str] = None         # None if missing
    region: Optional[str] = None      # None if missing
    profile: Optional[str] = None     # None if missing
    how : Optional[str] = None 
    