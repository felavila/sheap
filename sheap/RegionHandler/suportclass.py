from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class SpectralLine:
    center: float
    line_name: str
    kind: str
    component: int
    amplitude: float = 1.0            # default amplitude
    how: Optional[str] = None         # None if missing
    region: Optional[str] = None      # None if missing
    profile: Optional[str] = None     # None if missing
    how : Optional[str] = None 
    which : Optional[str] = None 
    
def fe_ties(entries: List[SpectralLine], by_region=True,tied_params=('center', 'width')) -> List[List[str]]:
    regions, centers, kinds = np.array([[e.region, e.center, e.kind] for e in entries]).T
    mask_fe = np.char.find(kinds, "fe") >= 0
    regions, centers, kinds, entries = (
        regions[mask_fe],
        centers[mask_fe],
        kinds[mask_fe],
        np.array(entries)[np.where(mask_fe)[0]]
    )
    
    ties: List[List[str]] = []
    
    if by_region:
        for reg in np.unique(regions):
            idx_region = np.where(regions == reg)[0]
            entries_region = entries[idx_region]
            centers_region = np.array([e.center for e in entries_region])
            idx_center = int(np.argmin(np.abs(centers_region - np.median(centers_region))))
            for i, e in enumerate(entries_region):
                if i == idx_center or 'fe' not in e.kind:
                    continue
                for p in ('center', 'width'): #
                    ties.append([
                        f"{p}_{e.line_name}_{e.component}_{e.kind}",
                        f"{p}_{entries_region[idx_center].line_name}_{entries_region[idx_center].component}_{entries_region[idx_center].kind}"
                    ])
    else:
        centers = np.array([e.center for e in entries])
        idx_center = int(np.argmin(np.abs(centers - np.median(centers))))
        for i, e in enumerate(entries):
            if i == idx_center or 'fe' not in e.kind:
                continue
            for p in ('center', 'width'):
                ties.append([
                    f"{p}_{e.line_name}_{e.component}_{e.kind}",
                    f"{p}_{entries[idx_center].line_name}_{entries[idx_center].component}_{entries[idx_center].kind}"
                ])
    
    return ties


def region_ties(
    local_region_list: List[SpectralLine],
    mainline_candidates: Union[str, List[str]],
    n_narrow: int,
    n_broad: int,
    tied_narrow_to: Union[str, Dict[int, Dict[str, Any]]] = None,
    tied_broad_to: Union[str, Dict[int, Dict[str, Any]]] = None,
    known_tied_relations: List[Tuple[Tuple[str, ...], List[str]]] = None,
    only_known: bool =False
) -> List[List[str]]:
    """
    Generate ties between narrow and broad components in a local region.

    - local_region_list: list of SpectralLine objects to tie
    - mainline_candidates: single line_name or list to select mainline from
    - n_narrow: expected narrow component count
    - n_broad: expected broad component count
    - tied_narrow_to: mapping or base name for narrow ties
    - tied_broad_to: mapping or base name for broad ties
    - known_tied_relations: optional predefined relations

    Returns a list of [param1, param2] tie declarations.
    """
    # Determine mainline
    if isinstance(mainline_candidates, (list, tuple)):
        available = {e.line_name for e in local_region_list}
        mainline = next((name for name in mainline_candidates if name in available),
                        mainline_candidates[0] if mainline_candidates else '')
    else:
        mainline = mainline_candidates

    ties: List[List[str]] = []

    # Validate optional mappings
    for name, mapping in (('tied_narrow_to', tied_narrow_to), ('tied_broad_to', tied_broad_to)):
        if mapping and not isinstance(mapping, (str, dict)):
            raise TypeError(f"{name} must be str or dict, got {type(mapping).__name__}")

    
    tied_narrow_to = tied_narrow_to or mainline
    tied_broad_to = tied_broad_to or mainline
    
    # Helper to build mapping dict
    def _to_map(target, count):
        if isinstance(target, str):
            return {k: {"line_name": target, "component": k}
                    for k in range(1, count + 1)}
        return {k: {"line_name": target.get(k, {}).get("line_name", mainline),
                    "component": target.get(k, {}).get("component", k)}
                for k in range(1, count + 1)}

    narrow_map = _to_map(tied_narrow_to, n_narrow)
    broad_map  = _to_map(tied_broad_to, n_broad)
   

    def add_tie_if_different(source, target):
        if source != target:
            ties.append([source, target])

    for e in local_region_list:
        comp = e.component
        if e.kind == "narrow":
            target = narrow_map[comp]
            suffix = "narrow"
        elif e.kind == "broad":
            target = broad_map[comp]
            suffix = "broad"
        else:
            continue  # unknown kind

        for p in ("center", "width"):
            source_name = f"{p}_{e.line_name}_{comp}_{suffix}"
            target_name = f"{p}_{target['line_name']}_{target['component']}_{suffix}"
            add_tie_if_different(source_name, target_name)
    
    if known_tied_relations:
        local_ties=[]
        present = {e.line_name for e in local_region_list}
        for pair, factor in known_tied_relations:
            if all(name in present for name in pair):
                for k in range(1, n_narrow + 1):
                    ties.append([f.replace("component", str(k)) for f in factor])
                    local_ties.append([f.replace("component", str(k)) for f in factor])
        if only_known:
            return local_ties
    ties_ = []
    for t in ties:
        if t not in ties_:
            ties_.append(t)
    
    return ties_
#maybe a dataclass of fitting rutine? 