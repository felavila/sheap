from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sheap.DataClass.DataClass import SpectralLine


def group_lines_by_region(
    lines: List[SpectralLine],
    kind: str = "fe",
    component: int = 20,
    profile: str = "gaussian",
    exception: List[str] = []
) -> List[SpectralLine]:
    grouped = defaultdict(list)

    # Step 1: Filter and group by kind and region, skipping exceptions
    for line in lines:
        if line.kind == kind and line.region is not None and line.region not in exception:
            grouped[line.region].append(line)

    # Step 2: Collapse groups into single SpectralLine instances
    collapsed_lines = []
    for region, group in grouped.items():
        centers = [line.center for line in group]
        line_names = [f"{line.region}" for line in group][0]
        region_lines = [line.line_name for line in group]
        amplitudes = [line.amplitude for line in group]

        base_line = group[0]
        collapsed_lines.append(
            SpectralLine(
                center=centers,  # type: ignore
                line_name=line_names,  # type: ignore
                kind=kind,
                component=component,
                amplitude=amplitudes,  # type: ignore
                how=base_line.how,
                region=region,
                profile=profile,
                which=base_line.which,
                region_lines=region_lines
            )
        )

    # Step 3: Keep all lines not grouped or excluded by exception
    new_lines = [
        line for line in lines
        if line.kind != kind or line.region in exception
    ]
    new_lines.extend(collapsed_lines)

    return new_lines



def fe_ties(
    entries: List[SpectralLine], by_region=True, tied_params=('center', 'fwhm')
) -> List[List[str]]:
    regions, centers, kinds,_ = np.array([[e.region, e.center, e.kind,e.how] for e in entries if e.how !="combine"]).T
    
    mask_fe = np.char.find(kinds.astype(str), "fe") >= 0
    regions, centers, kinds, entries = (
        regions[mask_fe],
        centers[mask_fe],
        kinds[mask_fe],
        [entries[i] for i in np.where(mask_fe)[0]],
    )

    ties: List[List[str]] = []

    if by_region:
        for reg in np.unique(regions):
            idx_region = np.where(regions == reg)[0]
            entries_region = [entries[i] for i in idx_region]
            centers_region = np.array([e.center for e in entries_region])
            idx_center = int(np.argmin(np.abs(centers_region - np.median(centers_region))))
            for i, e in enumerate(entries_region):
                if i == idx_center or 'fe' not in e.kind:
                    continue
                for p in tied_params:  #
                    ties.append(
                        [
                            f"{p}_{e.line_name}_{e.component}_{e.kind}",
                            f"{p}_{entries_region[idx_center].line_name}_{entries_region[idx_center].component}_{entries_region[idx_center].kind}",
                        ]
                    )
    else:
        centers = np.array([e.center for e in entries])
        idx_center = int(np.argmin(np.abs(centers - np.median(centers))))
        for i, e in enumerate(entries):
            if i == idx_center or 'fe' not in e.kind:
                continue
            for p in tied_params:
                ties.append(
                    [
                        f"{p}_{e.line_name}_{e.component}_{e.kind}",
                        f"{p}_{entries[idx_center].line_name}_{entries[idx_center].component}_{entries[idx_center].kind}",
                    ]
                )

    return ties


def region_ties(
    local_region_list: List[SpectralLine],
    n_narrow: int,
    n_broad: int,
    tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, Any]]]] = None,
    tied_broad_to: Optional[Union[str, Dict[int, Dict[str, Any]]]] = None,
    known_tied_relations: Optional[List[Tuple[Tuple[str, ...], List[str]]]] = None,
    only_known: bool = False,
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
    mainline_candidates_broad = [
        "Halpha",
        "Hbeta",
        "MgII",
        "CIVb",
        "Lyalpha",
        "Pad",
    ]  # this can be disscuss in the future
    mainline_candidates_narrow = [
        "OIIIc",
        "Halpha",
        "NIIb", #could mm
        "MgII",
        "CIII]",
        "SIIb",
        "OIIa",
    ]  # this can be disscuss in the future

    if isinstance(mainline_candidates_broad, (list, tuple)):
        available = {e.line_name for e in local_region_list if isinstance(e.line_name, str)}
        mainline_broad = next(
            (name for name in mainline_candidates_broad if name in available),
            mainline_candidates_broad[0] if mainline_candidates_broad else '',
        )
    if isinstance(mainline_candidates_narrow, (list, tuple)):
        available = {e.line_name for e in local_region_list if isinstance(e.line_name, str)}
        mainline_narrow = next(
            (name for name in mainline_candidates_narrow if name in available),
            mainline_candidates_narrow[0] if mainline_candidates_narrow else '',
        )

    # print(mainline_broad,mainline_narrow)
    ties: List[List[str]] = []

    # Validate optional mappings
    for name, mapping in (
        ('tied_narrow_to', tied_narrow_to),
        ('tied_broad_to', tied_broad_to),
    ):
        if mapping and not isinstance(mapping, (str, dict)):
            raise TypeError(f"{name} must be str or dict, got {type(mapping).__name__}")

    tied_narrow_to = tied_narrow_to or mainline_narrow
    tied_broad_to = tied_broad_to or mainline_broad

    # Helper to build mapping dict
    def _to_map(target, count):
        if isinstance(target, str):
            return {k: {"line_name": target, "component": k} for k in range(1, count + 1)}
        return {
            k: {
                "line_name": target.get(k, {}).get("line_name", tied_broad_to),
                "component": target.get(k, {}).get("component", k),
            }
            for k in range(1, count + 1)
        }

    narrow_map = _to_map(tied_narrow_to, n_narrow)
    broad_map = _to_map(tied_broad_to, n_broad)

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

        for p in ("center", "fwhm"):
            source_name = f"{p}_{e.line_name}_{comp}_{suffix}"
            target_name = f"{p}_{target['line_name']}_{target['component']}_{suffix}"
            add_tie_if_different(source_name, target_name)

    if known_tied_relations:
        local_ties = []
        present = {e.line_name for e in local_region_list if isinstance(e.line_name, str)}
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


# maybe a dataclass of fitting rutine?


def flatten_index_ties(index_ties: List[Tuple[int, int, str, float]]) -> Dict[int, Tuple[float, int]]:
    """
    Flatten ties into mapping: target_index → (coefficient, free_param_index)
    """
    resolved: Dict[int, Tuple[float, int]] = {}
    free_indices = set()

    sources = {src for _, src, _, _ in index_ties}
    targets = {tgt for tgt, _, _, _ in index_ties}
    free_indices.update(sources - targets)

    for i in free_indices:
        resolved[i] = (1.0, i)

    for tgt, src, op, val in index_ties:
        if src not in resolved:
            resolved[src] = (1.0, src)
        coef_src, free_idx = resolved[src]

        if op == '*':
            coef = coef_src * val
        elif op == '/':
            coef = coef_src / val
        else:
            raise ValueError(f"Unsupported operation: {op}")

        resolved[tgt] = (coef, free_idx)

    return resolved


def group_lines(
    lines: List[SpectralLine],
    kind: str = "fe",
    profile: str = "gaussian",
    exception_region: List[str] = [],
    include_region: List[str] = [],
    mode: str = "region",
    known_tied_relations: List[Tuple[Tuple[str, ...], List[str]]] = [],
    exception = []
) -> Tuple[List[SpectralLine], List[Tuple[int, float, int]]]:
    #main function builded to combine the functions 
    grouped = defaultdict(list)
    collapsed_lines = []

    # Group lines by (region/kind, component)
    for line in lines:
        if line.kind == kind and line.region is not None and line.region not in exception_region: #sheapectral.complex_region
            key_base = line.region if mode == "region" else line.kind
            key = (key_base, line.component)  # enforce component consistency
            grouped[key].append(line)

    for (region, comp), group in grouped.items():
        #present_names = [line.line_name for line in group]
        name_to_idx = {line.line_name: i for i, line in enumerate(group)}
        name_to_comp = {line.line_name: line.component for line in group}

        index_ties = []

        # Parse known ties and apply only if components match
        for pair, factor in known_tied_relations:
            if all(name in name_to_idx for name in pair):
                if len(factor) < 3:
                    factor += ["*1"]
                target_str, source_str, op_val = factor
                if "amplitude" in target_str:
                    target_name = target_str.split("_")[1]
                    source_name = source_str.split("_")[1]

                    if name_to_comp[target_name] != name_to_comp[source_name]:
                        continue  # skip if components don't match

                    target_idx = name_to_idx[target_name]
                    source_idx = name_to_idx[source_name]
                    op, val = op_val[0], float(op_val[1:])
                    index_ties.append((target_idx, source_idx, op, val))

        resolved_map = flatten_index_ties(index_ties)

        full_rules: List[Tuple[int, float, int]] = []
        dependent_list = []
        for i in range(len(group)):
            if i in resolved_map:
                coef, idx = resolved_map[i]
                dependent_list.append(i)
                full_rules.append((i, coef, idx))
            else:
                full_rules.append((i, 1.0, i))  # Free amplitude

        centers = [line.center for line in group]
        amplitudes = [line.amplitude for i, line in enumerate(group) if i not in dependent_list]
        region_lines = [line.line_name for line in group]
        base_line = group[0]
        if region=="feii_coronal" and kind=="fe":
            profile="sum_gaussian_amplitude_free"
        elif kind=="fe":
            profile="gaussian"
        collapsed_lines.append(
            SpectralLine(
                center=centers,
                line_name=region+str(comp), #maybe i should think about this a little more 
                kind=kind,
                component=comp,
                amplitude=amplitudes,
                how=base_line.how,
                region=region,
                profile=profile,
                which=base_line.which,
                region_lines=region_lines,
                amplitude_relations=full_rules #not necesary 
            )
        )

    new_lines = [
        line for line in lines
        if line.kind != kind or line.region in exception_region
    ]
    new_lines.extend(collapsed_lines)

    return new_lines
