"""This module contains the utilities use for ComplexBuilder."""

__author__ = 'felavila'


__all__ = [
    "default_known_tied_relations",
    "fe_ties",
    "flatten_index_ties",
    "group_lines",
]

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sheap.Core import SpectralLine,SpectralLineList

#TODO check how the ties change now we have logamp and not amplitude. e.g reparametrize 

def fe_ties(entries: SpectralLineList, routine_fe_tied) -> List[List[str]]:
    """
    Generate tied parameter expressions for Fe emission lines.

    Parameters
    ----------
    entries : list of SpectralLine
        Spectral line entries to be checked for Fe region membership.
    routine_fe_tied : dict
        Dictionary with tie rules, e.g. {'by': 'element', 'tied_params': ('center', 'fwhm')}.

    Returns
    -------
    list of list of str
        Each inner list is a pair of tied parameter names [dependent, reference].
    """
    tied_params = routine_fe_tied.get("tied_params", ('center', 'fwhm'))
    by = routine_fe_tied.get("by", "element")
    #fe_tied_params = {"by":"subregion","tied_params": ('center', 'fwhm')}
    subregion, centers, element,_ = np.array([[e.subregion, e.center, e.element,e.region] for e in entries]).T
    mask_fe = np.char.find(subregion.astype(str), "fe") >= 0
    subregion, centers, element, entries = (
        subregion[mask_fe],
        centers[mask_fe],
        element[mask_fe],
        [entries[i] for i in np.where(mask_fe)[0]],)

    ties: List[List[str]] = []

    if by == "element":
        for reg in np.unique(element):
            idx_region = np.where(element == reg)[0]
            entries_region = [entries[i] for i in idx_region]
            centers_region = np.array([e.center for e in entries_region])
            idx_center = int(np.argmin(np.abs(centers_region - np.median(centers_region))))
            for i, e in enumerate(entries_region):
                if i == idx_center or 'fe' not in e.region:
                    continue
                for p in tied_params:  #
                    ties.append(
                        [
                            f"{p}_{e.line_name}_{e.component}_{e.region}",
                            f"{p}_{entries_region[idx_center].line_name}_{entries_region[idx_center].component}_{entries_region[idx_center].region}",
                        ]
                    )
    elif by == "subregion":
        for reg in np.unique(subregion):
            idx_region = np.where(subregion == reg)[0]
            entries_region = [entries[i] for i in idx_region]
            centers_region = np.array([e.center for e in entries_region])
            idx_center = int(np.argmin(np.abs(centers_region - np.median(centers_region))))
            for i, e in enumerate(entries_region):
                if i == idx_center or 'fe' not in e.region:
                    continue
                for p in tied_params:  #
                    ties.append(
                        [
                            f"{p}_{e.line_name}_{e.component}_{e.region}",
                            f"{p}_{entries_region[idx_center].line_name}_{entries_region[idx_center].component}_{entries_region[idx_center].region}",
                        ]
                    )
    else:
        centers = np.array([e.center for e in entries])
        idx_center = int(np.argmin(np.abs(centers - np.median(centers))))
        for i, e in enumerate(entries):
            if i == idx_center or 'fe' not in e.region:
                continue
            for p in tied_params:
                ties.append(
                    [
                        f"{p}_{e.line_name}_{e.component}_{e.region}",
                        f"{p}_{entries[idx_center].line_name}_{entries[idx_center].component}_{entries[idx_center].region}",
                    ]
                )

    return ties

def _maketies(
    complex_class,
    tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, Any]]]] = None,
    tied_broad_to: Optional[Union[str, Dict[int, Dict[str, Any]]]] = None,
    known_tied_relations: Optional[List[Tuple[Tuple[str, ...], List[str]]]] = None,
    only_known: bool = False,
) -> List[List[str]]:
    """
    Generate parameter ties for narrow and broad components in a spectral region.

    Parameters
    ----------
    complex_class : ComplexRegion
        Region containing spectral line definitions.
    tied_narrow_to : str or dict, optional
        Line name or component mapping to tie narrow lines to.
    tied_broad_to : str or dict, optional
        Line name or component mapping to tie broad lines to.
    known_tied_relations : list, optional
        Predefined tie relations to enforce.
    only_known : bool, optional
        If True, return only known tied relations (ignore mainline ties).

    Returns
    -------
    list of list of str
        List of parameter name ties [dependent, reference].
    """
    # Determine mainline
    mainline_candidates_broad = ["Halpha","Hbeta","MgII","CIVb","Lyalpha","Pad",]  # this can be disscuss in the future
    mainline_candidates_narrow = ["OIIIc","Halpha","NIIb","MgII","CIII]","SIIb","OIIa",]  # this can be disscuss in the future
    
    n_components_per_region = complex_class.characteristics()["n_components_per_region"]
    n_broad = n_components_per_region["broad"]
    n_narrow = n_components_per_region["narrow"]
    dict_region = complex_class.group_by("region")
    local_region_list = dict_region["broad"].lines + dict_region["narrow"].lines
    
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
            mainline_candidates_narrow[0] if mainline_candidates_narrow else '',)

    ties: List[List[str]] = []


    for name, mapping in (
        ('tied_narrow_to', tied_narrow_to),
        ('tied_broad_to', tied_broad_to),
    ):
        if mapping and not isinstance(mapping, (str, dict)):
            raise TypeError(f"{name} must be str or dict, got {type(mapping).__name__}")

    tied_narrow_to = tied_narrow_to or mainline_narrow
    tied_broad_to = tied_broad_to or mainline_broad

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
    broad_map = _to_map(tied_broad_to,n_broad)

    def add_tie_if_different(source, target):
        if source != target:
            ties.append([source, target])

    for e in local_region_list:
        comp = e.component
        if e.region == "narrow":
            target = narrow_map[comp]
            suffix = "narrow"
        elif e.region == "broad":
            target = broad_map[comp]
            suffix = "broad"
        else:
            continue  # unknown region

        for p in ("center", "fwhm"):
            source_name = f"{p}_{e.line_name}_{comp}_{suffix}"
            target_name = f"{p}_{target['line_name']}_{target['component']}_{suffix}"
            add_tie_if_different(source_name, target_name)

    if known_tied_relations:
        _known_ties = []
        present = {e.line_name for e in local_region_list if isinstance(e.line_name, str)}
        for pair, factor in known_tied_relations:
            if all(name in present for name in pair):
                for k in range(1, n_narrow + 1):
                    #_ties.append([f.replace("component", str(k)) for f in factor])
                    _known_ties.append([f.replace("component", str(k)) for f in factor])
    #     if only_known:
    #         return local_ties
    # ties_ = []
    # for t in ties:
    #     if t not in ties_:
    #         ties_.append(t)

    return ties,_known_ties




def flatten_index_ties(index_ties: List[Tuple[int, int, str, float]]) -> Dict[int, Tuple[float, int]]:
    """
    Resolve index-based tie dependencies into coefficient + free parameter mapping.

    Parameters
    ----------
    index_ties : list of tuple
        Each entry is (target_idx, source_idx, operation, value).

    Returns
    -------
    dict
        Mapping from target index to (coefficient, source index).
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

#?
def group_lines(
    lines: SpectralLineList,
    region: str,
    profile: str = "gaussian",
    mode: str = "region",
    known_tied_relations: List[Tuple[Tuple[str, ...], List[str]]] = [],
) -> SpectralLineList:
    """
    Collapse multiple spectral lines into grouped pseudo-lines with SPAF profiles.

    Parameters
    ----------
    lines : list of SpectralLine
        Full list of lines in the region.
    region : str
        Region name to group (e.g., "fe", "narrow").
    profile : str, default "gaussian"
        Base profile to assign to grouped line.
    mode : {"region", "subregion", "element"}
        Grouping logic applied.
    known_tied_relations : list, optional
        Amplitude tie relations to apply.

    Returns
    -------
    list of SpectralLine
        Lines with some collapsed into SPAF composite profiles.
    """
    grouped = defaultdict(list)
    collapsed_lines = []
    lines_to_remove = set()

    for line in lines:
        if line.region != region:
            continue

        if mode == "region":
            key_base = line.region
        elif mode == "subregion":
            key_base = line.subregion
        elif mode == "element":
            key_base = line.element
        else:
            continue

        key = (key_base, line.component)
        grouped[key].append(line)

    for (region_key, comp), group in grouped.items():
        name_to_idx = {line.line_name: i for i, line in enumerate(group)}
        name_to_comp = {line.line_name: line.component for line in group}
        index_ties = []

        for pair, factor in known_tied_relations:
            if all(name in name_to_idx for name in pair):
                if len(factor) < 3:
                    factor += ["*1"]
                target_str, source_str, op_val = factor
                if "amplitude" in target_str:
                    target_name = target_str.split("_")[1]
                    source_name = source_str.split("_")[1]
                    if name_to_comp[target_name] != name_to_comp[source_name]:
                        continue
                    target_idx = name_to_idx[target_name]
                    source_idx = name_to_idx[source_name]
                    op, val = op_val[0], float(op_val[1:])
                    index_ties.append((target_idx, source_idx, op, val))

        resolved_map = flatten_index_ties(index_ties)
        full_rules: List[Tuple[int, float, int]] = []
        
        dependent_list = []
        for dx in range(len(group)):
            if dx in resolved_map:
                coef, idx = resolved_map[dx]
                if idx != dx:
                    dependent_list.append(int(dx))
                full_rules.append((dx, float(coef) , idx))
            else:
                full_rules.append((dx, 1.0, dx))
        #print("dependent_list",dependent_list)
        amplitudes = np.array([line.amplitude for i, line in enumerate(group) if i not in dependent_list])
        
        arg_max = np.argmax(amplitudes)

        if region == "fe" and mode == "element":
            full_rules = []
            dependent_list = []
            for n, coef in enumerate(amplitudes):
                full_rules.append((n, float(coef/ np.max(amplitudes)), int(arg_max)))
                if n != arg_max:
                    dependent_list.append(n)
            amplitudes = np.array([
                float(line.amplitude) for i, line in enumerate(group) if i not in dependent_list])

        centers = [line.center for line in group]
        region_lines = [line.line_name for line in group]
        elements = [line.element for line in group]
        subregions = [line.subregion for line in group]
        line_name = f"{region_key}{comp}" if mode=="region" else f"{region}{region_key}{comp}"
        #print(line_name)
        collapsed = SpectralLine(
            center=centers,
            line_name=line_name,
            region=region,
            component=comp,
            profile=profile,
            region_lines=region_lines,
            element=elements,
            subregion = subregions,
            amplitude=amplitudes.tolist(),
            amplitude_relations=full_rules,
        )

        collapsed_lines.append(collapsed)
        lines_to_remove.update(id(line) for line in group)

    new_lines = [line for line in lines if id(line) not in lines_to_remove]
    new_lines.extend(collapsed_lines)

    return new_lines


#TODO check how this affect the combination e.g what happends when hbeta depends on halpha and hg and hd depends on hbeta
def default_known_tied_relations(
    include_balmer: bool = True,
    include_forbidden: bool = True,
) -> List[Tuple[Tuple[str, ...], List[str]]]:
    ties: List[Tuple[Tuple[str, ...], List[str]]] = []
    """
    Return default tied relations for common line doublets and multiplets.

    Parameters
    ----------
    include_balmer : bool
        If True, include Balmer series constraints.
    include_forbidden : bool
        If True, include [O III], [N II], [S III], [O I], and [O II] constraints.

    Returns
    -------
    list of tuple
        Each tuple contains (line_name pair, constraint expression list).
    """

    ties += [
        (("OIIIb", "OIIIc"), ["amplitude_OIIIb_component_narrow", "amplitude_OIIIc_component_narrow", "*0.33"]),
        (("OIIIb", "OIIIc"), ["center_OIIIb_component_narrow", "center_OIIIc_component_narrow"]),
        (("NIIa", "NIIb"), ["amplitude_NIIa_component_narrow", "amplitude_NIIb_component_narrow", "*0.33"]),
        (("NIIa", "NIIb"), ["center_NIIa_component_narrow", "center_NIIb_component_narrow"]),
    ]

    if include_forbidden:
        ties += [
            (("OIa", "OIb"), ["amplitude_OIa_component_narrow", "amplitude_OIb_component_narrow", "*3.0"]),
            (("OIa", "OIb"), ["center_OIa_component_narrow", "center_OIb_component_narrow"]),
            (("SIIIa", "SIIIb"), ["amplitude_SIIIa_component_narrow", "amplitude_SIIIb_component_narrow", "*0.4"]),
            (("OIIa", "OIIb"), ["amplitude_OIIa_component_narrow", "amplitude_OIIb_component_narrow", "*0.77"]),
            (("OIIa", "OIIb"), ["center_OIIa_component_narrow", "center_OIIb_component_narrow"]),
        ]

    if include_balmer:
        ties += [
            (("Hbeta", "Halpha"), ["amplitude_Hbeta_component_narrow", "amplitude_Halpha_component_narrow", "*0.35"]),
            (("Hg", "Hbeta"), ["amplitude_Hgamma_component_narrow", "amplitude_Hbeta_component_narrow", "*0.47"]),
            (("Hd", "Hbeta"), ["amplitude_Hdelta_component_narrow", "amplitude_Hbeta_component_narrow", "*0.26"]),
        ]
    
    # if include_helium:
    #     ties += [
    #         (("HeId", "HeI5876"), ["amplitude_HeI4471_component_narrow", "amplitude_HeI5876_component_narrow", "*0.3"]),
    #         (("HeI6678", "HeI5876"), ["amplitude_HeI6678_component_narrow", "amplitude_HeI5876_component_narrow", "*0.2"]),
    #     ]
    
    return ties

