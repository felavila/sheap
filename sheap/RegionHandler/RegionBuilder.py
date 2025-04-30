# Balmer continuum, Balmer High order emission lines
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from sheap.RegionHandler.suportclass import SpectralLine, fe_ties, region_ties

#yaml_files = 
# Named constants for special components
OUTFLOW_COMPONENT = 10
FE_COMPONENT = 20
POWER_LAW_RANGE_THRESHOLD = 2000

    
class RegionBuilder:
    """
    Builds spectral fitting regions from YAML templates, with narrow, broad,
    outflow, and FeII components, plus parameter tying.
    """
    known_tied_relations: List[Tuple[Tuple[str, ...], List[str]]] = [
        (('OIIIb', 'OIIIc'), ['amplitude_OIIIb_component_narrow', 'amplitude_OIIIc_component_narrow', '*0.3']),
        (('NIIa', 'NIIb'), ['amplitude_NIIa_component_narrow', 'amplitude_NIIb_component_narrow', '*0.3']),
        (('NIIa', 'NIIb'), ['center_NIIa_component_narrow', 'center_NIIb_component_narrow']),
        (('OIIIb', 'OIIIc'), ['center_OIIIb_component_narrow', 'center_OIIIc_component_narrow']),
    ]

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n_narrow: int = 1,
        n_broad: int = 1,
        yaml_paths: Optional[List[Union[str, Path]]] = list(Path(__file__).resolve().parent.glob("regions_as_fantasy/*.yaml")),
        tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        mainline_candidates = ["Hbeta","Halpha"],
        fe_regions = ['Fe_uv',"FeII_IZw1","feII_forbidden","FeII_coronal"],
        template_mode_fe:bool = False,
        add_outflow:bool = False,
        add_narrowplus:bool = False,
        by_region:bool = False
        
    ) -> None:
        self.xmin: float = xmin
        self.xmax: float = xmax
        self.tied_narrow_to = tied_narrow_to
        self.tied_broad_to = tied_broad_to
        self.n_narrow: int = n_narrow
        self.n_broad: int = n_broad
        self.lines_regions_available: Dict[str, Any] = {}
        self.regions_available: List[str] = []
        self.regions_to_fit: List[SpectralLine] = []
        self._load_region_templates(yaml_paths)
        self.tied_params: List[List[str]] = []
        self.template_mode_fe = template_mode_fe
        self.mainline_candidates = mainline_candidates
        self.add_outflow = add_outflow
        self.add_narrowplus = add_narrowplus
        self.fe_regions = fe_regions
        self.by_region = by_region
        self.make_region()

    def _load_region_templates(self, paths: Optional[List[Union[str, Path]]]) -> None:
        """
        Load YAML files defining spectral regions.
        """
        if not paths:
            raise ValueError("No YAML paths provided for region templates.")

        for p in paths:
            path = Path(p)
            if not path.is_file():
                raise FileNotFoundError(f"Region YAML not found: {path}")
            data = yaml.safe_load(path.read_text())
            key = path.stem
            if 'region' not in data or not isinstance(data['region'], list):
                raise KeyError(f"Missing 'region' list in YAML: {path}")
            self.lines_regions_available[key] = data
        self.regions_available = list(self.lines_regions_available.keys())

    def make_region(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        n_broad: Optional[int] = None,
        n_narrow: Optional[int] = None,
        main_regions: List[str] = ['hydrogen', 'helium'],
        fe_regions: Optional[List[str]] = None,
        add_outflow: Optional[bool] = None,
        add_narrowplus: Optional[bool] = None,
        template_mode_fe: Optional[bool] = None,
        tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        force_linear: Optional[bool] = None,
        mainline_candidates = None,
        by_region: Optional[bool] = None
    ) -> None:
        # Override defaults
        xmin = xmin if xmin is not None else self.xmin
        xmax = xmax if xmax is not None else self.xmax
        n_broad = n_broad if n_broad is not None else self.n_broad
        n_narrow = n_narrow if n_narrow is not None else self.n_narrow
        tied_narrow_to = tied_narrow_to if tied_narrow_to is not None else self.tied_narrow_to
        tied_broad_to =  tied_broad_to if tied_broad_to is not None else self.tied_broad_to
        mainline_candidates =  mainline_candidates if mainline_candidates is not None else self.mainline_candidates
        template_mode_fe =  template_mode_fe if template_mode_fe is not None else self.template_mode_fe
        add_outflow = add_outflow if add_outflow is not None else self.add_outflow
        add_narrowplus = add_narrowplus if add_narrowplus is not None else self.add_narrowplus
        fe_regions = fe_regions if fe_regions is not None else self.fe_regions
        by_region = by_region if by_region is not None else self.by_region
            #template = {"line_name":"feop","kind": "fe","component":20,"how":"template","which":"OP"}
        
        self.regions_to_fit.clear()
        self.tied_params.clear()
        #self.tied_params_step_2.clear()

        narrow_keys = ['narrow_basic'] + (['narrow_plus'] if add_narrowplus else [])
        if template_mode_fe and (xmax - xmin) > 1000:
            if xmin>=4400 and xmax<=6000:
               self.regions_to_fit.extend([SpectralLine(
                    center=0,
                    line_name="feop",
                    kind="fe",  # fallback to empty
                    component= FE_COMPONENT+1,
                    amplitude=0,
                    profile="fitFeOP",
                    how="template",
                    which="OP",
                    region = "OP"
                )])
            else:
                print("the covered range is not accepted to use template moving to sum of lines mode n/ work in progress")
                template_mode_Fe = False
                
        for name, region in self.lines_regions_available.items():
            for entry in region['region']:
                center = float(entry.get('center', -np.inf))
                if not (xmin <= center <= xmax):
                    continue
                base = SpectralLine(
                    center=center,
                    line_name=str(entry['line_name']),
                    kind=str(entry.get('kind', '')),  # fallback to empty
                    component=int(entry.get('component', 1)),
                    amplitude=entry.get('amplitude'),
                    profile=entry.get('profile'),
                    how=entry.get('how'),
                    region = name
                )
                if name in main_regions:
                    comps = self._handle_main_line(base, n_narrow, n_broad)
                elif name in narrow_keys:
                    comps = self._handle_narrow_line(base, n_narrow, add_outflow)
                elif name == 'broad':
                    comps = self._handle_broad_line(base, n_broad)
                elif name in fe_regions and not template_mode_fe:
                    comps = [self._handle_fe_line(base)]
                else:
                    continue
                self.regions_to_fit.extend(comps)

        if (xmax - xmin) > POWER_LAW_RANGE_THRESHOLD and not force_linear:
            self.regions_to_fit.append(
                SpectralLine(center=0.0, line_name='powerlaw', kind='continuum', component=0,
                            profile='powerlaw',region='continuum')
            )

        # Build tied parameters
        self.tied_params.extend(
            fe_ties(self.regions_to_fit,by_region=by_region)
        )
        
        
    #    local_region_list: List[SpectralLine],
    # mainline_candidates: Union[str, List[str]],
    # n_narrow: int,
    # n_broad: int,
    # tied_narrow_to: Union[str, Dict[int, Dict[str, Any]]] = None,
    # tied_broad_to: Union[str, Dict[int, Dict[str, Any]]] = None,
    # known_tied_relations: List[Tuple[Tuple[str, ...], List[str]]] = None,
        
        self.tied_params.extend(
            region_ties(self.regions_to_fit,self.mainline_candidates,n_narrow,n_broad,
                        tied_narrow_to=tied_narrow_to,
                        tied_broad_to=tied_broad_to,
                        known_tied_relations=self.known_tied_relations))
        #tied_narrow_to add exeption that if narrow to is not in the region so it explode
        self.xmin, self.xmax = xmin, xmax
        self.n_narrow, self.n_broad = n_narrow, n_broad
        self.number_lines,self.number_tied = len(self.regions_to_fit),len(self.tied_params)
        self.template_mode_fe = template_mode_fe
        self.tied_narrow_to = tied_narrow_to
        self.tied_broad_to = tied_broad_to
    def _handle_main_line(
        self,
        entry: SpectralLine,
        n_narrow: int,
        n_broad: int
    ) -> List[SpectralLine]:
        comps: List[SpectralLine] = []
        total = n_narrow + n_broad
        for idx in range(total):
            kind = 'narrow' if idx < n_narrow else 'broad'
            comp_num = idx + 1 if kind == 'narrow' else idx - n_narrow + 1
            amp = SpectralLine.amplitude if kind == 'narrow' or comp_num == 1 else 0.5
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                kind=kind,
                component=comp_num,
                amplitude=amp,
                profile=entry.profile,
                how=entry.how,
                region = entry.region
            )
            comps.append(new)
        return comps

    def _handle_narrow_line(
        self,
        entry: SpectralLine,
        n_narrow: int,
        add_outflow: bool
    ) -> List[SpectralLine]:
        comps: List[SpectralLine] = []
        for idx in range(n_narrow):
            amp = SpectralLine.amplitude if idx == 0 else 0.5
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                kind='narrow',
                component=idx + 1,
                amplitude=amp,
                profile=entry.profile,
                region = entry.region
            )
            comps.append(new)
            if add_outflow and idx == 0 and 'OIII' in entry.line_name:
                out = SpectralLine(
                    center=entry.center,
                    line_name=entry.line_name,
                    kind='outflow',
                    component=OUTFLOW_COMPONENT,
                    amplitude= 0.5,
                    profile=entry.profile,
                    region = entry.region
                )
                comps.append(out)
        return comps

    def _handle_broad_line(
        self,
        entry: SpectralLine,
        n_broad: int
    ) -> List[SpectralLine]:
        comps: List[SpectralLine] = []
        for idx in range(n_broad):
            amp = SpectralLine.amplitude if idx == 0 else 0.5
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                kind='broad',
                component=idx + 1,
                amplitude=amp,
                profile=entry.profile,
                region = entry.region
            )
            comps.append(new)
        return comps

    def _handle_fe_line(self, entry: SpectralLine) -> SpectralLine:
        return SpectralLine(
            center=entry.center,
            line_name=entry.line_name,
            kind='fe',
            component=FE_COMPONENT,
            amplitude=0.1,
            how='sum',
            region = entry.region
        )
    def _fitting_rutine(self,add_step=True,tied_fe=False):
        "build a simple rutine to be fitted"
        _rutine_dict = {"complex_region":self.regions_to_fit,"fitting_rutine":
            {"step1":{"tied":self.tied_params,"non_optimize_in_axis":3,"learning_rate":1e-1,"num_steps":1000}},"outer_limits":[self.xmin,self.xmax],"inner_limits":[self.xmin+50,self.xmax-50]}
        if add_step:
            _tied_params = []
            _tied_params.extend(
            region_ties(self.regions_to_fit,self.mainline_candidates,self.n_narrow,self.n_broad,
                        tied_narrow_to=self.tied_narrow_to,
                        tied_broad_to=self.tied_broad_to,
                        known_tied_relations=self.known_tied_relations,only_known=True))
            if not self.template_mode_fe and tied_fe:
                _tied_params.extend(fe_ties(self.regions_to_fit))
            _rutine_dict["fitting_rutine"]["step2"] = {"tied":_tied_params,"non_optimize_in_axis":4,"learning_rate":1e-2,"num_steps":500}
        return _rutine_dict