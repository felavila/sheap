# Balmer continuum, Balmer High order emission lines
#3646.0 limit for balmer continuum after this we can move to another stuff 
from __future__ import annotations

#from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from sheap.RegionHandler.utils import fe_ties, region_ties,group_lines_by_region
from sheap.DataClass.DataClass import SpectralLine

#yaml_files = 
# Named constants for special components
OUTFLOW_COMPONENT = 10
FE_COMPONENT = 20
#hipper parameters should be 
POWER_LAW_RANGE_THRESHOLD = 1000

    
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
        yaml_paths: Optional[List[Union[str, Path]]] = list(Path(__file__).resolve().parent.glob("LineRepository/*.yaml")),
        tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,        
        fe_regions = ['fe_uv',"feII_IZw1","feII_forbidden","feII_coronal"],
        fe_mode = "template", #"sum,combined,template"
        add_outflow:bool = False,
        add_narrowplus:bool = False,
        by_region:bool = False,
        force_linear:bool = False,
        add_balmercontiniumm: bool = False,
        fe_tied_params = ('center', 'width'),
        #model_fii = False
        
    ) -> None:
        if fe_mode not in ["sum","model","template"]:
            print(f"fe_mode:{fe_mode} not recognized moving to template")
            fe_mode = "template"
        self.xmin: float = xmin
        self.xmax: float = xmax
        self.tied_narrow_to = tied_narrow_to
        self.tied_broad_to = tied_broad_to
        self.n_narrow: int = n_narrow
        self.n_broad: int = n_broad
        self.lines_regions_available: Dict[str, Any] = {}
        self.regions_available: List[str] = []
        self.complex_region: List[SpectralLine] = []
        self._load_region_templates(yaml_paths)
        self.tied_params: List[List[str]] = []
        self.fe_mode = fe_mode
        #self.template_mode_fe = template_mode_fe
        #self.mainline_candidates = mainline_candidates
        self.add_outflow = add_outflow
        self.add_narrowplus = add_narrowplus
        self.fe_regions = fe_regions
        self.by_region = by_region
        self.fe_tied_params = fe_tied_params
        self.force_linear = force_linear
        self.add_balmercontiniumm = add_balmercontiniumm
        #self.model_fii = model_fii
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

    def __call__(self,add_step=True,tied_fe=False,num_steps_list=[3000,3000]):
        "build a simple rutine to be fitted"
        _rutine_dict = {"complex_region":self.complex_region,"fitting_rutine":
            {"step1":{"tied":self.tied_params,"non_optimize_in_axis":3,"learning_rate":1e-1,"num_steps":num_steps_list[0]}},
            "outer_limits":[self.xmin,self.xmax],"inner_limits":[self.xmin+50,self.xmax-50],"model_keywords":self.model_keywords}
        if add_step:
            _tied_params = []
            _tied_params.extend(
            region_ties(self.complex_region,self.n_narrow,self.n_broad,
                        tied_narrow_to=self.tied_narrow_to,
                        tied_broad_to=self.tied_broad_to,
                        known_tied_relations=self.known_tied_relations,only_known=True))
            if self.fe_mode=="sum" and tied_fe:
                _tied_params.extend(fe_ties(self.complex_region))
            _rutine_dict["fitting_rutine"]["step2"] = {"tied":_tied_params,"non_optimize_in_axis":4,"learning_rate":1e-2,"num_steps":num_steps_list[1]}
        return _rutine_dict
    
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
        #template_mode_fe: Optional[bool] = None,
        tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        force_linear: Optional[bool] = None,
        #mainline_candidates = None,
        by_region: Optional[bool] = None,
        fe_tied_params: Optional[Tuple] = None,
        add_balmercontiniumm: Optional[Tuple] = None,
        #model_fii: Optional[Tuple] = None,
        fe_mode = None
    ) -> None:
        # Override defaults
        xmin = xmin if xmin is not None else self.xmin
        xmax = xmax if xmax is not None else self.xmax
        n_broad = n_broad if n_broad is not None else self.n_broad
        n_narrow = n_narrow if n_narrow is not None else self.n_narrow
        tied_narrow_to = tied_narrow_to if tied_narrow_to is not None else self.tied_narrow_to
        tied_broad_to =  tied_broad_to if tied_broad_to is not None else self.tied_broad_to
        #mainline_candidates =  mainline_candidates #if mainline_candidates is not None else self.mainline_candidates
        #template_mode_fe =  template_mode_fe if template_mode_fe is not None else self.template_mode_fe
        add_outflow = add_outflow if add_outflow is not None else self.add_outflow
        add_narrowplus = add_narrowplus if add_narrowplus is not None else self.add_narrowplus
        fe_regions = fe_regions if fe_regions is not None else self.fe_regions
        force_linear = force_linear if force_linear is not None else self.force_linear
        
        add_balmercontiniumm = add_balmercontiniumm if add_balmercontiniumm is not None else self.add_balmercontiniumm
        #model_fii = model_fii if model_fii is not None else self.model_fii
        fe_mode = fe_mode if fe_mode is not None else self.fe_mode
        by_region = by_region if by_region is not None else self.by_region
        fe_tied_params = fe_tied_params if fe_tied_params is not None else self.fe_tied_params
            #template = {"line_name":"feop","kind": "fe","component":20,"how":"template","which":"OP"}
        
        self.complex_region.clear()
        self.tied_params.clear()
        narrow_keys = ['narrow_basic'] + (['narrow_plus'] if add_narrowplus else [])
        if fe_mode.lower() == "template": #and (xmax - xmin) > 1000:
            #the cuantity of pixels should be related to the the region in where the spectra have to be 
            #if xmin>=3000 and xmax<=6000:
            #tested formes of 
            t_c = 0
            if max(0, min(xmax, 7484) - max(xmin, 3686))>= 1000:
                print("added OP template")
                self.complex_region.extend([SpectralLine(
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
                t_c += 1
            if max(0, min(xmax, 3500) - max(xmin, 1200))>= 1000:
                print("added UV template")
                self.complex_region.extend([SpectralLine(
                    center=0,
                    line_name="feuv",
                    kind="fe",  # fallback to empty
                    component= FE_COMPONENT+1,
                    amplitude=0,
                    profile="fitFeUV",
                    how="template",
                    which="UV",
                    region = "UV" #NOT SHURE 
                )])
                t_c += 1
            if t_c == 0:
                print("the covered range is not accepted to use template moving to sum of lines mode n/ work in progress")
                
                #template_mode_Fe = False
        if self.xmin > 3640. and add_balmercontiniumm:
            print("Warning: Balmer continiuum dosent have effect under 3640 A add_balmercontiniumm change to False")
            add_balmercontiniumm = False
        is_tied_broad = False if tied_broad_to is not None else True 
        is_tied_narrow = False if tied_narrow_to is not None else True 
        tie_fe = False 
        #print(model_fii)
        for name, region in self.lines_regions_available.items():
            #print(name)
            for entry in region['region']:
                
                center = float(entry.get('center', -np.inf))
                if not (xmin <= center <= xmax):
                    continue
                base = SpectralLine(
                    center=center,
                    line_name=str(entry['line_name']),
                    kind=str(entry.get('kind', '')),  # fallback to empty
                    component=int(entry.get('component', 1)),
                    amplitude=entry.get('amplitude', 1.0),
                    profile=entry.get('profile'),
                    how=entry.get('how'),
                    region = entry.get('region',name)
                )
                if tied_broad_to is not None:
                    if isinstance(tied_broad_to,str) and tied_broad_to==base.line_name:
                        is_tied_broad = True
                    elif isinstance(tied_broad_to,(list,Tuple)):
                        is_tied_broad = True
                        print("work in progress")
                
                if tied_narrow_to is not None:
                    if isinstance(tied_narrow_to,str) and tied_narrow_to==base.line_name:
                        is_tied_narrow = True
                    elif isinstance(tied_narrow_to,(list,Tuple)):
                        is_tied_narrow = True
                        print("work in progress")                         
                        
                if name in main_regions:
                    comps = self._handle_main_line(base, n_narrow, n_broad)
                elif name in narrow_keys:
                    comps = self._handle_narrow_line(base, n_narrow, add_outflow)
                elif name == 'broad':
                    comps = self._handle_broad_line(base, n_broad)
                elif name in fe_regions and fe_mode=="sum":
                    comps = [self._handle_fe_line(base)]
                    tie_fe = True 
                elif fe_mode=="model":
                    if name in ["feII_model","fe_uv"]:
                        comps = [self._handle_fe_line(base,how="combine")]
                else:
                    continue
                self.complex_region.extend(comps)
        
         
        assert is_tied_broad, f"'tied_broad_to': {tied_broad_to} not in the region"
        assert is_tied_narrow, f"'tied_narrow_to': {tied_narrow_to} not in the region"

        if add_balmercontiniumm:
         self.complex_region.append(
                 SpectralLine(center=0.0, line_name='balmerconti', kind='continuum', component=0,
                             profile='balmerconti',region='continuum')
             )
        
        
        if (xmax - xmin) > POWER_LAW_RANGE_THRESHOLD and not force_linear:
            self.complex_region.append(
                SpectralLine(center=0.0, line_name='powerlaw', kind='continuum', component=0,
                            profile='powerlaw',region='continuum')
            )

        # Build tied parameters
        if tie_fe:
            self.tied_params.extend(
                fe_ties(self.complex_region,by_region=by_region,tied_params=fe_tied_params)
            )
        
        self.tied_params.extend(
            region_ties(self.complex_region,n_narrow,n_broad,
                        tied_narrow_to=tied_narrow_to,
                        tied_broad_to=tied_broad_to,
                        known_tied_relations=self.known_tied_relations))
        if fe_mode == "model":
            self.complex_region = group_lines_by_region(self.complex_region)
        
    #    local_region_list: List[SpectralLine],
    # mainline_candidates: Union[str, List[str]],
    # n_narrow: int,
    # n_broad: int,
    # tied_narrow_to: Union[str, Dict[int, Dict[str, Any]]] = None,
    # tied_broad_to: Union[str, Dict[int, Dict[str, Any]]] = None,
    # known_tied_relations: List[Tuple[Tuple[str, ...], List[str]]] = None,
        
        
        #tied_narrow_to add exeption that if narrow to is not in the region so it explode
        self.xmin, self.xmax = xmin, xmax
        self.n_narrow, self.n_broad = n_narrow, n_broad
        self.number_lines,self.number_tied = len(self.complex_region),len(self.tied_params)
        #self.template_mode_fe = template_mode_fe
        self.tied_narrow_to = tied_narrow_to
        self.tied_broad_to = tied_broad_to
        self.model_keywords = {"n_broad":n_broad,"n_narrow":n_narrow,"add_outflow":add_outflow,"fe_mode":fe_mode}
        
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

    def _handle_fe_line(self, entry: SpectralLine,how='sum') -> SpectralLine:
        return SpectralLine(
            center=entry.center,
            line_name=entry.line_name,
            kind='fe',
            component=FE_COMPONENT,
            amplitude= 0.1 if entry.amplitude==1.0 else entry.amplitude,
            how=how,
            region = entry.region
        )
    
    # def _fitting_rutine(self,add_step=True,tied_fe=False,num_steps_list=[1000,500]):
    #     "build a simple rutine to be fitted"
    #     print("xd")
    #     _rutine_dict = {"complex_region":self.complex_region,"fitting_rutine":
    #         {"step1":{"tied":self.tied_params,"non_optimize_in_axis":3,"learning_rate":1e-1,"num_steps":num_steps_list[0]}},
    #         "outer_limits":[self.xmin,self.xmax],"inner_limits":[self.xmin+50,self.xmax-50],"model_keywords":self.model_keywords}
    #     if add_step:
    #         _tied_params = []
    #         _tied_params.extend(
    #         region_ties(self.complex_region,None,self.n_narrow,self.n_broad,
    #                     tied_narrow_to=self.tied_narrow_to,
    #                     tied_broad_to=self.tied_broad_to,
    #                     known_tied_relations=self.known_tied_relations,only_known=True))
    #         if not self.template_mode_fe and tied_fe:
    #             _tied_params.extend(fe_ties(self.complex_region))
    #         _rutine_dict["fitting_rutine"]["step2"] = {"tied":_tied_params,"non_optimize_in_axis":4,"learning_rate":1e-2,"num_steps":num_steps_list[1]}
    #     return _rutine_dict