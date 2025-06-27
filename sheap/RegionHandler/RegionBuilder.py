# Balmer continuum, Balmer High order emission lines
# 3646.0 limit for balmer continuum after this we can move to another stuff
# ADD NLR AS KIND LINE SEARCH FOR NLR PRONT IN THE SPECTRA
from __future__ import annotations

# from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from sheap.DataClass.DataClass import SpectralLine
from sheap.RegionHandler.utils import fe_ties, group_lines_by_region, region_ties,group_lines

# yaml_files =
# Named constants for special components
OUTFLOW_COMPONENT = 10
WINDS_COMPONENT = 15
FE_COMPONENT = 20
NLR_COMPONENT = 30
# hipper parameters should be
POWER_LAW_RANGE_THRESHOLD = 1000

class RegionBuilder:
    """
    Builds spectral fitting regions given a xmin and xmax, from YAML templates, with narrow, broad,
    outflow, and FeII components, plus parameter tying.
    
    
    
    
    """
    lines_prone_outflow = ["OIIIc","OIIIb","NeIIIa","OIIb","OIIa"]#,"NIIb","NIIa","SIIb","SIIa",]
    lines_prone_winds = ["CIVa","CIVb","AlIIIa","AlIIIb","MgII","HeIk","HeIId","Halpha","Hbeta","HeIe"]
    available_fe_modes = ["template","model","none"] # none is like No fe
    
    def __init__(
        self,
        xmin: float,
        xmax: float,
        n_narrow: int = 1,
        n_broad: int = 1,
        line_repository_path: Optional[List[Union[str, Path]]] = None,
        fe_mode = "template"
        #tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        #tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        #fe_regions=['fe_uv', "feii_IZw1", "feii_forbidden", "feii_coronal"],
        #fe_mode="template",  # "sum,combined,template"
        #grouped_method = False, #if this is true all the lines will be combine for kind, also is interesting see wich one could be the best one in this case 
        #add_outflow: bool = False,
        #add_narrow_plus: bool = False,
        #by_region: bool = False,
        
        #add_balmer_continuum: bool = False,
        
        #add_NLR : bool = False,
        #fe_tied_params=('center', 'fwhm'),
        #continuum_profile = "powerlaw",
        #no_fe = False
        ) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.n_narrow = n_narrow
        self.n_broad = n_broad
        self.fe_mode = fe_mode.lower()
        if self.fe_mode not in self.available_fe_modes:
            print(f"fe_mode: {self.fe_mode} not recognized moving to template, the current available are {self.available_fe_modes}")
            self.fe_mode = "template"
        if not line_repository_path:
            self.line_repository_path = list(Path(__file__).resolve().parent.glob("LineRepository/*.yaml"))
        self.lines_available: Dict[str, Any] = {}
        self._load_lines(self.line_repository_path) #this should be always here?
        self.make_region()

    def _load_lines(self, paths: Optional[List[Union[str, Path]]]) -> None:
        """
        Load YAML files with lines.
        """
        if not paths:
            raise ValueError("No YAML paths provided for region templates.")

        for p in paths:
            path = Path(p)
            if not path.is_file():
                raise FileNotFoundError(f"Region YAML not found: {path}")
            data = yaml.safe_load(path.read_text())
            key = path.stem
            if not isinstance(data, list) and not all(isinstance(item, dict) for item in data):
                raise KeyError(f"Not all element in the YAML are list filled with dict: {path}")
            self.lines_available[key] = data
        self.pseudo_region_available = list(self.lines_available.keys())
        
        
    def make_region(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        n_broad: Optional[int] = None,
        n_narrow: Optional[int] = None,
        fe_mode: Optional[str] = None,):
        def get(val, fallback):
            return val if val is not None else fallback
        xmin = get(xmin, self.xmin)
        xmax = get(xmax, self.xmax)
        n_broad = get(n_broad, self.n_broad)
        n_narrow = get(n_narrow, self.n_narrow)
        fe_mode = get(fe_mode, self.fe_mode).lower()#right?
        if fe_mode not in self.available_fe_modes:
            print(f"fe_mode: {fe_mode} not recognized moving to template, the current available are {self.available_fe_modes}")
            fe_mode = "template"
        self.complex_list = [] #place holder name  
        for pseudo_region_name,list_dict in self.lines_available.items():
            comps = []
            for raw_line in list_dict:
                center = float(raw_line.get('center', -np.inf))
                if not (xmin <= center <= xmax):
                    continue
                base = SpectralLine(**raw_line)
                if pseudo_region_name == "broad_and_narrow": #search of name
                    comps = self._handle_broad_and_narrow_lines(base, n_narrow, n_broad)
                elif pseudo_region_name == "narrows" and n_narrow>0:
                    comps = self._handle_narrow_line(base, n_narrow)
                elif pseudo_region_name == "broads" and n_broad>0:
                    comps = self._handle_broad_line(base, n_broad) 
                #elif self.fe_mode == "model":   
                self.complex_list.extend(comps)
        self.complex_list.extend(self._handle_fe(fe_mode,xmin,xmax))
        continuum_profile = ""
        self.complex_list.extend(self._continuum_handle(continuum_profile,xmax,xmin))
    # if name in main_regions:
    #                 comps = self._handle_main_line(base, n_narrow, n_broad,add_NLR,add_outflow)
    #             elif name in narrow_keys:
    #                 comps = self._handle_narrow_line(base, n_narrow, add_outflow)
    #             elif name == 'broad':
    #                 comps = self._handle_broad_line(base, n_broad)
    #             elif fe_mode == "sum" and name in fe_regions and not no_fe:
    #                 comps = [self._handle_fe_line(base)]
    #                 tie_fe = True
    #             #'fe_uv', "feii_IZw1", "feii_forbidden", "feii_coronal"
    #             elif fe_mode == "model" and not no_fe :
    #                 if  name in ["feii_model", "fe_uv"]:
    #                     comps = [self._handle_fe_line(base, how="combine")]
    #                 elif name in ["feii_coronal"]:
    #                     comps = [self._handle_fe_line(base)]
    #                 else:
    #                     continue 
                        
    #             else:
    #                 continue
    #             self.complex_region.extend(comps)
                
    def _handle_broad_and_narrow_lines(
        self, entry: SpectralLine, n_narrow: int, n_broad: int, add_winds=False) -> List[SpectralLine]:
        comps: List[SpectralLine] = []
        total = n_narrow + n_broad
        for idx in range(total):
            region = 'narrow' if idx < n_narrow else 'broad'
            comp_num = idx + 1 if region == 'narrow' else idx - n_narrow + 1
            amp = 1.0 if comp_num == 1 else 1.0/comp_num
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                region=region,
                component=comp_num,
                amplitude=amp,
                element=entry.element,
            )
            comps.append(new)
            if add_winds and idx == 0 and self.lines_prone_winds:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='winds',
                    component = WINDS_COMPONENT,
                    amplitude=0.5,
                    element = entry.element,
                )
                comps.append(out)
        return comps
    
    def _handle_narrow_line(
        self, entry: SpectralLine, n_narrow: int, add_outflow: bool = False, add_uncommon = False) -> List[SpectralLine]:
        comps: List[SpectralLine] = []
        for idx in range(n_narrow):
            amp = 1 if idx == 0 else 0.5
            if entry.rarity=="uncommon" and not add_uncommon:
                continue 
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                region ='narrow',
                component = idx + 1,
                amplitude =  amp,
                element = entry.element,
                rarity = entry.rarity
            )
            comps.append(new)
            if add_outflow and idx == 0 and self.line_prone_outflow:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='outflow',
                    component = OUTFLOW_COMPONENT,
                    amplitude=0.5,
                    element = entry.element,
                    rarity = entry.rarity)
                
                comps.append(out)
        return comps

    def _handle_broad_line(self, entry: SpectralLine, n_broad: int,add_winds=False) -> List[SpectralLine]:
        comps: List[SpectralLine] = []
        #extra broad? 
        #return comps
        for idx in range(n_broad):
            if idx>0:
                continue 
            amp = 1 if idx == 0 else 0.5
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                region='broad',
                component=idx + 1,
                amplitude=amp,
                element=entry.element,
            )
            comps.append(new)
            
            if add_winds and idx == 0 and self.lines_prone_winds:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='winds',
                    component = WINDS_COMPONENT,
                    amplitude=0.5,
                    element = entry.element,
                )
                comps.append(out)
                
        return comps
 
    def _handle_fe(self,fe_mode,xmin,xmax):
        fe_comps = []
        if fe_mode == "none":
            return fe_comps
        elif fe_mode == "template":
            t_c = 0
            if max(0, min(xmax, 7484) - max(xmin, 3686)) >= 1000:
                print("added OP template")
                fe_comps.extend(
                    [SpectralLine(center=None,line_name="feop",region="fe",component=FE_COMPONENT,profile="fitFeOP",how="template",which_template="OP",element="OP")])
                t_c += 1
            if max(0, min(xmax, 3500) - max(xmin, 1200)) >= 1000:
                print("added UV template")
                fe_comps.extend([SpectralLine(center=None,line_name="feuv",region="fe",component=FE_COMPONENT,profile="fitFeUV",how="template",which="UV",element="UV")])
                t_c += 1
            if t_c == 0:
                print("The covered range is not valid for template use. Switching to model mode. Work in progress, if no Fe wanted put fe_mode = none.")
                fe_mode = "model"
        elif fe_mode == "model":      
            for pseudo_region_name,list_dict in self.lines_available.items():
                for raw_line in list_dict:
                    center = float(raw_line.get('center', -np.inf))
                    if not (xmin <= center <= xmax) or pseudo_region_name not in ('feii_uv',"feii_model"):
                        continue
                    base = SpectralLine(**raw_line)
                    base.subregion = pseudo_region_name
                    fe_comps.extend([base])
        return fe_comps
    def _continuum_handle(self,continuum_profile,xmin,xmax):
        return []
class RegionBuilder_old:
    """
    Builds spectral fitting regions from YAML templates, with narrow, broad,
    outflow, and FeII components, plus parameter tying.
    """
    #- Arithmetic: "target source *2"  (target = source * 2)
    ##_, target, source, op, operand = dep
    known_tied_relations: List[Tuple[Tuple[str, ...], List[str]]] = [
        (
            ('OIIIb', 'OIIIc'),
            ['amplitude_OIIIb_component_narrow', 'amplitude_OIIIc_component_narrow', '*0.3'],
        ),
        (
            ('NIIa', 'NIIb'),
            ['amplitude_NIIa_component_narrow', 'amplitude_NIIb_component_narrow', '*0.3'],
        ),
        (('NIIa', 'NIIb'), ['center_NIIa_component_narrow', 'center_NIIb_component_narrow']),
        (
            ('OIIIb', 'OIIIc'),
            ['center_OIIIb_component_narrow', 'center_OIIIc_component_narrow'],
        ),
    ]

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n_narrow: int = 1,
        n_broad: int = 1,
        yaml_paths: Optional[List[Union[str, Path]]] = list(
            Path(__file__).resolve().parent.glob("LineRepository_old/*.yaml")
        ),
        tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        fe_regions=['fe_uv', "feii_IZw1", "feii_forbidden", "feii_coronal"],
        fe_mode="template",  # "sum,combined,template"
        grouped_method = False, #if this is true all the lines will be combine for kind, also is interesting see wich one could be the best one in this case 
        add_outflow: bool = False,
        add_narrow_plus: bool = False,
        by_region: bool = False,
        #force_linear: bool = False,
        add_balmer_continuum: bool = False,
        fe_tied_params=('center', 'fwhm'),
        add_NLR : bool = False,
        continuum_profile = "powerlaw",
        #powerlaw_profile: str = "powerlaw",
        no_fe = False
        # model_fii = False
    ) -> None:
        if fe_mode not in ["sum", "model", "template"]:
            print(f"fe_mode: {fe_mode} not recognized moving to template, the current available are sum, model, template")
            fe_mode = "template"
        if continuum_profile not in ['linear','powerlaw',"brokenpowerlaw"]:
            print(f"continuum_profile: {continuum_profile} not recognized moving to powerlaw the current available are linear,powerlaw,brokenpowerlaw")
            continuum_profile = "powerlaw"
        #basic ones
        self.xmin: float = xmin
        self.xmax: float = xmax
        self.n_narrow: int = n_narrow
        self.n_broad: int = n_broad
        self.continuum_profile = continuum_profile
        self.tied_narrow_to = tied_narrow_to
        self.tied_broad_to = tied_broad_to
        
        self.lines_regions_available: Dict[str, Any] = {}
        self.regions_available: List[str] = []
        self.by_region = by_region
        
        self._load_region_templates(yaml_paths) #this should be always here?
        self.tied_params: List[List[str]] = []
        #fe 
        self.fe_mode = fe_mode
        self.no_fe = no_fe
        self.fe_tied_params = fe_tied_params
        self.fe_regions = fe_regions
        #######
        # self.template_mode_fe = template_mode_fe
        # self.mainline_candidates = mainline_candidates
        #adds:
        self.add_outflow = add_outflow
        self.add_narrow_plus = add_narrow_plus
        self.add_balmer_continuum = add_balmer_continuum
        self.add_NLR = add_NLR
        ######
        #self.force_linear = force_linear
        #self.powerlaw_profile = powerlaw_profile
        self.grouped_method = grouped_method
        self.complex_region: List[SpectralLine] = []
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

    def __call__(self, add_step=True, tied_fe=False, num_steps_list=[3000, 3000]):
            """
            Build a fitting routine dictionary with support for multiple steps.
            
            Step1 uses self.tied_params, subsequent steps repeat known-tied logic.
            """
            fitting_routine = {}

            # Step 1: always present
            fitting_routine["step1"] = {
                "tied": self.tied_params,
                "non_optimize_in_axis": 3,
                "learning_rate": 1e-1,
                "num_steps": num_steps_list[0],
            }

            if add_step and len(num_steps_list) > 1:
                # Generate tied params for reuse in all later steps
                
                tied_later = region_ties(
                    self.complex_region,
                    self.n_narrow,
                    self.n_broad,
                    tied_narrow_to=self.tied_narrow_to,
                    tied_broad_to=self.tied_broad_to,
                    known_tied_relations=self.known_tied_relations,
                    only_known=True,
                )
                if self.grouped_method:
                   tied_later = [] 
                #print(tied_later)
                if self.fe_mode == "sum" and tied_fe:
                    tied_later.extend(fe_ties(self.complex_region))

                # Add step2, step3, ..., stepN
                for i, steps in enumerate(num_steps_list[1:], start=2):
                    fitting_routine[f"step{i}"] = {
                        "tied": tied_later,
                        "non_optimize_in_axis": 4,  # generalize axis as 4, 5, ...
                        "learning_rate": 1e-2,
                        "num_steps": steps,
                    }

            return {
                "complex_region": self.complex_region,
                "fitting_routine": fitting_routine,
                "outer_limits": [self.xmin, self.xmax],
                "inner_limits": [self.xmin + 50, self.xmax - 50],
                "model_keywords": self.model_keywords,
            }


    def make_region(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        n_broad: Optional[int] = None,
        n_narrow: Optional[int] = None,
        main_regions: List[str] = ['hydrogen', 'helium'],
        fe_regions: Optional[List[str]] = None,
        continuum_profile:Optional[str] = None, 
        tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        #force_linear: Optional[bool] = None,
        by_region: Optional[bool] = None,
        fe_tied_params: Optional[Tuple] = None,
        add_outflow: Optional[bool] = None,
        add_narrow_plus: Optional[bool] = None,
        add_balmer_continuum: Optional[Tuple] = None,
        grouped_method = None,
        add_NLR = None,
        fe_mode=None,
        no_fe = None,
        #powerlaw_profile = None,
        
    ) -> None:
        xmin = xmin if xmin is not None else self.xmin
        xmax = xmax if xmax is not None else self.xmax
        n_broad = n_broad if n_broad is not None else self.n_broad
        n_narrow = n_narrow if n_narrow is not None else self.n_narrow
        tied_narrow_to = tied_narrow_to if tied_narrow_to is not None else self.tied_narrow_to
        tied_broad_to = tied_broad_to if tied_broad_to is not None else self.tied_broad_to
        continuum_profile = continuum_profile if continuum_profile is not None else self.continuum_profile 
        add_outflow = add_outflow if add_outflow is not None else self.add_outflow
        add_narrow_plus = add_narrow_plus if add_narrow_plus is not None else self.add_narrow_plus
        fe_regions = fe_regions if fe_regions is not None else self.fe_regions
        add_NLR = add_NLR if add_NLR is not None else self.add_NLR
        add_balmer_continuum = (
            add_balmer_continuum
            if add_balmer_continuum is not None
            else self.add_balmer_continuum
        )
        # model_fii = model_fii if model_fii is not None else self.model_fii
        fe_mode = fe_mode if fe_mode is not None else self.fe_mode
        by_region = by_region if by_region is not None else self.by_region
        fe_tied_params = fe_tied_params if fe_tied_params is not None else self.fe_tied_params
        #powerlaw_profile = powerlaw_profile if powerlaw_profile is not None else self.powerlaw_profile
        no_fe = no_fe if no_fe is not None else self.no_fe
        grouped_method = grouped_method if grouped_method is not None else self.grouped_method
        # template = {"line_name":"feop","kind": "fe","component":20,"how":"template","which":"OP"}

        self.complex_region.clear()
        self.tied_params.clear()
        narrow_keys = ['narrow_basic'] + (['narrow_plus'] if add_narrow_plus else [])
        if fe_mode.lower() == "template" and not no_fe:  # and (xmax - xmin) > 1000:
            # the cuantity of pixels should be related to the the region in where the spectra have to be
            # if xmin>=3000 and xmax<=6000:
            # tested formes of
            t_c = 0
            if max(0, min(xmax, 7484) - max(xmin, 3686)) >= 1000:
                print("added OP template")
                self.complex_region.extend(
                    [
                        SpectralLine(
                            center=0,
                            line_name="feop",
                            kind="fe",  # fallback to empty
                            component=FE_COMPONENT + 1,
                            amplitude=0,
                            profile="fitFeOP",
                            how="template",
                            which="OP",
                            region="OP",
                        )
                    ]
                )
                t_c += 1
            if max(0, min(xmax, 3500) - max(xmin, 1200)) >= 1000:
                print("added UV template")
                self.complex_region.extend(
                    [
                        SpectralLine(
                            center=0,
                            line_name="feuv",
                            kind="fe",  # fallback to empty
                            component=FE_COMPONENT + 1,
                            amplitude=0,
                            profile="fitFeUV",
                            how="template",
                            which="UV",
                            region="UV",  # NOT SHURE
                        )
                    ]
                )
                t_c += 1
            if t_c == 0:
                print(
                    "the covered range is not accepted to use template moving to sum of lines mode n/ work in progress"
                )

                # template_mode_Fe = False
        if self.xmin > 3640.0 and add_balmer_continuum:
            print(
                "Warning: Balmer continiuum dosent have effect under 3640 A add_balmer_continuum change to False")
            add_balmer_continuum = False
        is_tied_broad = False if tied_broad_to is not None else True
        is_tied_narrow = False if tied_narrow_to is not None else True
        tie_fe = False
        Kinds = []
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
                    amplitude=entry.get('amplitude', 1.0),
                    profile=entry.get('profile'),
                    how=entry.get('how'),
                    region=entry.get('region', name),
                )
                if tied_broad_to is not None:
                    if isinstance(tied_broad_to, str) and tied_broad_to == base.line_name:
                        is_tied_broad = True
                    elif isinstance(tied_broad_to, (list, Tuple)):
                        is_tied_broad = True
                        print("work in progress")

                if tied_narrow_to is not None:
                    if isinstance(tied_narrow_to, str) and tied_narrow_to == base.line_name:
                        is_tied_narrow = True
                    elif isinstance(tied_narrow_to, (list, Tuple)):
                        is_tied_narrow = True
                        print("work in progress")

                if name in main_regions:
                    comps = self._handle_main_line(base, n_narrow, n_broad,add_NLR,add_outflow)
                elif name in narrow_keys:
                    comps = self._handle_narrow_line(base, n_narrow, add_outflow)
                elif name == 'broad':
                    comps = self._handle_broad_line(base, n_broad)
                elif fe_mode == "sum" and name in fe_regions and not no_fe:
                    comps = [self._handle_fe_line(base)]
                    tie_fe = True
                #'fe_uv', "feii_IZw1", "feii_forbidden", "feii_coronal"
                elif fe_mode == "model" and not no_fe :
                    if  name in ["feii_model", "fe_uv"]:
                        comps = [self._handle_fe_line(base, how="combine")]
                    elif name in ["feii_coronal"]:
                        comps = [self._handle_fe_line(base)]
                    else:
                        continue 
                        
                else:
                    continue
                self.complex_region.extend(comps)
        Kinds = np.unique(np.array([sp.kind for sp in self.complex_region]))
        assert is_tied_broad, f"'tied_broad_to': {tied_broad_to} not in the region"
        assert is_tied_narrow, f"'tied_narrow_to': {tied_narrow_to} not in the region"

        if add_balmer_continuum:
            self.complex_region.append(
                SpectralLine(
                    center=0.0,
                    line_name='balmerconti',
                    kind='continuum',
                    component=0,
                    profile='balmerconti',
                    region='continuum',
                )
            )

        #if (xmax - xmin) > POWER_LAW_RANGE_THRESHOLD:
        if 'powerlaw' in continuum_profile and (xmax - xmin) < POWER_LAW_RANGE_THRESHOLD:
            print('POWER_LAW_RANGE_THRESHOLD:',POWER_LAW_RANGE_THRESHOLD,"<",(xmax - xmin) )
        self.complex_region.append(
            SpectralLine(
                center=None,
                line_name=continuum_profile,
                kind='continuum',
                component=0,
                profile=continuum_profile,
                region='continuum',
            )
        )

       
        if not no_fe and not grouped_method:
            self.tied_params.extend(
                    fe_ties(self.complex_region, by_region=by_region, tied_params=fe_tied_params))
            
        if fe_mode == "model" and not no_fe and not grouped_method:
            print("put to true this after")
            self.complex_region = group_lines(self.complex_region,kind = "fe",mode="region") # in some place i have to add the restriction for feii_coronal
        
        if grouped_method:
            for k in Kinds:
                if k=="fe" or k=="outflow":
                    continue
                self.complex_region = group_lines(self.complex_region,kind = k,profile="SPAF",mode="kind", exception_region = [],known_tied_relations = self.known_tied_relations)
            if fe_mode == "model":
                self.complex_region = group_lines(self.complex_region,kind = "fe",mode="region",profile="SPAF")
                #self.complex_region = [i for i in self.complex_region if i.region not in ["feii_coronal"]] #hard to see 
            self.grouped_method = grouped_method    
        if not grouped_method:
            self.tied_params.extend(
                region_ties(
                    self.complex_region,
                    n_narrow,
                    n_broad,
                    tied_narrow_to=tied_narrow_to,
                    tied_broad_to=tied_broad_to,
                    known_tied_relations=self.known_tied_relations,
                )
            )
            

        self.xmin, self.xmax = xmin, xmax
        self.n_narrow, self.n_broad = n_narrow, n_broad
        self.number_lines, self.number_tied = len(self.complex_region), len(self.tied_params)
        # self.template_mode_fe = template_mode_fe
        self.tied_narrow_to = tied_narrow_to
        self.tied_broad_to = tied_broad_to
        self.model_keywords = {
            "n_broad": n_broad,
            "n_narrow": n_narrow,
            "add_outflow": add_outflow,
            "fe_mode": fe_mode,
        }


    def _handle_main_line(
        self, entry: SpectralLine, n_narrow: int, n_broad: int, add_NLR:bool , add_outflow:bool
    ) -> List[SpectralLine]:
        comps: List[SpectralLine] = []
        total = n_narrow + n_broad
        for idx in range(total):
            kind = 'narrow' if idx < n_narrow else 'broad'
            comp_num = idx + 1 if kind == 'narrow' else idx - n_narrow + 1
            amp = SpectralLine.amplitude if kind == 'narrow' or comp_num == 1 else 1.0/comp_num
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                kind=kind,
                component=comp_num,
                amplitude=amp,
                profile=entry.profile,
                how=entry.how,
                region=entry.region,
            )
            
            if comp_num>1 and entry.region == "hydrogen":
                comps.append(new)
            elif comp_num==1:
                comps.append(new)
            else:
                continue
            
        if add_outflow and 'Halpha' in entry.line_name:
            out = SpectralLine(
                    center=entry.center,
                    line_name=entry.line_name,
                    kind='outflow',
                    component=OUTFLOW_COMPONENT,
                    amplitude=0.5,
                    profile=entry.profile,
                    region=entry.region,
                )
            comps.append(out)
                   
        if add_NLR and entry.region == "hydrogen":
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                kind="nlr",
                component=NLR_COMPONENT,
                amplitude=amp*0.1,
                profile=entry.profile,
                how=entry.how,
                region=entry.region,
            )
            comps.append(new)
        return comps

    def _handle_narrow_line(
        self, entry: SpectralLine, n_narrow: int, add_outflow: bool
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
                region=entry.region,
            )
            comps.append(new)
            if add_outflow and idx == 0 and 'OIII' in entry.line_name:
                out = SpectralLine(
                    center=entry.center,
                    line_name=entry.line_name,
                    kind='outflow',
                    component=OUTFLOW_COMPONENT,
                    amplitude=0.5,
                    profile=entry.profile,
                    region=entry.region,
                )
                comps.append(out)
        return comps

    def _handle_broad_line(self, entry: SpectralLine, n_broad: int) -> List[SpectralLine]:
        comps: List[SpectralLine] = []
        #extra broad? 
        #return comps
        for idx in range(n_broad):
            if idx>0:
                continue 
            amp = SpectralLine.amplitude if idx == 0 else 0.5
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                kind='broad',
                component=idx + 1,
                amplitude=amp,
                profile=entry.profile,
                region=entry.region,
            )
            comps.append(new)
        return comps

    def _handle_fe_line(self, entry: SpectralLine, how='sum') -> SpectralLine:
        return SpectralLine(
            center=entry.center,
            line_name=entry.line_name,
            kind='fe',
            component=FE_COMPONENT,
            amplitude=0.1 if entry.amplitude == 1.0 else entry.amplitude,
            how=how,
            region=entry.region,
        )

    # def _fitting_routine(self,add_step=True,tied_fe=False,num_steps_list=[1000,500]):
    #     "build a simple rutine to be fitted"
    #     print("xd")
    #     _rutine_dict = {"complex_region":self.complex_region,"fitting_routine":
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
    #         _rutine_dict["fitting_routine"]["step2"] = {"tied":_tied_params,"non_optimize_in_axis":4,"learning_rate":1e-2,"num_steps":num_steps_list[1]}
    #     return _rutine_dict
