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
from sheap.DataClass.ComplexRegion import ComplexRegion
from sheap.Functions.profiles import PROFILE_FUNC_MAP,PROFILE_CONTINUUM_FUNC_MAP
from sheap.RegionHandler.utils import fe_ties, region_ties, group_lines

# yaml_files =
# Named constants for special components
OUTFLOW_COMPONENT = 10
WINDS_COMPONENT = 15
FE_COMPONENT = 20
NLR_COMPONENT = 30
# Now looks alot more clear this. 

#TODO add the uncommon lines narrow?
class RegionBuilder:
    """
    Builds spectral fitting regions given a xmin and xmax, from YAML templates, with narrow, broad,
    outflow, and FeII components, plus parameter tying.
    """
    
    lines_prone_outflow = ["OIIIc","OIIIb"]#,"NeIIIa","OIIb","OIIa"]#,"NIIb","NIIa","SIIb","SIIa",]
    lines_prone_winds = ["CIVa","CIVb","AlIIIa","AlIIIb","MgII","Halpha","Hbeta"]#,"HeIe","HeIk","HeIId"]
    available_fe_modes = ["template","model","none"] # none is like No fe
    available_continuum_profiles = list(PROFILE_CONTINUUM_FUNC_MAP.keys())
    LINEAR_RANGE_THRESHOLD = 1000
    known_tied_relations: List[Tuple[Tuple[str, ...], List[str]]] = [(('OIIIb', 'OIIIc'),['amplitude_OIIIb_component_narrow', 'amplitude_OIIIc_component_narrow', '*0.3'],),
        (('NIIa', 'NIIb'),['amplitude_NIIa_component_narrow', 'amplitude_NIIb_component_narrow', '*0.3'],),
        (('NIIa', 'NIIb'), ['center_NIIa_component_narrow', 'center_NIIb_component_narrow']),
        (('OIIIb', 'OIIIc'),['center_OIIIb_component_narrow', 'center_OIIIc_component_narrow'],),]
    
    def __init__(
        self,
        xmin: float,
        xmax: float,
        n_narrow: int = 1,
        n_broad: int = 1,
        line_repository_path: Optional[List[Union[str, Path]]] = None,
        fe_mode = "template",
        continuum_profile = "powerlaw",
        group_method = False,
        add_outflow = False,
        add_winds = False,
        add_balmer_continuum = False,
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
        self.group_method = group_method
        self.add_balmer_continuum = add_balmer_continuum
        self.fe_mode = fe_mode.lower()
        self.add_outflow = add_outflow
        self.add_winds = add_winds
        if self.fe_mode not in self.available_fe_modes:
            print(f"fe_mode: {self.fe_mode} not recognized moving to template, the current available are {self.available_fe_modes}")
            self.fe_mode = "template"
        self.continuum_profile = continuum_profile.lower()
        if self.continuum_profile not in self.available_continuum_profiles:
            print(f"continuum_profile: {self.continuum_profile} not recognized moving to powerlaw, the current available are {self.available_continuum_profiles}")
            self.continuum_profile = "powerlaw"
        
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
        fe_mode: Optional[str] = None,
        continuum_profile: Optional[str] = None,
        group_method: Optional[bool] = None,
        add_outflow= None,
        add_winds = None,
        add_balmer_continuum = None):
        
        def get(val, fallback):
            return val if val is not None else fallback
        
        xmin = get(xmin, self.xmin)
        xmax = get(xmax, self.xmax)
        n_broad = get(n_broad, self.n_broad)
        n_narrow = get(n_narrow, self.n_narrow)
        fe_mode = get(fe_mode, self.fe_mode).lower()#right?
        add_outflow = get(add_outflow, self.add_outflow)
        add_balmer_continuum = get(add_balmer_continuum, self.add_balmer_continuum)
        add_winds = get(add_winds, self.add_winds)
        continuum_profile = get(continuum_profile, self.continuum_profile).lower()#right?
        
        if fe_mode not in self.available_fe_modes:
            print(f"fe_mode: {fe_mode} not recognized moving to template, the current available are {self.available_fe_modes}")
            fe_mode = "template"
        if continuum_profile not in self.available_continuum_profiles:
            print(f"continuum_profile: {continuum_profile} not recognized moving to powerlaw, the current available are {self.available_continuum_profiles}")
            continuum_profile = "powerlaw"
        self.group_method = get(group_method,self.group_method)
        
        self.complex_list = [] #place holder name  
        for pseudo_region_name,list_dict in self.lines_available.items():
            comps = []
            for raw_line in list_dict:
                center = float(raw_line.get('center', -np.inf))
                if not (xmin <= center <= xmax):
                    continue
                base = SpectralLine(**raw_line)
                if pseudo_region_name == "broad_and_narrow": #search of name
                    comps = self._handle_broad_and_narrow_lines(base, n_narrow, n_broad,add_winds=add_winds)
                elif pseudo_region_name == "narrows" and n_narrow>0:
                    comps = self._handle_narrow_line(base, n_narrow,add_outflow=add_outflow)
                elif pseudo_region_name == "broads" and n_broad>0:
                    comps = self._handle_broad_line(base, n_broad,add_winds=add_winds) 
                #elif self.fe_mode == "model":   
                self.complex_list.extend(comps)
        self.complex_list.extend(self._handle_fe(fe_mode,xmin,xmax))
        self.complex_list.extend(self._continuum_handle(continuum_profile,xmin,xmax,add_balmer_continuum=add_balmer_continuum))#here we already are able to create the complex_class
        self.complex_class = ComplexRegion(self.complex_list)
        self.tied_relations = []
        if self.group_method:
             self.complex_class = self._apply_group_method(self.complex_class,fe_mode,self.known_tied_relations) #
             #self.tied_relations = []
        else:
            #todo add the tied_broad_to and narrow_to in cases in where is best use a line selected for the user
            self.tied_relations.extend(region_ties(self.complex_class,tied_narrow_to = None, tied_broad_to = None,known_tied_relations=self.known_tied_relations))
            if fe_mode not in ["none","template"]:
                routine_fe_tied = {"by":"subregion","tied_params": ('center', 'fwhm')}
                self.tied_relations.extend(fe_ties(self.complex_class.group_by("region").get("fe").lines, routine_fe_tied))
            
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
            if add_winds and idx == 0 and new.line_name in self.lines_prone_winds:
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
            if add_outflow and idx == 0 and new.line_name in self.lines_prone_outflow:
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
                    base.component = FE_COMPONENT
                    fe_comps.extend([base])
        return fe_comps
    
    def _continuum_handle(self,continuum_profile,xmin,xmax,add_balmer_continuum=False):
        continuum_comps = []
        if add_balmer_continuum and not xmax<3646:
            continuum_comps.append(SpectralLine(line_name='balmercontinuum',region='continuum',component=0,profile='balmercontinuum'))
        if 'linear' != continuum_profile and (xmax - xmin) < self.LINEAR_RANGE_THRESHOLD:
            print(f"xmax - xmin less than LINEAR_RANGE_THRESHOLD:{self.LINEAR_RANGE_THRESHOLD} < {(xmax - xmin)} moving to linear continuum")
            continuum_comps.append(SpectralLine(line_name="linear",region='continuum',component=0,profile="linear"))
            return continuum_comps
        continuum_comps.append(SpectralLine(line_name=continuum_profile,region='continuum',component=0,profile=continuum_profile))
        return continuum_comps

    def _apply_group_method(self,complex_class,fe_mode,known_tied_relations):
        #this can be a function outside.
        dict_regions = complex_class.group_by("region")
        new_complex_list = []
        for key,values in dict_regions.items():
            if key in ["outflow","winds","continuum"]:
                new_complex_list.extend(values.lines)
            elif key == "fe":
                #here much more can be done 
                if fe_mode=="model":
                    new_complex_list.extend(group_lines(values.lines,"fe",mode="element",profile="SPAF"))
                else:
                    new_complex_list.extend(values.lines)
            else:
                new_complex_list.extend(group_lines(values.lines,key,mode="region",known_tied_relations=known_tied_relations,profile="SPAF"))
        return ComplexRegion(new_complex_list)

    def _make_fitting_routine(self,list_num_steps = [1000],list_learning_rate = [1e-1]):
        #?
        fitting_routine = {}
        fitting_routine["step1"] = {"tied": self.tied_relations,"non_optimize_in_axis": 3,"learning_rate": list_learning_rate[0],"num_steps": list_num_steps[0]}
        assert len(list_num_steps) == len(list_learning_rate), "len(list_num_steps) != len(list_learning_rate) "
        for i, steps in enumerate(list_num_steps[1:], start=2):
            if self.group_method:
               tied_params = []
            else:
                print("add other tieds") 
            fitting_routine[f"step{i}"] = {"tied": [],"non_optimize_in_axis": 4,"learning_rate":list_learning_rate[i-1],"num_steps": list_num_steps[i-1]}
        return {"complex_class": self.complex_class,"outer_limits": [self.xmin, self.xmax], "inner_limits": [self.xmin + 50, self.xmax - 50],"fitting_routine":fitting_routine}
