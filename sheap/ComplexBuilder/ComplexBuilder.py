"""This module ."""
from __future__ import annotations
__author__ = 'felavila'

__all__ = [
    "ComplexBuilder",
]

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import yaml

from sheap.Core import SpectralLine,ComplexRegion
from sheap.ComplexBuilder.utils import fe_ties, _maketies, group_lines # asistant material

from sheap.Profiles.profiles import PROFILE_CONTINUUM_FUNC_MAP
from sheap.Profiles.profiles_templates import make_host_function #?



#TODO ADD the rutines of gaussians and tied methods in general. 
# Balmer continuum, Balmer High order emission lines
# 3646.0 limit for balmer continuum after this we can move to another stuff
# ADD NLR AS KIND LINE SEARCH FOR NLR PRONT IN THE SPECTRA

class ComplexBuilder:
    """
    Builds spectral fitting regions given wavelength bounds and YAML templates,
    including narrow, broad, outflow, and FeII components, plus parameter tying.

    Parameters
    ----------
    xmin : float
        Lower wavelength bound of the region (Å).
    xmax : float
        Upper wavelength bound of the region (Å).
    n_narrow : int, optional
        Number of narrow components per line, by default 1.
    n_broad : int, optional
        Number of broad components per line, by default 1.
    line_repository_path : list of str or Path, optional
        Paths to YAML files defining line templates. Defaults to all in `LineRepository/`.
    fe_mode : {'template', 'model', 'none'}, optional
        Mode for FeII components, by default "template".
    continuum_profile : str, optional
        Continuum profile name (must be in PROFILE_CONTINUUM_FUNC_MAP), by default "powerlaw".
    group_method : bool, optional
        Whether to apply automatic grouping and tying of parameters, by default True.
    add_outflow : bool, optional
        Include outflow components for narrow lines, by default False.
    add_winds : bool, optional
        Include wind components for broad lines, by default False.
    add_balmer_continuum : bool, optional
        Include Balmer continuum component, by default False.
    add_balmerhighorder_continuum : bool, optional
        Include Balmer high order continuum component, by default False.
    add_uncommon_narrow : bool, optional
        Include uncommon narrow lines, by default False.
    add_host_miles : bool or dict, optional
        Include host‐galaxy template from MILES; if dict, passes its keys to the builder.
    verbose : bool, optional
        Print informational messages during building, by default True.

    Attributes
    ----------
    lines_available : dict[str, list of dict]
        Loaded line definitions from YAML.
    pseudo_region_available : list[str]
        Keys of available pseudo‑regions from YAML.
    complex_class : ComplexRegion
        Container of all SpectralLine objects after building.
    tied_relations : list
        Parameter‐tying specifications used in fitting routine.

    Examples
    --------
    >>> rb = ComplexBuilder(6500, 6600, n_narrow=2, n_broad=1, add_outflow=True)
    >>> rb.make_region()
    >>> routine = rb._make_fitting_routine(list_num_steps=[2000,2000], list_learning_rate=[1e-1,1e-2])
    """
    
    #OUTFLOW_COMPONENT = 10
    #WINDS_COMPONENT = 15
    #FE_COMPONENT = 20
    #NLR_COMPONENT = 30
    #HOST_COMPONENT = 40 
    #BAL_COMPONENT = 50 
    lines_prone_outflow = ["OIIIc","OIIIb","NeIIIa","OIIb","OIIa"]#,"NIIb","NIIa","SIIb","SIIa",]
    lines_prone_winds = ["CIVa","CIVb","AlIIIa","AlIIIb","MgIIa","Halpha","Hbeta"]#,"HeIe","HeIk","HeIId"]
    lines_prone_bal = ["CIVa","CIVb","AlIIIa","AlIIIb","MgIIa","NVa","NVb","SiIV","OIV]","AlIIIa","AlIIIb","MgIIb"," OVIa"," OVIb"]#,"HeIe","HeIk","HeIId"]
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
        group_method = True,
        add_outflow = False,
        add_winds = False,
        add_balmer_continuum = False,
        add_uncommon_narrow = False,
        add_BAL = False,
        add_balmerhighorder_continuum = False,
        add_host_miles: Optional[Union[Dict,bool]] = None,
        tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        n_max_component_outflow = 1,
        n_max_component_winds = 1,
        n_max_component_bal = 1,
        #fe_regions=['fe_uv', "feii_IZw1", "feii_forbidden", "feii_coronal"],
        #fe_tied_params=('center', 'fwhm'),
        verbose=True,
        ) -> None:
        """
        Initialize the ComplexBuilder with region bounds and options.

        Parameters
        ----------
        xmin : float
            Minimum wavelength (Å) for region.
        xmax : float
            Maximum wavelength (Å) for region.
        n_narrow : int, optional
            Number of narrow-line components per line.
        n_broad : int, optional
            Number of broad-line components per line.
        line_repository_path : list of str or Path, optional
            Filepaths to YAML templates for lines.
        fe_mode : str, optional
            FeII handling mode: 'template', 'model', or 'none'.
        continuum_profile : str, optional
            Continuum profile to use ('powerlaw', etc.).
        group_method : bool, optional
            Whether to group and tie parameters automatically.
        add_outflow : bool, optional
            Whether to include outflow components.
        add_winds : bool, optional
            Whether to include wind components.
        add_balmer_continuum : bool, optional
            Whether to include Balmer continuum.
        add_uncommon_narrow : bool, optional
            Whether to include uncommon narrow lines.
        add_host_miles : bool or dict, optional
            Host-galaxy template inclusion or options dict.
        verbose : bool, optional
            Print status messages.
        """
        self.xmin = xmin
        self.xmax = xmax
        self.n_narrow = n_narrow
        self.n_broad = n_broad
        self.group_method = group_method
        self.add_balmer_continuum = add_balmer_continuum
        self.fe_mode = fe_mode.lower()
        self.add_outflow = add_outflow
        self.add_winds = add_winds
        self.add_uncommon_narrow = add_uncommon_narrow
        self.add_balmerhighorder_continuum = add_balmerhighorder_continuum
        self.verbose = verbose
        self.add_host_miles = add_host_miles
        self.tied_broad_to = tied_broad_to
        self.tied_narrow_to = tied_narrow_to
        self.add_BAL = add_BAL
        self.n_max_component_outflow = n_max_component_outflow
        self.n_max_component_winds = n_max_component_winds
        self.n_max_component_bal = n_max_component_bal
        if self.fe_mode not in self.available_fe_modes:
            print(f"fe_mode: {self.fe_mode} not recognized moving to template, the current available are {self.available_fe_modes}")
            self.fe_mode = "template"
        self.continuum_profile = continuum_profile.lower()
        if self.continuum_profile not in self.available_continuum_profiles:
            print(f"continuum_profile: {self.continuum_profile} not recognized moving to powerlaw, the current available are {self.available_continuum_profiles}")
            self.continuum_profile = "powerlaw"
        
        if not line_repository_path:
            TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "SuportData" / "LineRepository"
            self.line_repository_path = list(TEMPLATES_PATH.glob("*.yaml"))
        self.lines_available: Dict[str, Any] = {}
        self._load_lines(self.line_repository_path) #this should be always here?
        self.make_region()
        
             
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
        add_balmer_continuum = None,
        add_uncommon_narrow = None,
        add_host_miles = None,
        tied_broad_to= None,
        tied_narrow_to = None,
        add_BAL = None,
        add_balmerhighorder_continuum = None):
        """
        Build a `ComplexRegion` of `SpectralLine` objects based on settings.

        Parameters
        ----------
        xmin : float, optional
            Override for lower wavelength bound.
        xmax : float, optional
            Override for upper wavelength bound.
        n_broad : int, optional
            Override for broad-line count.
        n_narrow : int, optional
            Override for narrow-line count.
        fe_mode : str, optional
            Override for FeII mode.
        continuum_profile : str, optional
            Override for continuum profile.
        group_method : bool, optional
            Override for grouping behavior.
        add_outflow : bool, optional
            Override for outflow inclusion.
        add_winds : bool, optional
            Override for wind inclusion.
        add_balmer_continuum : bool, optional
            Override for Balmer continuum.
        add_uncommon_narrow : bool, optional
            Override for uncommon narrow-line inclusion.
        add_host_miles : bool or dict, optional
            Override for host-galaxy options.
        """
        def get(val, fallback):
            return val if val is not None else fallback
        
        xmin = get(xmin, self.xmin)
        xmax = get(xmax, self.xmax)
        n_broad = get(n_broad, self.n_broad)
        n_narrow = get(n_narrow, self.n_narrow)
        fe_mode = get(fe_mode, self.fe_mode).lower()
        add_outflow = get(add_outflow, self.add_outflow)
        add_balmer_continuum = get(add_balmer_continuum, self.add_balmer_continuum)
        add_winds = get(add_winds, self.add_winds)
        add_uncommon_narrow = get(add_uncommon_narrow,self.add_uncommon_narrow)
        add_host_miles = get(add_host_miles,self.add_host_miles)
        add_BAL = get(add_BAL,self.add_BAL)
        add_balmerhighorder_continuum = get(add_balmerhighorder_continuum,self.add_balmerhighorder_continuum)
        continuum_profile = get(continuum_profile, self.continuum_profile).lower()
        tied_broad_to = get(tied_broad_to,self.tied_broad_to)
        tied_narrow_to = get(tied_narrow_to,self.tied_narrow_to)
        if add_BAL:
            warnings.warn("The addition of BALs to the fit is still in development not well tested yet.")
            
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
                    comps = self._handle_broad_and_narrow_lines(base, n_narrow, n_broad,add_winds=add_winds,add_BAL=add_BAL,add_outflow=add_outflow)
                elif pseudo_region_name == "narrows" and n_narrow>0:
                    comps = self._handle_narrow_line(base, n_narrow,add_outflow=add_outflow,add_uncommon_narrow=add_uncommon_narrow)
                elif pseudo_region_name == "broads" and n_broad>0:
                    comps = self._handle_broad_line(base, n_broad,add_winds=add_winds,add_BAL=add_BAL) 
                self.complex_list.extend(comps)        
        if add_host_miles:
            self._handle_host(add_host_miles,xmin,xmax)
        self.complex_list.extend(self._handle_fe(fe_mode,xmin,xmax))
        self.complex_list.extend(self._continuum_handle(continuum_profile,xmin,xmax,add_balmer_continuum=add_balmer_continuum,add_balmerhighorder_continuum = add_balmerhighorder_continuum))#here we already are able to create the complex_class
        self.complex_class = ComplexRegion(self.complex_list)
        self._ties = []
        self._known_ties = []
        self._feties = []
        if self.group_method:
             self.complex_class = self._apply_group_method(self.complex_class,fe_mode,self.known_tied_relations)
        else:
            #todo add the tied_broad_to and narrow_to in cases in where is best use a line selected for the user
            self._ties,self._known_ties =_maketies(self.complex_class,tied_narrow_to = tied_narrow_to, tied_broad_to = tied_broad_to,known_tied_relations=self.known_tied_relations)
            #self.tied_relations.extend([*_ties,*_known_ties])
            #self._ties = []
            if fe_mode not in ["none","template"]:
                routine_fe_tied = {"by":"subregion","tied_params": ('center', 'fwhm')}
                self._feties = fe_ties(self.complex_class.group_by("region").get("fe").lines, routine_fe_tied)
                #self.tied_relations.extend(fe_ties(self.complex_class.group_by("region").get("fe").lines, routine_fe_tied))
        del self.complex_list
        
    def _handle_broad_and_narrow_lines(
        self, entry: SpectralLine, n_narrow: int, n_broad: int, add_winds=False,add_BAL = False ,add_outflow=False) -> List[SpectralLine]:
        """
        Create narrow, broad, and optional wind components for a single line.

        Parameters
        ----------
        entry : SpectralLine
            Base line definition.
        n_narrow : int
            Number of narrow components.
        n_broad : int
            Number of broad components.
        add_winds : bool, optional
            Whether to append a wind component to first broad line.

        Returns
        -------
        list of SpectralLine
            Generated line components.
        """
        comps: List[SpectralLine] = []
        total = n_narrow + n_broad
        for idx in range(total):
            region = 'narrow' if idx < n_narrow else 'broad'
            comp_num = idx + 1 if region == 'narrow' else idx - n_narrow + 1
            amp = 1.0 #if comp_num == 1 else 1.0/comp_num
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                region=region,
                component=comp_num,
                amplitude=amp,
                element=entry.element,
            )
            comps.append(new)
            #self.n_max_component_outflow
            if add_outflow and comp_num <= self.n_max_component_outflow and new.line_name in self.lines_prone_outflow:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='outflow',
                    component = comp_num,
                    amplitude=1.0,
                    element = entry.element,
                    rarity = entry.rarity)
                
                comps.append(out)
            elif add_winds and comp_num <= self.n_max_component_winds  and new.line_name in self.lines_prone_winds:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='winds',
                    component = comp_num,
                    amplitude=1.0,
                    element = entry.element,
                )
                comps.append(out)
            elif add_BAL and comp_num <= self.n_max_component_bal and new.line_name in self.lines_prone_bal:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='bal',
                    component = comp_num,
                    amplitude=1.0,
                    element = entry.element,
                )
                comps.append(out)
        return comps
    
    def _handle_narrow_line(
        self, entry: SpectralLine, n_narrow: int, add_outflow: bool = False, add_uncommon_narrow = False) -> List[SpectralLine]:
        """
        Create narrow and optional outflow components for a single line.

        Parameters
        ----------
        entry : SpectralLine
            Base line definition.
        n_narrow : int
            Number of narrow components.
        add_outflow : bool, optional
            Include an outflow component for the first narrow line.
        add_uncommon_narrow : bool, optional
            Include lines marked as 'uncommon'.

        Returns
        -------
        list of SpectralLine
            Generated narrow (and outflow) components.
        """
        comps: List[SpectralLine] = []
        for idx in range(n_narrow):
            amp = 1.0
            if entry.rarity=="uncommon" and not add_uncommon_narrow:
                continue 
            comp_num = idx + 1
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                region ='narrow',
                component = comp_num,
                amplitude =  amp,
                element = entry.element,
                rarity = entry.rarity
            )
            comps.append(new)
            if add_outflow and comp_num <= self.n_max_component_outflow and new.line_name in self.lines_prone_outflow:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='outflow',
                    component = comp_num,
                    amplitude=1.0,
                    element = entry.element,
                    rarity = entry.rarity)
                
                comps.append(out)
        return comps

    def _handle_broad_line(self, entry: SpectralLine, n_broad: int,add_winds=False,add_BAL=False) -> List[SpectralLine]:
        """
        Create broad and optional wind components for a single line.
        
        Notes
        -----
        Only for the first broad component is possible add the extra-broad lines.

        Parameters
        ----------
        entry : SpectralLine
            Base line definition.
        n_broad : int
            Number of broad components.
        add_winds : bool, optional
            Include a wind component for the first broad line.

        Returns
        -------
        list of SpectralLine
            Generated broad (and wind) components.
        """
        #extra broad? 
        #return comps
        comps: List[SpectralLine] = []
        for idx in range(n_broad):
            if idx>0:
                continue 
            amp = 1 #if idx == 0 else 0.5
            comp_num = idx + 1
            new = SpectralLine(
                center=entry.center,
                line_name=entry.line_name,
                region='broad',
                component=comp_num,
                amplitude=amp,
                element=entry.element,
            )
            comps.append(new)
            
            if add_winds and comp_num <= self.n_max_component_winds and self.lines_prone_winds:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='winds',
                    component = comp_num,
                    amplitude=1.0,
                    element = entry.element,
                )
                comps.append(out)
            elif add_BAL and comp_num <= self.n_max_component_bal and new.line_name in self.lines_prone_bal:
                out = SpectralLine(
                    center= entry.center,
                    line_name=entry.line_name,
                    region ='bal',
                    component = comp_num,
                    amplitude=1.0,
                    element = entry.element,
                )
                comps.append(out)
                
        return comps
 
    def _handle_fe(self,fe_mode,xmin,xmax):
        """
        Generate FeII components based on selected mode and wavelength range.

        Parameters
        ----------
        fe_mode : {'template', 'model', 'none'}
            FeII handling mode.
        xmin : float
            Lower bound of region.
        xmax : float
            Upper bound of region.

        Returns
        -------
        list of SpectralLine
            FeII line components (empty if mode 'none').

        Notes
        -----
        - 'template' adds broad template components if range is sufficient.
        - 'model' loads individual FeII lines from YAML.
        """
        fe_comps = []
        if fe_mode == "none":
            return fe_comps
        elif fe_mode == "template":
            print("Added feuvop template")
            fe_comps.extend([SpectralLine(line_name="feuvop",region="fe",component=1,profile="fetemplate",template_info = {"name":"feuvop","x_min":xmin,"x_max":xmax})])
            # t_c = 0
            # if max(0, min(xmax, 7484) - max(xmin, 3686)) >= 1000:
            #     if self.verbose:
            #         print("added OP template")
            #     fe_comps.extend(
            #         [SpectralLine(line_name="feop",region="fe",component=1,profile="fetemplate",template_info = {"name":"feop"})])
            #     t_c += 1
            # if max(0, min(xmax, 3500) - max(xmin, 1200)) >= 500:
            #     #maybe it is a good time to r
            #     if self.verbose:
            #         print("added UV template")
            #     fe_comps.extend([SpectralLine(line_name="feuv",region="fe",component=1,profile="fetemplate",template_info = {"name":"feuv"})])
            #     t_c += 1
            # if t_c == 0:
            #     print("The covered range is not valid for template use. Switching to model mode. Work in progress, if no Fe wanted put fe_mode = none.")#this have to be a warning
            #     fe_mode = "model"
        elif fe_mode == "model":      
            for pseudo_region_name,list_dict in self.lines_available.items():
                for raw_line in list_dict:
                    center = float(raw_line.get('center', -np.inf))
                    if not (xmin <= center <= xmax) or pseudo_region_name not in ('feii_uv',"feii_model"):
                        continue
                    base = SpectralLine(**raw_line)
                    base.subregion = pseudo_region_name
                    base.component = 1
                    fe_comps.extend([base])
        return fe_comps
    
    def _continuum_handle(self,continuum_profile,xmin,xmax,add_balmer_continuum=False,add_balmerhighorder_continuum=False):
        """
        Create continuum components: linear, powerlaw, or Balmer.

        Parameters
        ----------
        continuum_profile : str
            Continuum profile to use.
        xmin : float
            Lower wavelength bound.
        xmax : float
            Upper wavelength bound.
        add_balmer_continuum : bool, optional
            Include a Balmer continuum component if True.

        Returns
        -------
        list of SpectralLine
            Continuum components.
        """
        continuum_comps = []
        if add_balmer_continuum and not xmax< 3646:
            continuum_comps.append(SpectralLine(line_name='balmercontinuum',region='continuum',component=0,profile='balmercontinuum'))
        if add_balmerhighorder_continuum and not xmax< 3646:
            #fe_comps.extend([SpectralLine(line_name="feuvop",region="fe",component=1,profile="fetemplate",template_info = {"name":"feuvop","x_min":xmin,"x_max":xmax})])
            # t_c = 0
            continuum_comps.append(SpectralLine(line_name='balmerhighorder',region='continuum',component=0,profile='fetemplate',template_info = {"name":"BalHiOrd","x_min":xmin,"x_max":xmax}))
        
        
        if 'linear' != continuum_profile and (xmax - xmin) < self.LINEAR_RANGE_THRESHOLD:
            print(f"xmax - xmin less than LINEAR_RANGE_THRESHOLD:{self.LINEAR_RANGE_THRESHOLD} < {(xmax - xmin)} moving to linear continuum")
            continuum_comps.append(SpectralLine(line_name="linear",region='continuum',component=0,profile="linear"))
            return continuum_comps
        continuum_comps.append(SpectralLine(line_name=continuum_profile,region='continuum',component=0,profile=continuum_profile))
        return continuum_comps

    def _apply_group_method(self,complex_class,fe_mode,known_tied_relations):
        """
        Group lines by region, apply known ties, and return a new ComplexRegion.

        -This function in particular could be useful to run it outside.
        Parameters
        ----------
        complex_class : ComplexRegion
            Ungrouped region object.
        fe_mode : str
            FeII mode for grouping logic.
        known_tied_relations : list of tuple
            Predefined tie relations for line ratios and centers.

        Returns
        -------
        ComplexRegion
            Grouped and tied region.
        """
        dict_regions = complex_class.group_by("region")
        new_complex_list = []
        for key,values in dict_regions.items():
            if key in ["continuum","host"]:
                new_complex_list.extend(values.lines)
            elif key == "fe":
                #here much more can be done 
                if fe_mode=="model":
                    new_complex_list.extend(group_lines(values.lines,key,mode="element",profile="SPAF"))
                else:
                    new_complex_list.extend(values.lines)
            elif key in ["outflow","winds"]:
                new_complex_list.extend(group_lines(values.lines,key,mode="element",known_tied_relations=known_tied_relations,profile="SPAF"))
            else:
                new_complex_list.extend(group_lines(values.lines,key,mode="region",known_tied_relations=known_tied_relations,profile="SPAF"))
        return ComplexRegion(new_complex_list)

    def _handle_host(self,add_host_miles,xmin,xmax):
        """
        Add a host-galaxy template component using MILES SSP models.

        Parameters
        ----------
        add_host_miles : dict or bool
            Configuration for host model (kwargs for `make_host_function` or True for defaults).
        xmin : float
            Lower wavelength bound.
        xmax : float
            Upper wavelength bound.

        Side Effects
        ------------
        Appends a `SpectralLine` of region 'host' to `self.complex_list`.
        """
        pos_defaults = make_host_function.__defaults__ or ()  
        #kw_defaults = make_host_function.__kwdefaults__ or {}  
        argcount = make_host_function.__code__.co_argcount
        all_args = make_host_function.__code__.co_varnames[:argcount]
        names_for_pos = all_args[-len(pos_defaults):] if pos_defaults else []
        defaults = {name: val for name, val in zip(names_for_pos, pos_defaults)}
        defaults["x_max"] = xmax
        defaults["x_min"] = xmin
        if isinstance(add_host_miles,bool):
            _host_model = make_host_function(**defaults)
        elif isinstance(add_host_miles,Dict):
            add_host_miles.update({"xmax":xmax,"xmin":xmax})
            _host_model = make_host_function(**add_host_miles)
        else:
            Warning("Not accepted type of add_host_moles")
            return self.complex_list
        line = SpectralLine(line_name="host",region="host",component=1,template_info=_host_model["host_info"],profile="hostmiles")    
        self.complex_list.extend([line])
        
    def _make_fitting_routine(self,list_num_steps = [1000],list_learning_rate = [1e-2]):
        """
        Assemble the fitting routine dictionary for `RegionFitting.from_builder`.
        TODO improve the loop. 
        Parameters
        ----------
        list_num_steps : list of int, optional
            Number of optimization steps for each stage.
        list_learning_rate : list of float, optional
            Learning rates for each stage.

        Returns
        -------
        dict
            Dictionary with keys 'complex_class', 'outer_limits', 'inner_limits',
            and 'fitting_routine' ready for passing to RegionFitting.

        Raises
        ------
        AssertionError
            If lengths of `list_num_steps` and `list_learning_rate` differ.
        """
        fitting_routine = {}
        tied_relations = [self._known_ties]
        if len(self._ties) > 0:
            list_num_steps.extend([1000])
            list_learning_rate.extend([1e-2])
            tied_relations = [[*self._ties,*self._known_ties,*self._feties],[*self._known_ties]]
        assert len(list_num_steps) == len(list_learning_rate), "len(list_num_steps) != len(list_learning_rate) "
        if not  self.group_method:
            for i, steps in enumerate(list_num_steps):
                if i>0:
                    fitting_routine[f"step{i+1}"] = {"tied": tied_relations[1],"non_optimize_in_axis": 4,"learning_rate": list_learning_rate[i],"num_steps": list_num_steps[i]}
                else:
                    fitting_routine[f"step{i+1}"] = {"tied": tied_relations[i],"non_optimize_in_axis": 4,"learning_rate": list_learning_rate[i],"num_steps": list_num_steps[i]}
                #fitting_routine[f"step{steps}"] = {"tied": [],"non_optimize_in_axis": 4,"learning_rate":list_learning_rate[i-1],"num_steps": list_num_steps[i-1]}
        else:
            for i, steps in enumerate(list_num_steps):
                if i>0:
                    fitting_routine[f"step{i+1}"] = {"tied": [],"non_optimize_in_axis": 4,"learning_rate": list_learning_rate[i],"num_steps": list_num_steps[i]}
                else:
                    fitting_routine[f"step{i+1}"] = {"tied": [],"non_optimize_in_axis": 4,"learning_rate": list_learning_rate[i],"num_steps": list_num_steps[i]}
            
        return {"complex_class": self.complex_class,"outer_limits": [self.xmin, self.xmax], "inner_limits": [self.xmin + 50, self.xmax - 50],"fitting_routine":fitting_routine}

    def _load_lines(self, paths: Optional[List[Union[str, Path]]]) -> None:
            """
            Load YAML files defining spectral lines into `lines_available`.

            Parameters
            ----------
            paths : list of str or Path, optional
                Paths to YAML templates.

            Raises
            ------
            ValueError
                If no paths are provided.
            FileNotFoundError
                If any path does not point to an existing file.
            KeyError
                If YAML content is not a list of dicts per file.
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
