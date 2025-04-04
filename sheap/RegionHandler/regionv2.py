
#import profile
from SHEAP.fitting.functions import GaussianSum,linear,loglinear,gaussian_func,lorentzian_func
import os 
from SHEAP.fitting.utils import combine_auto
from SHEAP.tools.others import  kms_to_wl
import jax.numpy as jnp
from jax import jit 
from typing import Union
import yaml
from SHEAP.utils import mask_builder
from SHEAP.fitting.main_fitting_I import MasterMinimizer
from .fit_fe import fitFeOP,fitFeUV
region_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"regions")


region_available = {"Halpha": os.path.join(region_path,"Halpha.yaml"),"Hbeta": os.path.join(region_path,"Hbeta.yaml"), 
                    "CIII": os.path.join(region_path,"CIII.yaml"),"CIV": os.path.join(region_path,"CIV.yaml"),
                    "Lyalpha": os.path.join(region_path,"Lyalpha.yaml"),"MgII": os.path.join(region_path,"MgII.yaml")}


class ComplexRegion:
    """
    Class to represent and process a complex spectral region for fitting emission lines.
    """
    #TODO dinting maybe the outflows of the semi broad or narrow broad or similar
    def __init__(self, dict_region: [str, dict],limits_list=None,broad_upper_width=5000.0,broad_lower_width=425,narrow_upper_width=200.0,\
                narrow_lower_width=50.,narrow_broad_lower_width=1000.,narrow_broad_upper_width=300,broad_center_shift_limit=5000.0,
                narrow_center_shift_limit=2500.,narrow_broad_center_shift_limit=2000.,
                log_mode=False,tied_params=None):
        """
        #10000/2.355 4246.284501061571 gamma_lorentian = 1.1775sigma
        Parameters:
            dict_region (str or dict): List of dictionaries, each containing the parameters for a spectral line.
            x_min, x_max: Not used in this snippet, but could be used to further restrict the fitting region.
            tied_params (list, optional): List defining parameter ties between lines.
            broad_upper_width  # Maximum width for broad emission lines
            broad_lower_width  # Minimum width for broad emission lines (sigma) def 1000>FWHM
            narrow_lower_width   # Minimum width for narrow emission lines (sigma)
            narrow_broad_lower_width  # Minimum width for broad emission lines (sigma)
            narrow_broad_upper_width  # Minimum width for broad emission lines (sigma)
            broad_center_shift_limit
            narrow_center_shift_limit
            narrow_broad_center_shift_limit
        """
        if isinstance(dict_region,str):
            if dict_region in region_available.keys():
                with open(region_available[dict_region], 'r') as file:
                    self.dict_region = yaml.safe_load(file)
            else:
                raise Exception("Your region is not in the list available try with",region_available.keys())
        elif isinstance(dict_region,dict):
            self.dict_region = dict_region
        elif isinstance(dict_region,list):
            self.dict_region = {"region":dict_region}
            
        self.tied_params_sequence = []
        for keys in self.dict_region:
            if "tied_params" in keys:
                self.tied_params_sequence.append(keys)
        if len(self.tied_params_sequence) == 0:
            self.tied_params_sequence.append("tied_params_step_1")
            self.dict_region["tied_params_step_1"] = tied_params if tied_params is not None else []
        
        self.region = self.dict_region.get("region")
        self.tied_params = self.dict_region.get(self.tied_params_sequence[0])
        #self.tied_params = tied_params if tied_params is not None else []
        self.limits_list = limits_list if limits_list is not None else []  # You can also pass a limits_list if needed
        
        
        self.log_mode = log_mode
        
        # Width limits in km/s
        self.broad_upper_width = broad_upper_width  # Maximum width for broad emission lines
        self.broad_lower_width = broad_lower_width  # Minimum width for broad emission lines (sigma)
        self.narrow_upper_width = narrow_upper_width  # Maximum width for narrow emission lines (sigma)
        self.narrow_lower_width = narrow_lower_width   # Minimum width for narrow emission lines (sigma)
        self.narrow_broad_lower_width = narrow_broad_lower_width
        self.narrow_broad_upper_width = narrow_broad_upper_width

        # Center shift limits in km/s
        self.broad_center_shift_limit = broad_center_shift_limit
        self.narrow_center_shift_limit = narrow_center_shift_limit
        self.narrow_broad_center_shift_limit = narrow_broad_center_shift_limit

        # Build the firts fitting region and function mostly to check 
        self.create_region_to_be_fit()
        
    #iguess this could be in his own place outside
    def constrains(self, center: float, kind: str, amplitude, **kwargs):
        """
        Compute constraint parameters for a given emission line.
        #TODO this is a very gaussian way if in some point this should be upgrade.
        Parameters:
            center (float): Central wavelength of the line.
            kind (str): Type of line ('broad', 'narrow', or other for mixed).
            amplitude: Amplitude of the emission line.
            as_array (bool): If True, return a list; otherwise, return a dictionary.
            **kwargs: Additional keyword arguments (if needed).
        
        Returns:
            list or dict: The constraint parameters.
        """
        broad_center_shift_limit = kwargs.get("broad_center_shift_limit",self.broad_center_shift_limit)
        broad_upper_width = kwargs.get("broad_upper_width",self.broad_upper_width)
        broad_lower_width = kwargs.get("broad_lower_width",self.broad_lower_width)
        narrow_center_shift_limit = kwargs.get("narrow_center_shift_limit",self.narrow_center_shift_limit)
        narrow_upper_width = kwargs.get("narrow_upper_width",self.narrow_upper_width)
        narrow_lower_width = kwargs.get("narrow_lower_width",self.narrow_lower_width)
        narrow_broad_center_shift_limit = kwargs.get("narrow_broad_center_shift_limit",self.narrow_broad_center_shift_limit)
        narrow_broad_lower_width = kwargs.get("narrow_broad_lower_width",self.narrow_broad_lower_width)
        narrow_broad_upper_width = kwargs.get("narrow_broad_upper_width",self.narrow_broad_upper_width)
        profile =  kwargs.get("profile")       
        #init_,lower_,upper_ = [],[],[]
        #given we know at this point the profile we just can jouse the parameters in the process. before the end 
        if kind.lower() == "broad":
            center_upper = center + kms_to_wl(broad_center_shift_limit, center)
            center_lower = center - kms_to_wl(broad_center_shift_limit, center)
            amplitude_upper, amplitude_lower =  10,0.0
            width_upper = kms_to_wl(broad_upper_width, center)
            width_lower = kms_to_wl(broad_lower_width, center)
            width = width_lower
            
            gamma_upper = 1.1775*width_upper
            gamma_lower = 1.1775*width_lower
            gamma = gamma_lower
        elif kind.lower() == "narrow":
            center_upper = center + kms_to_wl(narrow_center_shift_limit, center)
            center_lower = center - kms_to_wl(narrow_center_shift_limit, center)
            width_upper = kms_to_wl(narrow_upper_width, center)
            width_lower = kms_to_wl(narrow_lower_width, center)
            width = width_upper
            amplitude_upper, amplitude_lower =  10,0.0
            gamma_upper = 1.1775*width_upper
            gamma_lower = 1.1775*width_lower
            gamma = 2*gamma_lower
        elif kind.lower() == "nlr":
            center_upper = center + kms_to_wl(narrow_center_shift_limit, center)
            center_lower = center - kms_to_wl(narrow_center_shift_limit, center)
            width_upper = kms_to_wl(narrow_upper_width, center)
            width_lower = kms_to_wl(narrow_lower_width, center)
            width =  width_lower
            amplitude_upper, amplitude_lower =  10,0.0
            gamma_upper = 1.1775*width_upper
            gamma_lower = 1.1775*width_lower
            gamma = gamma_lower/2
        else:
            print(kind.lower(), "this kind of line is not added yet we will assume initial and limits parameter of narrow line")
            center_upper = center + kms_to_wl(narrow_broad_center_shift_limit, center)
            center_lower = center - kms_to_wl(narrow_broad_center_shift_limit, center)
            width_upper = kms_to_wl(narrow_broad_upper_width, center)
            width_lower = kms_to_wl(narrow_broad_lower_width, center)
            width = width_lower
            amplitude_upper, amplitude_lower = 10,0.0 
            gamma_upper = 1.1775*width_upper
            gamma_lower = 1.1775*width_lower
            gamma = gamma_lower
       # if self.log_mode:
        #    #print(center,center_upper,center_lower,width_upper,width_lower,width )
         #   center,center_upper,center_lower,width_upper,width_lower,width = map(lambda x: 1e-3 * x, [center, center_upper, center_lower, width_upper, width_lower, width])
          #  #print(center,center_upper,center_lower,width_upper,width_lower,width )
        init_ = [amplitude, center]
        upper_ = [amplitude_upper,center_upper]
        lower_ = [amplitude_lower, center_lower]
        if profile=="lorentzian":
            init_.append(gamma)
            upper_.append(gamma_upper)
            lower_.append(gamma_lower)
            params = ["amplitude","center","width"]
        elif profile=="guassian":
            init_.append(width)
            upper_.append(width_upper)
            lower_.append(width_lower)
            params = ["amplitude","center","width"]
        elif profile =="voigt":
            print("wtf")
        return init_,upper_,lower_,profile,params
    
    def free_constrains_list(self):
        self.constrains_list = []
    
    
    
    def created_fit_routine(self,**kwargs):
        """
        Build the region constraints and ties for the spectral fitting.
        #TODO what to do when the user add the multicomponent option maybe given it already did it maybe they also can define itselft is multi index option?
        Parameters:
            as_array (bool): Whether to store the region constraints as a JAX array.
            tied_params (list, optional): Overrides the instance tied_params if provided.
            tie param_target to param_source
            [param_target, param_source,operand,value]
            limits_list (list, optional): Overrides the instance limits_list if provided.
        """
        #import numpy as np 
        tied_params = kwargs.get("tied_params",self.tied_params)
        limits_list = kwargs.get("limits_list",self.limits_list)
        list_dependencies = []
        initial_params,upper_bound,lower_bound,profile_list,profile_index_list,profile_function_list,lines_,multi_comp = [],[],[],[],[],[],[],set()
        params_dict ={}
        C = 0  
        fit_Fe = [] 
        for _,line in enumerate(self.region):
            if "Fe" in line["line_name"]:
                fit_Fe.append(line)
                continue
            l_ = []
            l_.append(C)
            if "profile" not in line.keys():
                line["profile"] = "guassian"
            if "amplitude" not in line.keys():
                line["amplitude"] = 1.0
            line.update(kwargs)
            _,kind,line_name = (line[i]  for i in ["center","kind","line_name"])
            initial_params_,upper_bound_,lower_bound_,profile,params = self.constrains(**line)
            if profile=="guassian":
                profile_function_list.append(gaussian_func)
            elif profile=="lorentzian":
                profile_function_list.append(lorentzian_func)
            initial_params += initial_params_
            upper_bound +=  upper_bound_
            lower_bound += lower_bound_
            profile_list.append(profile)
            line_ = f"{line_name}_{kind}"
            if line_ in lines_:
                multi_comp.add(line_)
                count = sum([line_ in (i) for i in lines_])+1
                line_ +=f"_{count}"
                multi_comp.add(line_)
            params_dict.update({f"{p}_{line_}":n+C for n,p in enumerate(params)})
            C += len(params)
            l_.append(C)
            profile_index_list.append(l_)
            lines_.append(line_)
        
        if "cont" not in params_dict.keys():
            print("We assume a local linear continuum")
            profile_function_list.append(linear)
            self.linear_func = linear
            initial_params += [0.01,0.1]
            upper_bound += [10,10]
            lower_bound += [-10,-10]
            profile_list.append("linear")
            profile_index_list.append([C,C+2])
            params_dict.update({f"{p}_{"cont"}":n+C for n,p in enumerate(["m","b"])})
            C += 2
            
        if len(fit_Fe)>0:
            for i in fit_Fe:
                #how,line = i
                if i["how"]=="template":
                    if i["which"]=="OP":
                        profile_function_list.append(fitFeOP)
                        initial_params += [5., 0, 1.]
                        upper_bound += [8.3010300e+00,10,10]
                        lower_bound += [3.0000000e+00,-10,0]
                        profile_list.append("Fe_OP")
                        profile_index_list.append([C,C+3])
                        params_dict.update({f"{p}_{"Fe_OP"}":n+C for n,p in enumerate(["log_FWHM_broad","shift","scale"])})
                    elif i["which"]=="OP":
                        profile_function_list.append(fitFeUV)
                        initial_params += [5., 0, 1.]
                        upper_bound += [8.3010300e+00,10,10]
                        lower_bound += [3.0000000e+00,-10,0]
                        profile_list.append("Fe_UV")
                        profile_index_list.append([C,C+3])
                        params_dict.update({f"{p}_{"Fe_UV"}":n+C for n,p in enumerate(["log_FWHM_broad","shift","scale"])})
                    else:
                        print("Currently we only support OP and UV template other options are not allowed")
        
        
        
        upper_bound = jnp.array(upper_bound)
        lower_bound = jnp.array(lower_bound)
        initial_params = jnp.array(initial_params)
        
        
        self.profile_index_list = profile_index_list
        self.profile_list = profile_list
        
        def get_param_coord_value(param, line_name, kind):
            pos_ = params_dict.get(f"{param}_{line_name}_{kind}")
            return pos_,initial_params[pos_],param
        
        if len(tied_params)>0:
            for tied in tied_params:
                param1, param2 = tied[:2]
                pos_param1, val_param1,param_1 = get_param_coord_value(*param1.split("_"))
                pos_param2, val_param2,param_2 = get_param_coord_value(*param2.split("_"))
                if  len(tied)==2:
                    if param_1==param_2=="center" and len(tied):
                        delta = val_param1 - val_param2
                        tied_val = "+" + str(delta) if delta>0 else "-" + str(abs(delta))
                        #if log_mode:        
                    elif param_1==param_2:
                        tied_val = "*1"
                    else:
                        print(f"Define constraints properly. {tied_params}")
                else:
                    tied_val = tied[-1]
                
                if isinstance(tied_val, str):
                    list_dependencies.append(f"{pos_param1} {pos_param2} {tied_val}")
                else:
                    print("Define constraints properly.")
        else:
            list_dependencies = []
        self.list_dependencies = list_dependencies
        self.params_dict  = params_dict
        self.constraints = jnp.stack([lower_bound,upper_bound]).T  #this also can came from the class maybe is the user wants to edit them also is possible 
        self.initial_params = initial_params
        self.profile_function_list = profile_function_list
        self.profile_function_combine = jit(combine_auto([*profile_function_list])) 
        self.multi_comp = multi_comp
        self.lines = lines_
        
                
    def create_region_to_be_fit(self,**kwargs):
        """
        Build the region constraints and ties for the spectral fitting.
        #TODO what to do when the user add the multicomponent option maybe given it already did it maybe they also can define itselft is multi index option?
        Parameters:
            as_array (bool): Whether to store the region constraints as a JAX array.
            tied_params (list, optional): Overrides the instance tied_params if provided.
            tie param_target to param_source
            [param_target, param_source,operand,value]
            limits_list (list, optional): Overrides the instance limits_list if provided.
        """
        #import numpy as np 
        tied_params = kwargs.get("tied_params",self.tied_params)
        limits_list = kwargs.get("limits_list",self.limits_list)
        list_dependencies = []
        initial_params,upper_bound,lower_bound,profile_list,profile_index_list,profile_function_list,lines_,multi_comp = [],[],[],[],[],[],[],set()
        params_dict ={}
        C = 0  
        fit_Fe = [] 
        for _,line in enumerate(self.region):
            if "Fe" in line["line_name"]:
                fit_Fe.append(line)
                continue
            l_ = []
            l_.append(C)
            if "profile" not in line.keys():
                line["profile"] = "guassian"
            if "amplitude" not in line.keys():
                line["amplitude"] = 1.0
            line.update(kwargs)
            _,kind,line_name = (line[i]  for i in ["center","kind","line_name"])
            initial_params_,upper_bound_,lower_bound_,profile,params = self.constrains(**line)
            if profile=="guassian":
                profile_function_list.append(gaussian_func)
            elif profile=="lorentzian":
                profile_function_list.append(lorentzian_func)
            initial_params += initial_params_
            upper_bound +=  upper_bound_
            lower_bound += lower_bound_
            profile_list.append(profile)
            line_ = f"{line_name}_{kind}"
            if line_ in lines_:
                multi_comp.add(line_)
                count = sum([line_ in (i) for i in lines_])+1
                line_ +=f"_{count}"
                multi_comp.add(line_)
            params_dict.update({f"{p}_{line_}":n+C for n,p in enumerate(params)})
            C += len(params)
            l_.append(C)
            profile_index_list.append(l_)
            lines_.append(line_)
        
        if "cont" not in params_dict.keys():
            print("We assume a local linear continuum")
            profile_function_list.append(linear)
            self.linear_func = linear
            initial_params += [0.01,0.1]
            upper_bound += [10,10]
            lower_bound += [-10,-10]
            profile_list.append("linear")
            profile_index_list.append([C,C+2])
            params_dict.update({f"{p}_{"cont"}":n+C for n,p in enumerate(["m","b"])})
            C += 2
            
        if len(fit_Fe)>0:
            for i in fit_Fe:
                #how,line = i
                if i["how"]=="template":
                    if i["which"]=="OP":
                        profile_function_list.append(fitFeOP)
                        initial_params += [5., 0, 1.]
                        upper_bound += [8.3010300e+00,10,10]
                        lower_bound += [3.0000000e+00,-10,0]
                        profile_list.append("Fe_OP")
                        profile_index_list.append([C,C+3])
                        params_dict.update({f"{p}_{"Fe_OP"}":n+C for n,p in enumerate(["log_FWHM_broad","shift","scale"])})
                    elif i["which"]=="OP":
                        profile_function_list.append(fitFeUV)
                        initial_params += [5., 0, 1.]
                        upper_bound += [8.3010300e+00,10,10]
                        lower_bound += [3.0000000e+00,-10,0]
                        profile_list.append("Fe_UV")
                        profile_index_list.append([C,C+3])
                        params_dict.update({f"{p}_{"Fe_UV"}":n+C for n,p in enumerate(["log_FWHM_broad","shift","scale"])})
                    else:
                        print("Currently we only support OP and UV template other options are not allowed")
        
        upper_bound = jnp.array(upper_bound)
        lower_bound = jnp.array(lower_bound)
        initial_params = jnp.array(initial_params)
        
        self.profile_index_list = profile_index_list
        self.profile_list = profile_list
        def get_param_coord_value(param, line_name, kind):
            pos_ = params_dict.get(f"{param}_{line_name}_{kind}")
            return pos_,initial_params[pos_],param
        
        if len(tied_params)>0:
            for tied in tied_params:
                param1, param2 = tied[:2]
                pos_param1, val_param1,param_1 = get_param_coord_value(*param1.split("_"))
                pos_param2, val_param2,param_2 = get_param_coord_value(*param2.split("_"))
                if  len(tied)==2:
                    if param_1==param_2=="center" and len(tied):
                        delta = val_param1 - val_param2
                        tied_val = "+" + str(delta) if delta>0 else "-" + str(abs(delta))
                        #if log_mode:        
                    elif param_1==param_2:
                        tied_val = "*1"
                    else:
                        print(f"Define constraints properly. {tied_params}")
                else:
                    tied_val = tied[-1]
                
                if isinstance(tied_val, str):
                    list_dependencies.append(f"{pos_param1} {pos_param2} {tied_val}")
                else:
                    print("Define constraints properly.")
        else:
            list_dependencies = []
        self.list_dependencies = list_dependencies
        self.params_dict  = params_dict
        self.constraints = jnp.stack([lower_bound,upper_bound]).T  #this also can came from the class maybe is the user wants to edit them also is possible 
        self.initial_params = initial_params
        self.profile_function_list = profile_function_list
        self.profile_function_combine = jit(combine_auto([*profile_function_list])) 
        self.multi_comp = multi_comp
        self.lines = lines_
    
    def mapping_params(self,params):
        """
        params is a str or list
        [["width","broad"],"cont"]
        """
        if isinstance(params,str):
            params = [params]
        match_list = []
        for param in params:
            if isinstance(param,str):
                param = [param]
            #print(self.params_dict.keys())
            #print([[self.params_dict[key],key] for key in self.params_dict.keys() if all([p in key for p in param])])
            
            match_list += ([self.params_dict[key] for key in self.params_dict.keys() if all([p in key for p in param])])
        match_list = jnp.array(match_list)
        unique_arr =jnp.unique(match_list)
        return unique_arr
    
    #TODO save routine and read routine 
    
    def __call__(self,spectra,inner_limits=None,outer_limits=None,**kwargs):
        """_summary_
            main fitting sequence 
        """
        inner_limits = self.dict_region.get("inner_limits",inner_limits)
        outer_limits = self.dict_region.get("outer_limits",outer_limits)
        constraints =  kwargs.get("constraints",self.constraints)
        #print(inner_limits)
        if inner_limits is None or outer_limits is None:
            Exception("Check the limits")
        num_steps = int(kwargs.get("num_steps",1000))
        region_to_fit_norm, _,_,mask_region = mask_builder(spectra,outer_limits=outer_limits)
        max_value = jnp.nanmax(jnp.where(mask_region,0, spectra[:, 1, :]),axis=1)
        region_to_fit_norm = region_to_fit_norm.at[:,[1,2],:].divide(jnp.moveaxis(jnp.tile(max_value,(2,1)),0,1)[:,:,None])
        
        #uncertainties_region_to_fit_norm = uncertainties_region_to_fit_norm.at[:].divide(max_value[:,None])
        #median_region = jnp.nanmedian(jnp.where(mask_fit_cont,jnp.nan,region_to_fit_norm[:,1,:]),axis=1)
        Master_region = MasterMinimizer(self.profile_function_combine, non_optimize_in_axis=3,num_steps=num_steps,list_dependencies=self.list_dependencies)
        if len(self.tied_params_sequence)>0:
            print("Runing:",self.tied_params_sequence[0])
        else:
            print("Runing")
        params,_ = Master_region(self.initial_params,*region_to_fit_norm.transpose(1, 0, 2),constraints)
        if len(self.tied_params_sequence)>1:
            for i in range(1,len(self.tied_params_sequence)):
                print("Runing:",self.tied_params_sequence[i])
                self.create_region_to_be_fit(tied_params=self.dict_region.get(self.tied_params_sequence[i]))
                Master_region.non_optimize_in_axis = 4
                Master_region.num_steps = 2*num_steps
                Master_region.learning_rate = 1e-2
                params,_ = Master_region(params,*region_to_fit_norm.transpose(1, 0, 2),constraints,list_dependencies=self.list_dependencies)
        self.Master_region  = Master_region
        self.params = params.at[:,self.mapping_params([["amplitude"],["cont"],["scale"]])].multiply(max_value[:,None])
        self.region_to_fit =  region_to_fit_norm.at[:,[1,2],:].multiply(jnp.moveaxis(jnp.tile(max_value,(2,1)),0,1)[:,:,None])
        self.outer_limits = outer_limits
        self.inner_limits = inner_limits
        #for ploting could be useful 
        self.max_value = max_value
        
        #self.norm_amplitude = params[:,RegionClass.mapping_params(["amplitude"])]
        #self.params_region = params_region.at[:,self.mapping_params(["amplitude"])].multiply(max_value)
        #self.region_to_fit =  region_to_fit_norm.at[:,[1,2],:].multiply(jnp.moveaxis(jnp.tile(max_value,(2,1)),0,1)[:,:,None])
        #params_region[amp]*max_value 
    #I suppose from here the best option is do all the stats 