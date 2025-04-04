import os 
import yaml
import glob 
import numpy as np 
regions_as_fantasy_path = glob.glob(os.path.dirname(os.path.abspath(__file__))+"/regions_as_fantasy/*.yaml")

class RegionBuilder:
    """"
    
    The regions created from here ussually need around 100 iterations for converge 
    
    """
    
    def __init__(self,xmin,xmax,n_narrow=1,n_broad=1):
        self.xmax = xmax
        self.xmin = xmin
        self.read_ymls()
        self.n_narrow = n_narrow
        self.n_broad = n_broad
        #Known relations
        self.tied_pairs = [(["OIIIb","OIIIc"],["amplitude_OIIIb_narrow", "amplitude_OIIIc_narrow", "*0.3"]),
                        (["NIIb","NIIa"],["amplitude_NIIb_narrow", "amplitude_NIIa_narrow", "*0.3"])]

    def read_ymls(self):
        paths = regions_as_fantasy_path
        self.full_regions = {}
        for path in paths:
            with open(path, 'r') as file:
                key = path.split("/")[-1].replace(".yaml", '')
                region = yaml.safe_load(file)
            self.full_regions[key] = region
        self.regions_available = self.full_regions.keys()
    def make_region(self,xmin=None,xmax=None,n_broad=None,n_narrow=None, 
                main_regions = ['hydrogen', "helium"],Fe_regions = ['Fe_uv'],narrow_plus=False,verbose=False,tied_to=[],force_linear=False):
        """_summary_
        summary of regions 'FeII_coronal', 'FeII_IZw1', 'narrow_basic', 'narrow_plus', 'Fe_uv', 'oiii_nii', 'feII_forbidden'
        Args:
            xmin (_type_, optional): _description_. Defaults to None.
            xmax (_type_, optional): _description_. Defaults to None.
            n_broad (_type_, optional): _description_. Defaults to None.
            n_narrow (_type_, optional): _description_. Defaults to None.
            main_regions (list, optional): _description_. Defaults to ['hydrogen', "helium"].
            Fe_regions (list, optional): _description_. Defaults to [].
        """
        xmin = xmin or self.xmin
        xmax = xmax or self.xmax
        n_broad = n_broad or self.n_broad
        n_narrow = n_narrow or self.n_narrow
        narrow_regions = ["narrow_basic"]
        if narrow_plus:
            narrow_regions += ["narrow_plus"]
        #Fe_regions = None or Fe_regions
        n_components = n_broad + n_narrow
        self.regions_as_fantasy = {}
        self.regions_to_fit = []
        tied_list = []
        super_tied = []
        self.main_region_lines = {}#
        if verbose:
            print(f"A region with limits {xmin} to {xmax} (A), \nwith n_broad = {n_broad} and n_narrow = {n_narrow} \nand Fe regions {Fe_regions}, \nwill be created")
        for key,region in self.full_regions.items():
            list_ = []
            for values in  region['region']:
                if xmin <= values["center"] <= xmax:
                    if key in main_regions:
                        for n in range(n_components):
                            local_copy = values.copy()
                            if n < n_narrow:
                                kind = "narrow"
                                if n>0:
                                     local_copy.update({"line_name": f"{local_copy['line_name']}{n + 1}","amplitude":0.5})  
                            else:
                                kind = "broad"
                                if n-n_narrow>=1:
                                    #print(f"{local_copy['line_name']}{n-n_narrow + 1}")
                                    if local_copy['line_name'] not in self.main_region_lines.keys():
                                        self.main_region_lines[local_copy['line_name']] = local_copy['center']
                                        
                                    local_copy.update({"line_name": f"{local_copy['line_name']}{n-n_narrow +1}","amplitude":0.5})
                            local_copy.update({"kind": kind})
                            #local_copy.update({"line_name": f"{values['line_name']}_{values['center']}"})
                            list_.append(local_copy)
                    elif key in narrow_regions:
                        for n in range(n_narrow):
                            local_copy = values.copy()
                            local_copy.update({"kind": "narrow"})
                            if n>0:
                                 local_copy.update({"line_name": f"{local_copy['line_name']}{n-1}","amplitude":0.5})
                            #local_copy.update({"line_name": f"{values['line_name']}_{values['center']}"})
                            list_.append(local_copy)
                    elif key == "broad":
                        for n in range(n_broad):
                            local_copy = values.copy()
                            local_copy.update({"kind": "broad"})
                            #local_copy.update({"line_name": f"{values['line_name']}_{values['center']}"})
                            list_.append(local_copy)
                    elif key in Fe_regions:
                        local_copy = values.copy()
                        local_copy.update({"kind": "Fe"})
                        #local_copy.update({"line_name": f"{values['line_name']}_{values['center']}"})
                        list_.append(local_copy)
                    elif key =="outflow":
                        print("COMING")
                else:
                    continue
            if list_:
                self.regions_as_fantasy[key] = list_
                self.regions_to_fit += list_
                available_lines = np.array([line.get("line_name") for line in list_ if "line_name" in line]).ravel()
                for pair,factor in self.tied_pairs:
                    if sum(line in pair for line in available_lines)==2:
                        tied_list.append(factor)
                        super_tied.append(factor)
                        
        if any("Fe" in  key for key in self.regions_as_fantasy.keys()):
            params = ["center","width"]
            #print(self.regions_as_fantasy.keys())
            centers = np.array([line["center"] for key, region in self.regions_as_fantasy.items() if "Fe" in key for line in region])
            fe_region = [line for key, region in self.regions_as_fantasy.items() if "Fe" in key for line in region]
            n_r = np.argmin(abs(centers - np.median(centers)))
            center_line = fe_region[n_r]
            #self.fe_region = fe_region
            for _,fe in enumerate(fe_region):
                if _ == n_r:
                    continue
                for p in params:
                    tied_list.append([f"{p}_{fe["line_name"]}_{fe["kind"]}",f"{p}_{center_line["line_name"]}_{center_line["kind"]}"])
        #_Halpha_narrow
        if all(key in self.main_region_lines.keys() for key in ("Halpha", "Hbeta")):
            mainline = "Halpha"
        elif any(key in self.main_region_lines.keys() for key in ("Halpha", "Hbeta")):
            mainline = next(key for key in ("Halpha", "Hbeta") if key in self.main_region_lines.keys())
        params = ["center","width"]
        tied_list += [[f"{p}_{line['line_name']}_{line['kind']}", f"{p}_{mainline}_narrow"] for line in self.regions_to_fit if (line['kind'] == "narrow" and line["line_name"] != "Halpha") for p in params]
        tied_list += [[f"{p}_{line['line_name']}_{line['kind']}", f"{p}_{mainline}_broad"] for line in self.regions_to_fit if (line['kind'] == "broad" and line["line_name"] != "Halpha") for p in params]
        #self.narrows = narrows
        #[line for key, region in self.regions_as_fantasy.items() if "narrow" in line["kind"] for line in region]
        
        #if any("narrow" == line["kind"] for key, region in self.regions_as_fantasy.items() if "Fe" not in key for line in region)
        #narrows = np.array([line["center"] for key, region in self.regions_as_fantasy.items() if "Fe" in key for line in region])
        #broads = np.array([line["center"] for key, region in self.regions_as_fantasy.items() if "Fe" in key for line in region])
        #self.narrows
        
        if len(tied_to)>0:
            for t in tied_to:
                region,params,line_kind = t
                if isinstance(region,str):
                    region = [region]
                if isinstance(params,str):
                    params = [params]
                for r in region:
                    for ii in self.regions_as_fantasy.get(r,[]):
                        #print(ii )
                        local_ = f'{ii["line_name"]}_{ii["kind"]}'
                        if line_kind==local_:
                            continue
                        for p in params:
                            tied_list.append([f"{p}_{local_}",f"{p}_{line_kind}"])
        if (xmax-xmin > 2000) and not force_linear:
            self.regions_to_fit += [{"center":0,"kind":"cont","line_name":"cont","profile":"power_law"}]
        self.xmax = xmax
        self.xmin = xmin
        self.tied_list = []
        self.super_tied = []
        for i in tied_list:
            if i not in self.tied_list:
                self.tied_list.append(i)
            else:
                continue
        for i in super_tied:
            if i not in self.super_tied:
                self.super_tied.append(i)
            else:
                continue
        #self.tied_list 
    def to_complex(self,add_free=True):
        complex = {"region":self.regions_to_fit,"tied_params_step_1":self.tied_list,"inner_limits": [self.xmin+50 , self.xmax-50 ], "outer_limits": [self.xmin , self.xmax ]}
        if add_free:
            complex["tied_params_step_2"] = self.super_tied
        return complex
    



# def build_a_region(xmin,xmax,n_narrow=1,n_broad=1,tied_to={},add=["hydrogen",'helium']):
    
#     """build our own region to be fited after 
#     #TODO clean, add constrain in kind of lines this means look for all the narrow and contraint the witdh 
#     Args:
#         xlim (_type_): _description_
#         xmax (_type_): _description_
#     """
#     #Fe_regions = {}
#     n_ = n_narrow + n_broad
#     regions_as_fantasy = {}
#     tied_list = []
#     tie_pairs = [
#     (["OIIIb","OIIIc"],["amplitude_OIIIb_narrow", "amplitude_OIIIc_narrow", "*0.3"]),
#     (["NIIb","NIIa"],["amplitude_NIIb_narrow", "amplitude_NIIa_narrow", "*0.3"])
# ]
#     for path in regions_as_fantasy_path:
#         with open(path, 'r') as file:
#             key = path.split("/")[-1].replace(".yaml", '')
#             region = yaml.safe_load(file)
#             list_ = []
#             for local in region["region"]:
#                 if xmin <= local["center"] <= xmax:
#                     # Handle the 'hydrogen' and 'helium' keys differently
#                     if key in ['hydrogen', "helium"]:
#                         for n in range(n_):
#                             # Make a copy of local to avoid mutating the same dictionary
#                             local_copy = local.copy()
#                             if n < n_narrow:
#                                 kind = "narrow"
#                             else:
#                                 kind = "broad"
#                             local_copy.update({"kind": kind})
#                             list_.append(local_copy)
                            
#                             # # Example of populating tied_list (adjust as needed):
#                             # line_name = local_copy.get("line_name")
#                             # if line_name is not None:
#                             #     tied_list.append([
#                             #         f"center_{line_name}_{kind}",
#                             #         f"center_{line_name}_{kind}"
#                             #     ])
#                     elif "Fe" in key:
#                         local_copy = local.copy()
#                         local_copy.update({"kind": "Fe"})
#                         list_.append(local_copy)
#                     elif key == "narrow_basic":
#                         local_copy = local.copy()
#                         local_copy.update({"kind": "narrow"})
#                         list_.append(local_copy)
#                     else:
#                         list_.append(local)
#             # Update the dictionary once after processing all items in the file
#             if list_:
#                 regions_as_fantasy[key] = list_
#             #print(list_)
#             available_lines = np.array([line.get("line_name") for line in list_ if "line_name" in line]).ravel()
#             for pair,factor in tie_pairs:
#                 if sum(line in pair for line in available_lines)==2:
#                     tied_list.append(factor)
    
#     if len(tied_to)>0:
#         for t in tied_to:
#             region,params,line_kind = t
#             if isinstance(region,str):
#                 region = [region]
#             if isinstance(params,str):
#                 params = [params]
#             for r in region:
#                 for ii in regions_as_fantasy[r]:
#                     #print(ii )
#                     local_ = f'{ii["line_name"]}_{ii["kind"]}'
#                     if line_kind==local_:
#                         continue
#                     for p in params:
#                         tied_list.append([f"{p}_{local_}",f"{p}_{line_kind}"])
    
    
#     regions_to_sheap = [item for sublist in regions_as_fantasy.values() for item in sublist]
#     return regions_to_sheap,regions_as_fantasy,tied_list