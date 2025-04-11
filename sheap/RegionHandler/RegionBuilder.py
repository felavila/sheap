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
        #this can be updated for the different "components"
        #TODO this have to be added for the different components for instance if one of this is cover have to be added for component 1,2,3 
        self.known_tied_relations = [(["OIIIb","OIIIc"],["amplitude_OIIIb_component_narrow", "amplitude_OIIIc_component_narrow", "*0.3"]),
                        (["NIIb","NIIa"],["amplitude_NIIb_component_narrow", "amplitude_NIIa_component_narrow", "*0.3"]),
                         (["NIIb","NIIa"],["center_NIIb_component_narrow", "center_NIIa_component_narrow"]),
                         (["OIIIb","OIIIc"],["center_OIIIb_component_narrow", "center_OIIIc_component_narrow"]),
                         #(["NIIb","NIIa","2"],["amplitude_NIIb_2_narrow", "amplitude_NIIa_2_narrow", "*0.3"]),
                         #(["NIIb","NIIa","2"],["center_NIIb_2_narrow", "center_NIIa_2_narrow"])
                         ]
    def read_ymls(self):
        paths = regions_as_fantasy_path
        self.full_regions = {}
        for path in paths:
            with open(path, 'r') as file:
                key = path.split("/")[-1].replace(".yaml", '')
                region = yaml.safe_load(file)
            self.full_regions[key] = region
        self.regions_available = self.full_regions.keys()
        #self.regions_to_fit = {}
        
    def main_lines_handler(self,values,n_broad,n_narrow):
        """_summary_

        Args:
            values (_type_): dictionary that contain ?
            n_broad (_type_): _description_
            n_narrow (_type_): _description_
        """
        n_components = n_broad + n_narrow
        local_ = []
        for n in range(n_components):
            local_copy = values.copy()
            local_copy["component"] = 1
            if n < n_narrow:
                kind = "narrow"
                if n > 0:
                    local_copy.update({"line_name":local_copy['line_name'],"amplitude":0.5,"component":n+1}) 
            else:
                kind = "broad"
                if n - n_narrow >= 1:
                    local_copy.update({"line_name": f"{local_copy['line_name']}","amplitude":0.5,"component":n-n_narrow +1})
            local_copy.update({"kind": kind})
            if local_copy not in self.main_region_lines:
                self.main_region_lines.append(local_copy)
            local_.append(local_copy)
        
        return local_
    def narrow_lines_handler(self,values,n_narrow,add_out_flow):
        local_ = []
        for n in range(n_narrow):
            local_copy = values.copy()
            #local_copy["component"] = 1
            local_copy.update({"kind": "narrow","component": 1})
            if n>0:
                local_copy.update({"line_name": f"{local_copy['line_name']}","amplitude":0.5,"component":n+1})
            local_.append(local_copy)
            if add_out_flow and n==0:
                local_copy = values.copy()
                #maybe add list of names 
                local_copy.update({"kind": "outflow","component": 10}) #the idea is this never get attached or it is?
                local_.append(local_copy)
        #print(local_)
        return local_
    
    def broad_regions_handler(self,values,n_broad):
        local_ = []
        for n in range(n_broad):
            local_copy = values.copy()
            local_copy.update({"kind": "broad","component": 1})
            if n>0:
                local_copy.update({"line_name": f"{local_copy['line_name']}","amplitude":0.5,"component":n+1})
            local_.append(local_copy)
        return local_
            
    def make_region(self,xmin=None,xmax=None,n_broad=None,n_narrow=None, 
                main_regions = ['hydrogen', "helium"],Fe_regions = ['Fe_uv']
                ,narrow_plus= False,verbose= False,force_linear= False,add_out_flow= False
                ,main_lines = ["Hbeta","Halpha"],tied_narrow_to = None ,tied_broad_to = None):
        
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
        self.n_broad = n_broad or self.n_broad
        self.n_narrow = n_narrow or self.n_narrow
        narrow_regions = ["narrow_basic"]
        self.tied_params = []
        self.tied_params_ = [] #super tied
        self.tied_fe = []
        
        if narrow_plus:
            narrow_regions += ["narrow_plus"]
        self.main_region_lines = []
        local_region_list = []
        for key,region in self.full_regions.items(): #full_regions this should already have reduce the regions?
            for values in  region['region']:  
                local_ = []
                if key in main_regions and xmin <= values["center"] <= xmax:
                    local_= self.main_lines_handler(values,n_broad,n_narrow)
                    #local_region_list += local_
                elif key in narrow_regions and xmin <= values["center"] <= xmax:
                    local_ = self.narrow_lines_handler(values,n_narrow,add_out_flow)
                elif key in ["broad"] and xmin <= values["center"] <= xmax:
                    local_ = self.broad_regions_handler(values,n_broad)
                elif key in Fe_regions:
                    local_copy = values.copy()
                    local_copy.update({"kind": "Fe","component":20,"amplitude":0.1})
                    local_.append(local_copy)
                local_region_list += local_
        available_lines =  np.array([line.get("line_name") for line in local_region_list if "line_name" in line]).ravel()
        available_lines = np.unique(available_lines) # it is necesary make it unique?
        available_kind = np.array([line.get("kind") for line in local_region_list if "line_name" in line]).ravel()
                    
        if sum("Fe" in key for key in available_kind)>=2:
            params = ["center","width"]
            centers,line_names,kinds = np.array([[region.get("center"),region.get("line_name"),region.get("kind")] for region in local_region_list if "Fe" in region.get("kind")]).T
            centers = centers.astype(float)
            n_central_line = np.argmin(abs(centers - np.median(centers))) #problematic if you thing about the central one could be in the center of the region this means hbeta
            for _,fe in enumerate(line_names):
                if _ == n_central_line:
                    continue
                for p in params:
                    tied_ = [f"{p}_{line_names[_]}_20_{kinds[_]}",f"{p}_{line_names[n_central_line]}_20_{kinds[n_central_line]}"]
                    self.tied_params.append(tied_)
                    self.tied_fe.append(tied_)
        
        main_list_available = []
        for line_dic in self.main_region_lines:
            if line_dic.get("line_name") in main_lines:
                main_list_available.append(line_dic.get("line_name"))
                #main_list_available.append([line_dic.get("line_name"),line_dic.get("component")])
        mainline = main_list_available[0] #?
        
        if tied_broad_to:
            assert isinstance(tied_broad_to,str) or  isinstance(tied_broad_to,dict), "tied_broad_to only can be dict or str "
        if tied_narrow_to:
            assert isinstance(tied_narrow_to,str) or  isinstance(tied_narrow_to,dict), "tied_broad_to only can be dict or str "
        if not tied_narrow_to:
            tied_narrow_to = mainline
        if not tied_broad_to:
            tied_broad_to = mainline   
        if isinstance(tied_narrow_to,str):
            tied_narrow_to = {k:{"line_name":tied_narrow_to,"component":k} for k in range(1,n_narrow+1)}
        if isinstance(tied_broad_to,str):
            tied_broad_to = {k:{"line_name":tied_broad_to,"component":k} for k in range(1,n_broad+1)}
        if isinstance(tied_narrow_to,dict):
            tied_narrow_to = {k: {"line_name": tied_narrow_to.get(k).get("line_name"),"component": tied_narrow_to.get(k).get("component",k)} for k in range(1,n_narrow+1)}
        if isinstance(tied_broad_to,dict):
            tied_broad_to = {k: {"line_name": tied_broad_to.get(k).get("line_name"),"component": tied_broad_to.get(k).get("component",k)} for k in range(1,n_broad+1)}

        if local_region_list:
            for pair,factor in self.known_tied_relations:
                #print(pair,factor)
                if sum(line in pair for line in available_lines)==2:
                    factor_copy = factor.copy()
                    for k  in range(1,n_narrow+1):
                        factor = [f.replace("component",str(k)) for f in factor_copy]
                        self.tied_params.append(factor)
                        self.tied_params_.append(factor)
        
        params = ["center","width"]
        self.tied_params += [[f"{p}_{line['line_name']}_{line["component"]}_{line['kind']}", f"{p}_{tied_narrow_to.get(line["component"]).get("line_name")}_{tied_narrow_to.get(line["component"]).get("component")}_narrow"] for line in local_region_list if (line['kind'] == "narrow" and line["line_name"] != tied_narrow_to) for p in params]
        
        self.tied_params += [[f"{p}_{line['line_name']}_{line["component"]}_{line['kind']}", f"{p}_{tied_broad_to.get(line["component"]).get("line_name")}_{tied_broad_to.get(line["component"]).get("component")}_broad"] for line in local_region_list if (line['kind'] == "broad" and line["line_name"] != tied_broad_to) for p in params]       
        
        if (xmax-xmin > 2000) and not force_linear:
            local_region_list += [{"center":0,"kind":"cont","line_name":"cont","profile":"power_law"}]
        
        self.xmax = xmax
        self.xmin = xmin
        self.regions_to_fit = local_region_list
        #return local_region_list,self.main_region_lines
    
    # def make_region_old(self,xmin=None,xmax=None,n_broad=None,n_narrow=None, 
    #             main_regions = ['hydrogen', "helium"],Fe_regions = ['Fe_uv'],narrow_plus=False,verbose=False,tied_to=[],force_linear=False,out_flow=False):
    #     """_summary_
    #     summary of regions 'FeII_coronal', 'FeII_IZw1', 'narrow_basic', 'narrow_plus', 'Fe_uv', 'oiii_nii', 'feII_forbidden'
    #     Args:
    #         xmin (_type_, optional): _description_. Defaults to None.
    #         xmax (_type_, optional): _description_. Defaults to None.
    #         n_broad (_type_, optional): _description_. Defaults to None.
    #         n_narrow (_type_, optional): _description_. Defaults to None.
    #         main_regions (list, optional): _description_. Defaults to ['hydrogen', "helium"].
    #         Fe_regions (list, optional): _description_. Defaults to [].
    #     """
    #     xmin = xmin or self.xmin
    #     xmax = xmax or self.xmax
    #     n_broad = n_broad or self.n_broad
    #     n_narrow = n_narrow or self.n_narrow
    #     narrow_regions = ["narrow_basic"]
    #     if narrow_plus:
    #         narrow_regions += ["narrow_plus"]
    #     #Fe_regions = None or Fe_regions
    #     n_components = n_broad + n_narrow
    #     self.regions_as_fantasy = {}
    #     self.regions_to_fit = []
    #     tied_list = []
    #     super_tied = []
    #     self.main_region_lines = {}#
    #     if verbose:
    #         print(f"A region with limits {xmin} to {xmax} (A), \nwith n_broad = {n_broad} and n_narrow = {n_narrow} \nand Fe regions {Fe_regions}, \nwill be created")
    #     for key,region in self.full_regions.items():
    #         list_ = []
    #         for values in  region['region']:
    #             if xmin <= values["center"] <= xmax:
    #                 if key in main_regions:
    #                     for n in range(n_components):
    #                         local_copy = values.copy()
    #                         if n < n_narrow:
    #                             kind = "narrow"
    #                             if n>0:
    #                                 local_copy.update({"line_name": f"{local_copy['line_name']}{n + 1}","amplitude":0.5})  
    #                         else:
    #                             kind = "broad"
    #                             if local_copy['line_name'] not in self.main_region_lines.keys():
    #                                     self.main_region_lines[local_copy['line_name']] = local_copy['center']
    #                             if n-n_narrow>=1:
    #                                 local_copy.update({"line_name": f"{local_copy['line_name']}{n-n_narrow +1}","amplitude":0.5})
    #                         local_copy.update({"kind": kind})
    #                         #local_copy.update({"line_name": f"{values['line_name']}_{values['center']}"})
    #                         list_.append(local_copy)
    #                 elif key in narrow_regions:
    #                     for n in range(n_narrow):
    #                         local_copy = values.copy()
    #                         local_copy.update({"kind": "narrow"})
    #                         if n>0:
    #                             local_copy.update({"line_name": f"{local_copy['line_name']}{n-1}","amplitude":0.5})
    #                         #local_copy.update({"line_name": f"{values['line_name']}_{values['center']}"})
    #                         list_.append(local_copy)
    #                         if out_flow:
    #                             local_copy.update({"kind": "outflow"})
    #                             list_.append(local_copy)
    #                 elif key == "broad":
    #                     for n in range(n_broad):
    #                         local_copy = values.copy()
    #                         local_copy.update({"kind": "broad"})
    #                         #local_copy.update({"line_name": f"{values['line_name']}_{values['center']}"})
    #                         list_.append(local_copy)
    #                 elif key in Fe_regions:
    #                     local_copy = values.copy()
    #                     local_copy.update({"kind": "Fe"})
    #                     #local_copy.update({"line_name": f"{values['line_name']}_{values['center']}"})
    #                     list_.append(local_copy)
    #                 elif key =="outflow":
    #                     print("COMING")
    #             else:
    #                 continue
    #         if list_:
    #             self.regions_as_fantasy[key] = list_
    #             self.regions_to_fit += list_
    #             available_lines = np.array([line.get("line_name") for line in list_ if "line_name" in line]).ravel()
    #             for pair,factor in self.tied_pairs:
    #                 if sum(line in pair for line in available_lines)==2:
    #                     tied_list.append(factor)
    #                     super_tied.append(factor)
    #     self.tied_fe = []              
    #     if any("Fe" in  key for key in self.regions_as_fantasy.keys()):
    #         params = ["center","width"]
    #         #print(self.regions_as_fantasy.keys())
    #         centers = np.array([line["center"] for key, region in self.regions_as_fantasy.items() if "Fe" in key for line in region])
    #         fe_region = [line for key, region in self.regions_as_fantasy.items() if "Fe" in key for line in region]
    #         n_r = np.argmin(abs(centers - np.median(centers)))
    #         center_line = fe_region[n_r]
    #         #self.fe_region = fe_region
    #         for _,fe in enumerate(fe_region):
    #             if _ == n_r:
    #                 continue
    #             for p in params:
    #                 tied_ = [f"{p}_{fe["line_name"]}_{fe["kind"]}",f"{p}_{center_line["line_name"]}_{center_line["kind"]}"]
    #                 tied_list.append(tied_)
    #                 self.tied_fe.append(tied_)
    #     #_Halpha_narrow
    #     if all(key in self.main_region_lines.keys() for key in ("Halpha", "Hbeta")):
    #         mainline = "Halpha"
    #     elif any(key in self.main_region_lines.keys() for key in ("Halpha", "Hbeta")):
    #         mainline = next(key for key in ("Halpha", "Hbeta") if key in self.main_region_lines.keys())
        
    #     params = ["center","width"]
    #     tied_list += [[f"{p}_{line['line_name']}_{line['kind']}", f"{p}_{mainline}_narrow"] for line in self.regions_to_fit if (line['kind'] == "narrow" and line["line_name"] != "Halpha") for p in params]
    #     tied_list += [[f"{p}_{line['line_name']}_{line['kind']}", f"{p}_{mainline}_broad"] for line in self.regions_to_fit if (line['kind'] == "broad" and line["line_name"] != "Halpha") for p in params]
    #     #self.narrows = narrows
    #     #[line for key, region in self.regions_as_fantasy.items() if "narrow" in line["kind"] for line in region]
        
    #     #if any("narrow" == line["kind"] for key, region in self.regions_as_fantasy.items() if "Fe" not in key for line in region)
    #     #narrows = np.array([line["center"] for key, region in self.regions_as_fantasy.items() if "Fe" in key for line in region])
    #     #broads = np.array([line["center"] for key, region in self.regions_as_fantasy.items() if "Fe" in key for line in region])
    #     #self.narrows
        
    #     if len(tied_to)>0:
    #         for t in tied_to:
    #             region,params,line_kind = t
    #             if isinstance(region,str):
    #                 region = [region]
    #             if isinstance(params,str):
    #                 params = [params]
    #             for r in region:
    #                 for ii in self.regions_as_fantasy.get(r,[]):
    #                     #print(ii )
    #                     local_ = f'{ii["line_name"]}_{ii["kind"]}'
    #                     if line_kind==local_:
    #                         continue
    #                     for p in params:
    #                         tied_list.append([f"{p}_{local_}",f"{p}_{line_kind}"])
    #     if (xmax-xmin > 2000) and not force_linear:
    #         self.regions_to_fit += [{"center":0,"kind":"cont","line_name":"cont","profile":"power_law"}]
    #     self.xmax = xmax
    #     self.xmin = xmin
    #     self.tied_list = []
    #     self.super_tied = []
    #     for i in tied_list:
    #         if i not in self.tied_list:
    #             self.tied_list.append(i)
    #         else:
    #             continue
    #     for i in super_tied:
    #         if i not in self.super_tied:
    #             self.super_tied.append(i)
    #         else:
    #             continue
    #     #self.tied_list 
    def to_complex(self,add_free=True,free_Fe= True):
        complex = {"region":self.regions_to_fit,"tied_params_step_1":self.tied_params,"inner_limits": [self.xmin+50 , self.xmax-50 ], "outer_limits": [self.xmin , self.xmax ]}
        if add_free:
            complex["tied_params_step_2"] = self.tied_params_
            if not free_Fe:
                complex["tied_params_step_2"] = self.tied_params_ + self.tied_fe
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