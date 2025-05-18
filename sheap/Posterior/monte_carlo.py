# import os
# import numpy as np
# import yaml
# import pandas as pd
# import copy
# from sheap.Posterior
# __all__ = ['results_after_montecarlo','monte_carlo']
# #i have to solve this
# module_dir = os.path.dirname(os.path.abspath(__file__))
# with open(os.path.join(module_dir,"tabuled_values/dictionary_values.yaml"), 'r') as file:
#     #this should be done more eficiently in some way
#     tabuled_values = yaml.safe_load(file)

# def results_after_montecarlo(results,model_lines,line_dictionary,line_to_fit):
#     properties=['FWHM','FWnarrow','FWbroad','SIGMA','EW','DLAMBDA','DLnarrow','DLbroad','DLmax','DL50','DL90','DL95','DLt','Lcont','Lpeak','Lline','Lnarrow','Lbroad','SKEWNESS','KURTOSIS','MASS','MDOT']
#     values=['mean','median','error_up','error_low','best_fit']
#     bestfit_measurements,mean_measurements,errorup_measurements,errorlow_measurements,median_measurements = monte_carlo(results,model_lines,line_dictionary,line_name=line_to_fit)
#     pandas_obj=pd.DataFrame(np.array([mean_measurements,median_measurements,errorup_measurements,errorlow_measurements,bestfit_measurements]),columns=properties,index=values)
#     return pandas_obj

# #line_parameters ->model1, line_decomposition_measurements->model
# def monte_carlo(results,model_lines,line_dictionary,line_name="Hbeta",iterations_per_object=100):
#     copy_line_dictionary = copy.deepcopy(line_dictionary)
#     fwhm,fwhm_low,fwhm_up, luminosity,EW,EW_low,EW_up,dlambda,dlambda_low,dlambda_up,lambda0,c_total,conti,varr,xarr,std,ske,kurto,dlmax,dl50,dl90,dl95,dlt = line_parameters(results,model_lines,line_dictionary,line_name =line_name)
#     fwmin,fwmax,l1,l2,dv1,dv2,fwhmine,fwhmaxe,l1e,l2e,dv1e,dv2e = line_decomposition_measurements(model_lines,line_dictionary,line_name =line_name)
#     #iterations_per_object = 100
#     continuum_bands,logK,alpha,slope,A,B    = tabuled_values.values()
#     spdata,spxarr,mag_order = [results[key] for key in ["data_raw",'wavelenght',"mag_order"]]
#     ###########################################
#     fwhm_array=np.zeros(iterations_per_object)
#     luminosity_array=np.zeros(iterations_per_object)
#     EW_array=np.zeros(iterations_per_object)
#     dlambda_array=np.zeros(iterations_per_object)
#     std_array=np.zeros(iterations_per_object)
#     dlmax_array=np.zeros(iterations_per_object)
#     dl50_array=np.zeros(iterations_per_object)
#     dl90_array=np.zeros(iterations_per_object)
#     dl95_array=np.zeros(iterations_per_object)
#     dlt_array=np.zeros(iterations_per_object)
#     ske_array=np.zeros(iterations_per_object)
#     kurto_array=np.zeros(iterations_per_object)
#     lcont_array=np.zeros(iterations_per_object)
#     lmax_array=np.zeros(iterations_per_object)
#     Mass_array=np.zeros(iterations_per_object)
#     Mdot_array=np.zeros(iterations_per_object)
#     ##########################################
#     ske=skew(c_total,nan_policy='omit')
#     kurto=kurtosis(c_total,nan_policy='omit')
#     lmax=c_total.max()*10**(mag_order)*1.0
#     wavelength_cont=continuum_bands[line_name]
#     arg1450=np.argmin(np.abs(xarr-wavelength_cont))
#     arg14501=np.argmin(np.abs(spxarr-wavelength_cont))
#     cont1450=conti[arg1450]*10**(mag_order)*wavelength_cont*1.0
#     lcont=cont1450
#     npix=20
#     #This should be somethink like just took this values outside the array range
#     # Cropping logic based on arg14501 value
#     if arg14501 >= npix:
#         # Crop the array by removing the specified range
#         cropped_array = np.delete(spdata, np.arange(arg14501 - npix, arg14501 + npix))
#     else:
#         # Crop the array by removing the first 2*npix elements
#         cropped_array = np.delete(spdata, np.arange(0, 2 * npix))
#     # Calculate lconte and signal-to-noise ratio (sn)
#     lconte = np.std(cropped_array) * wavelength_cont
#     sn = lcont / lconte
#     lcont_array=normal_array(lcont,lconte,iterations_per_object)
#     lmax_array=normal_array(lmax,lconte/(1.0*wavelength_cont),int(wavelength_cont))#np.random.normal(lmax,lconte,iterations_per_object)
#     if line_name=='CIV' or line_name=='MgII':
#         print('MgII or CIV\n ')
#         fwhm1,fwhm_low1,fwhm_up1, luminosity,EW,EW_low1,EW_up1,dlambda1,dlambda_low1,dlambda_up1,lambda01,c_total1,conti1,varr1,xarr1,std1,ske1,kurto1,dmax1,dl501,dl901,dl951,dlt1 = line_parameters(results,model_lines,line_dictionary)
#         lmax=c_total1.max()*10**(mag_order)*1.0
#         lmax_array=normal_array(lmax,lconte/(1.0*wavelength_cont),iterations_per_object)#np.random.normal(lmax,lconte,iterations_per_object)

#     Mass=(logK[line_name]+alpha[line_name]*np.log10(lcont/1e44)+slope[line_name]*np.log10(fwhm/1e3))
#     try:
#         L5100pred=1e44*A[line_name]*(lcont/1e44)**B[line_name]
#     except ValueError:
#         print('Lcont is negative, cannot be raised to a power for object ',"?")
#         L5100pred=0
#     Lv5100pred=5100e-8*L5100pred/3e10
#     f0=1.2e30# erg/sec/Hz
#     bv=2.0
#     fth1=f0*0.86*(1+bv*0.86)/(1+bv) #assuming an inclination of 30*
#     if Lv5100pred<0:
#         Mdot=0
#     else:
#         Mdot=(Lv5100pred/(fth1))**1.5/10**(Mass-8)
#     ############################################
#     # monte carlo iteration to obtain median and errors
#     for itera in range(iterations_per_object):
#         loop=0
#         while fwhm_array[itera]==0 and loop<10:
#             loop=loop+1
#         for component in model_lines.global_dict[f"{line_name}_complex"]['lines'][line_name]['components']:
#             standard_dev0=line_dictionary[component]['modelpars'][0]
#             amplitude0=line_dictionary[component]['modelpars'][2]
#             lambda_center0=line_dictionary[component]['modelpars'][1]
#             if component in line_dictionary.keys():
#                 try:
#                     amplitude=np.random.normal(line_dictionary[component]['modelpars'][0],line_dictionary[component]['modelerrs'][0])
#                 except:
#                     amplitude=line_dictionary[component]['modelpars'][0]
#                 try:
#                     lambda_center=np.random.normal(line_dictionary[component]['modelpars'][1],line_dictionary[component]['modelerrs'][1])
#                 except:
#                     lambda_center=line_dictionary[component]['modelpars'][1]
#                 try:
#                     standard_dev=np.random.normal(line_dictionary[component]['modelpars'][2],line_dictionary[component]['modelerrs'][2])
#                 except:
#                     standard_dev=line_dictionary[component]['modelpars'][2]
#                 copy_line_dictionary[component]['modelpars']=[amplitude, lambda_center, standard_dev]
#                 try:
#                     copy_line_dictionary[component]['flux']=line_dictionary[component]['flux']*standard_dev*amplitude/(standard_dev0*amplitude0)

#                 except:
#                     copy_line_dictionary[component]['flux']=line_dictionary[component]['flux']
#             else:
#                 print(component, " does not belong to ", line_name, ' complex in the object ', "?")
#                 print("check that you are using the right directory of  the files need for this measurement")
#                 print("and or that all the files exists for this particular object")
#                 sys.exit()
#         fwhm_array[itera], luminosity_array[itera],EW_array[itera],dlambda_array[itera],std_array[itera],dlmax_array[itera],dl50_array[itera],dl90_array[itera],dl95_array[itera],dlt_array[itera]=line_parameters(results,model_lines,copy_line_dictionary,line_measurements=True,line_name=line_name)
#         ske_array[itera]=skew(c_total,nan_policy='omit')
#         kurto_array[itera]=kurtosis(c_total,nan_policy='omit')
#         #lmax_array[itera]=c_total.max()*10**(mag_order)*1.0
#         Mass_array[itera]=(logK[line_name]+alpha[line_name]*np.log10(lcont_array[itera]/1e44)+slope[line_name]*np.log10(fwhm_array[itera]/1e3))
#         L5100pred=1e44*A[line_name]*(lcont_array[itera]/1e44)**B[line_name]
#         Lv5100pred=5100e-8*L5100pred/3e10
#         f0=1.2e30# erg/sec/Hz
#         bv=2.0
#         fth1=f0*0.86*(1+bv*0.86)/(1+bv) #assuming an inclination of 30*
#         Mdot_array[itera]=(Lv5100pred/(fth1))**1.5/10**(Mass_array[itera]-8)
#     #import matplotlib.pyplot as plt
#     #plt.plot(fwhm_array)
#     bestfit_measurements=np.array([fwhm,fwmin,fwmax,std,EW,dlambda,dv1,dv2,dlmax,dl50,dl90,dl95,dlt,lcont,lmax,luminosity,l1,l2,ske,kurto,Mass,Mdot])#,'Mass','Mdot','LLedd'])

#     mean_measurements=np.array([np.mean(fwhm_array),fwmin,fwmax,np.mean(std_array),np.mean(EW_array),np.mean(dlambda_array),dv1,dv2,np.mean(dlmax_array),np.mean(dl50_array),np.mean(dl90_array),np.mean(dl95_array),np.mean(dlt_array),np.mean(lcont_array),np.mean(lmax_array),np.mean(luminosity_array),l1,l2,np.mean(ske_array),np.mean(kurto_array),np.mean(Mass_array),np.mean(Mdot_array)])#,'Mass','Mdot','LLedd'])

#     errorup_measurements=np.array([np.percentile(fwhm_array,84),fwmin+fwhmine,fwmax+fwhmaxe,np.percentile(std_array,84),np.percentile(EW_array,84),np.percentile(dlambda_array,84),dv1e+dv1,dv2e+dv2,np.percentile(dlmax_array,84),np.percentile(dl50_array,84),np.percentile(dl90_array,84),np.percentile(dl95_array,84),np.percentile(dlt_array,84),lcont+lconte,np.percentile(lmax_array,84),np.percentile(luminosity_array,84),l1+l1e,l2+l2e,np.percentile(ske_array,84),np.percentile(kurto_array,84),np.percentile(Mass_array,84),np.percentile(Mdot_array,84)])#,'Mass','Mdot','LLedd'])

#     errorlow_measurements=np.array([np.percentile(fwhm_array,16),fwmin-fwhmine,fwmax-fwhmaxe,np.percentile(std_array,16),np.percentile(EW_array,16),np.percentile(dlambda_array,16),dv1-dv1e,dv2-dv2e,np.percentile(dlmax_array,16),np.percentile(dl50_array,16),np.percentile(dl90_array,16),np.percentile(dl95_array,16),np.percentile(dlt_array,16),lcont-lconte,np.percentile(lmax_array,16),np.percentile(luminosity_array,16),l1-l1e,l2-l2e,np.percentile(ske_array,16),np.percentile(kurto_array,16),np.percentile(Mass_array,16),np.percentile(Mdot_array,16)])#,'Mass','Mdot','LLedd'])

#     median_measurements=np.array([np.percentile(fwhm_array,50),fwmin,fwmax,np.percentile(std_array,50),np.percentile(EW_array,50),np.percentile(dlambda_array,50),dv1,dv2,np.percentile(dlmax_array,50),np.percentile(dl50_array,50),np.percentile(dl90_array,50),np.percentile(dl95_array,50),np.percentile(dlt_array,50),lcont,np.percentile(lmax_array,50),np.percentile(luminosity_array,50),l1,l2,np.percentile(ske_array,50),np.percentile(kurto_array,50),np.percentile(Mass_array,50),np.percentile(Mdot_array,50)])#,'Mass','Mdot','LLedd'])#,'Mass','Mdot','LLedd'])

#     ####################################################################
#     errorup_measurements=errorup_measurements-median_measurements

#     errorlow_measurements=median_measurements-errorlow_measurements

#     return bestfit_measurements,mean_measurements,errorup_measurements,errorlow_measurements,median_measurements
