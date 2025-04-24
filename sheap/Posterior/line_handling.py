__all__ = ['line_parameters','line_decomposition_measurements']

def line_decomposition_measurements(model_lines,line_dictionary,line_name ="Hbeta",c=2.99792458e5):
    components=model_lines.global_dict[f"{line_name}_complex"]['lines'][line_name]['components']
    guesses= model_lines.global_dict[f"{line_name}_complex"]['lines'][line_name]['guesses']
    if line_name=='CIV' or line_name=='MgII':
        components=components[:2]
        guesses=guesses[:2]
    lambda0=np.mean(guesses[1::3])
    s_0 = np.array([value['modelpars'] for key,value in line_dictionary.items() if key in components])
    s_e = np.array([value['modelerrs'] for key,value in line_dictionary.items() if key in components])
    sflux = np.array([value['flux'] for key,value in line_dictionary.items() if key in components])
    FWHM=2.35842*s_0[::,2]*c/lambda0
    FWHMe=2.35842*s_e[::,2]*c/lambda0
    DV=(s_0[::,1]-lambda0)*c/lambda0
    DVe=(s_e[::,1]-lambda0)*c/lambda0
    L= sflux + np.append(sflux[1:],0)[::-1]
    Le=L*s_e[::,0]/s_0[::,0]
    argfwmin=np.argmin(FWHM)
    argfwmax=np.argmax(FWHM)
    fwhmin=FWHM[argfwmin]
    fwhmine=FWHMe[argfwmin]
    fwhmax=FWHM[argfwmax]
    fwhmaxe=FWHMe[argfwmax]
    L1=L[argfwmin]
    L1e=Le[argfwmin]
    L2=L[argfwmax]
    L2e=Le[argfwmax]
    dv1=DV[argfwmin]
    dv1e=DVe[argfwmin]
    dv2=DV[argfwmax]
    dv2e=DVe[argfwmax]
    return fwhmin,fwhmax,L1,L2,dv1,dv2,fwhmine,fwhmaxe,L1e,L2e,dv1e,dv2e


#fwhm, luminosity,EW,dlambda,std,dmax,dl50,dl90,dl95,dlt
def line_parameters(results,model_lines,line_dictionary,line_name="Hbeta",c=2.99792458e5,line_measurements=False):
    """" results is a dictionary with all the results from the fit 
        model_lines: is a class that came from constrains should be update to something more easy to handle 
        or just add to results and in there it solve what to do 
        line_dictionary it contain the result for the different lines 
    """
    spxarr = results['wavelenght']
    components=model_lines.global_dict[f"{line_name}_complex"]['lines'][line_name]['components']
    guesses= model_lines.global_dict[f"{line_name}_complex"]['lines'][line_name]['guesses']
    ##########
    if line_name=='CIV' or line_name=='MgII':
        components=components[:2]
        guesses=guesses[:2]
    ###########
    lambda0=np.mean(guesses[1::3])
    #why this?
    if (spxarr[1]-spxarr[0])>0:
        xarr1=np.arange(1100.0,spxarr[0],(spxarr[1]-spxarr[0]))
        xarr2=np.arange(spxarr[-1],20000.0,(spxarr[1]-spxarr[0]))
        xarr=np.append(xarr1,spxarr)
        xarr=np.append(xarr,xarr2)
    elif (spxarr[2]-spxarr[0])>0:
        xarr1=np.arange(1100.0,spxarr[0].value,(spxarr[2]-spxarr[0]))
        xarr2=np.arange(spxarr[-1],20000.0,(spxarr[2]-spxarr[0]))
        xarr=np.append(xarr1,spxarr)
        xarr=np.append(xarr,xarr2)
    else:
        xarr1=np.arange(1100.0,spxarr[0],1.0)
        xarr2=np.arange(spxarr[-1],20000.0,1.0)
        xarr=np.append(xarr1,spxarr)
        xarr=np.append(xarr,xarr2)
    ###########
    continuous = results['continium']*10**(-results['mag_order']) # meanwhile we will assume this is in non flux units
    continuous = np.interp(xarr,spxarr,continuous,right=continuous[-1],left=continuous[0])
    #######
    xarrb=np.arange(lambda0-1000.0,lambda0+1000,0.1)
    continuousb=np.interp(xarrb,xarr,continuous,right=continuous[-1],left=continuous[0])
    #########
    luminosity = 0
    fwhm=0
    #######loop cicle######
    s_0 = np.array([value['modelpars'] for key,value in line_dictionary.items() if key in components])
    s_e = np.array([value['modelerrs'] for key,value in line_dictionary.items() if key in components])
    sflux = np.array([value['flux'] for key,value in line_dictionary.items() if key in components])
    luminosity = sflux.sum() + luminosity
    total=np.sum(gaussian_vector(xarr,*s_0.T),axis=0)# looks like i have to sum all of them 
    totalb=np.sum(gaussian_vector(xarrb,*s_0.T),axis=0)
    ###########################################
    argmaxi=arglocalmax(totalb)
    ############################
    if len(argmaxi)==1:
        #print "argmax = 1"
        f_obs_func = interp1d(xarrb, totalb-totalb[argmaxi[0]]/2.0, kind='linear')
        spline_obs = interpolation(xarrb, f_obs_func(xarrb))
        roots_obs = spline_obs.roots()
        try:
            fwhm=roots_obs[-1]-roots_obs[0]
        except:
             if line_measurements==True:
                    return 0,0,0,0,0,0,0,0,0,0
             else:
                    return 0,0,0,0,0,0,0,0,0,0,0,0*xarr,0*xarr,0*xarr,0*xarr,0,0,0,0,0,0,0,0
    elif len(argmaxi)==0 or np.mean(totalb)<=0.0:
            fwhm=0.0
    else:
        width_obs=0
        reorder=np.argsort(totalb[argmaxi])
        totalmax=totalb[argmaxi][reorder][::-1]
        argmaxi=argmaxi[reorder][::-1]
        count=0
        for arg in argmaxi:
            #print "argmax dif= 1", len(argmaxi)
            if totalb[arg]>=0.7*totalmax[0]:
                f_obs_func = interp1d(xarrb, totalb-totalb[arg]/2.0, kind='linear')
                spline_obs = interpolation(xarrb, f_obs_func(xarrb))
                roots_obs=spline_obs.roots()
                roots_obs=np.sort(roots_obs)
                try:
                    width_obs=width_obs + np.abs(roots_obs[-1]-roots_obs[0])
                    count=count+1
                except:
                    width_obs=width_obs
        try:
            fwhm=width_obs/(count)
        except:
            fwhm=width_obs
    fwhm=fwhm*c/lambda0
    ####################################################
    EW=simpson(totalb/continuousb,x=xarrb) #totalb?? maybe it should be in mag 
    if np.isnan(EW):
        EW=np.sum(totalb*(xarrb[1]-xarrb[0])/continuousb)
    if np.max(total)<=0 :
        #print('total is a 0 array')
        dlambda=0
        dmax=0
        dl95=0
        dlt=0
        dl50=0
        dl90=0
    else:
        dlambda=(np.sum(totalb*xarrb)/np.sum(totalb) - lambda0)*c/lambda0
        amax=np.argmax(totalb)
        dmax=(xarrb[amax]-lambda0)*c/lambda0
        xb=xarrb

        f_obs_func = interp1d(xarrb, totalb-totalb[amax]/2.0, kind='linear')
        spline_obs = interpolation(xarrb, f_obs_func(xarrb))
        roots_obs = spline_obs.roots()
        if len(roots_obs)<2:
                dlambda=0
                dmax=0
                dl95=0
                dlt=0
                dl50=0
                dl90=0
        else:
            try:
                arghm1=np.argmin(np.abs(xarrb-roots_obs[0]))
                arghm2=np.argmin(np.abs(xarrb-roots_obs[-1]))
                dl50=(np.sum((totalb*xarrb)[arghm1:arghm2+1])/np.sum(totalb[arghm1:arghm2+1]) - lambda0)*c/lambda0
                dl50=dl50-dmax
                f_obs_func = interp1d(xarrb, totalb-totalb[amax]/10.0, kind='linear')
                spline_obs = interpolation(xarrb, f_obs_func(xarrb))
                roots_obs=spline_obs.roots()
                
                arghm1=np.argmin(np.abs(xarrb-roots_obs[0]))
                arghm2=np.argmin(np.abs(xarrb-roots_obs[-1]))
                
                dl90=(np.sum((totalb*xarrb)[arghm1:arghm2+1])/np.sum(totalb[arghm1:arghm2+1]) - lambda0)*c/lambda0
                dl90=dl90-dmax
                
                f_obs_func = interp1d(xarrb, totalb-totalb[amax]/20.0, kind='linear')
                spline_obs = interpolation(xarrb, f_obs_func(xarrb))
                roots_obs=spline_obs.roots()
                
                arghm1=np.argmin(np.abs(xarrb-roots_obs[0]))
                arghm2=np.argmin(np.abs(xarrb-roots_obs[-1]))
                
                dl95=(np.sum((totalb*xarrb)[arghm1:arghm2+1])/np.sum(totalb[arghm1:arghm2+1]) - lambda0)*c/lambda0
                dl95=dl95-dmax
                dlt=dlambda-dmax
            except:
                dlambda=0
                dmax=0
                dl95=0
                dlt=0
                dl50=0
                dl90=0
    xm=np.dot(xarr,total)/np.sum(total)
    x2m=np.dot(xarr*xarr,total)/np.sum(total)
    std=np.sqrt(np.abs(x2m - xm*xm))*c/lambda0
########################################## ########################################
    if line_measurements==True:
        return fwhm,luminosity,EW,dlambda,std,dmax,dl50,dl90,dl95,dlt

    xo=np.linspace(xarr[0],xarr[-1],30000)
    fwhm_up=0
    fwhm_low=0
    s_up = s_0 + s_e
    s_low = np.zeros_like(s_up)
    s_low[s_up< 0] = (s_0 - s_e)[s_up< 0]
    s_up[s_up< 0] = 0
    s_low[s_low< 0] = 0
    s_up[s_low< 0] = 0
    total_up=np.sum(gaussian_vector(xarr,*s_up.T),axis=0)
    total_low=np.sum(gaussian_vector(xarr,*s_low.T),axis=0)
    fwhm_up=fwhm_up*c/lambda0
    fwhm_low=fwhm_low*c/lambda0
    argmaxi_up=arglocalmax(total_up)
    argmaxi_low=arglocalmax(total_low)
    if  len(argmaxi_up)==1:
        #print "argmax = 1"
        f_obs_func = interp1d(xarr, total-total[argmaxi_up[0]]/2.0, kind='linear')
        spline_obs = interpolation(xo, f_obs_func(xo))
        roots_obs=spline_obs.roots()
    try:
        fwhm_up=roots_obs[-1]-roots_obs[0]
    except:
        fwhm_up=0.0
    if len(argmaxi_up)==0 or np.mean(total_up)<=0.0:
        fwhm_up=0.0
    else:
        width_obs_up=0
        reorder=np.argsort(total_up[argmaxi_up])
        totalmax_up=total_up[argmaxi_up][reorder][::-1]
        argmaxi_up=argmaxi_up[reorder][::-1]
        count_up=0

        for arg_up in argmaxi_up:
            if total_up[arg_up]>=0.7*totalmax_up[0]:
                f_obs_func_up = interp1d(xarr, total_up-total_up[arg_up]/2.0, kind='linear')
                spline_obs_up = interpolation(xo, f_obs_func_up(xo))
                roots_obs_up=spline_obs_up.roots()
                try:
                    width_obs_up=width_obs_up + np.abs(roots_obs[-1]-roots_obs[0])
                    #count_up=count_up+1
                except:
                    width_obs_up=width_obs_up
        if count_up==0:
            count_up=1
        fwhm_up=width_obs_up/count_up
    if len(argmaxi_low)==1:
        try:
            f_obs_func = interp1d(xarr, total-total[argmaxi[0]]/2.0, kind='linear')
            spline_obs = interpolation(xo, f_obs_func(xo))
            roots_obs=spline_obs.roots()
            fwhm_low=roots_obs[-1]-roots_obs[0]
        except:
            fwhm_low=0
    if len(argmaxi_low)==0 or np.mean(total_low)<=0.0:
        fwhm_low=0.0
    else:
        width_obs_low=0
        reorder=np.argsort(total_low[argmaxi_low])
        totalmax_low=total_low[argmaxi_low][reorder][::-1]
        argmaxi_low=argmaxi_low[reorder][::-1]
        count_low=0
        for arg_low in argmaxi_low:
            if total_low[arg_low]>=0.7*totalmax_low[0]:
                f_obs_func_low = interp1d(xarr, total_low-total_low[arg_low]/2.0, kind='linear')
                spline_obs_low = interpolation(xo, f_obs_func_low(xo))
                roots_obs_low=spline_obs_low.roots()
                if len(roots_obs_low)==0:
                    width_obs_low=0
                    count_low=count_low+1
                    continue
                try:
                    width_obs_low=width_obs_low + np.abs(roots_obs[-1]-roots_obs[0])
                    count=count+1
                except:
                    width_obs_low=width_obs_low
        if count_low==0:
            count_low=1
            fwhm_low=width_obs_low/count_low
        if fwhm!=0 and fwhm_low==0:
            fwhm_low=2*fwhm -fwhm_up

    EW_up = simpson(total_up/continuous,x=xarr)
    EW_low = simpson(total_low/continuous,x=xarr)
    if EW!=0 and (EW_low==0):
        EW_low=2*EW -EW_up
    else:
        #dlambda=(np.sum(totalb*xarrb)/np.sum(totalb) - lambda0)*c/lambda0
        #xb=xarrb
        f_obs_func = interp1d(xarrb, totalb-totalb[amax]/2.0, kind='linear')
        spline_obs = interpolation(xarrb, f_obs_func(xarrb))
        roots_obs=spline_obs.roots()
        if np.shape(roots_obs)[0]<2:
            dlambda=0
            dmax=0
            dl95=0
            dlt=0
            dl50=0
            dl90=0
        else:
            arghm1=np.argmin(np.abs(xarrb-roots_obs[0]))
            arghm2=np.argmin(np.abs(xarrb-roots_obs[-1]))
            dl50=(np.sum((totalb*xarrb)[arghm1:arghm2+1])/np.sum(totalb[arghm1:arghm2+1]) - lambda0)*c/lambda0
            dl50=dl50-dmax
            f_obs_func = interp1d(xarrb, totalb-totalb[amax]/10.0, kind='linear')
            spline_obs = interpolation(xarrb, f_obs_func(xarrb))
            roots_obs=spline_obs.roots()
            if np.shape(roots_obs)[0]<2:
                dl90=0
            else:
                arghm1=np.argmin(np.abs(xarrb-roots_obs[0]))
                arghm2=np.argmin(np.abs(xarrb-roots_obs[-1]))

                #dl90=(np.sum((totalb*xarrb)[arghm1:arghm2+1])/np.sum(totalb[arghm1:arghm2+1]) - lambda0)*c/lambda0
                #dl90=dl90-dmax

            f_obs_func = interp1d(xarrb, totalb-totalb[amax]/20.0, kind='linear')
            spline_obs = interpolation(xarrb, f_obs_func(xarrb))
            roots_obs=spline_obs.roots()

            if np.shape(roots_obs)[0]<2:
                dl95=0
            else:
                arghm1=np.argmin(np.abs(xarrb-roots_obs[0]))
                arghm2=np.argmin(np.abs(xarrb-roots_obs[-1]))
                dl95=(np.sum((totalb*xarrb)[arghm1:arghm2+1])/np.sum(totalb[arghm1:arghm2+1]) - lambda0)*c/lambda0
                dl95=dl95-dmax
            #dlt=dlambda-dmax



    if np.max(total_up)<=0 or np.sum(total_up)==0 :
        dlambda_up=0
    else:
        dlambda_up=(np.sum(total_up*xarr)/np.sum(total_up) - lambda0)*c/lambda0

    if np.max(total_low)<=0 or np.sum(total_low)==0:
        dlambda_low=0
    else:
        dlambda_low=(np.sum(total_low*xarr)/np.sum(total_low) - lambda0)*c/lambda0



    if (dlambda!=0 and dlambda_low==0):
        dlambda_low=2*dlambda - dlambda_up
    if (dlambda!=0 and dlambda_up==0):
        dlambda_up=2*dlambda - dlambda_low

    #print component,"dlambda=",dlambda_low,dlambda, dlambda_up
    #print "sum",np.sum(total_low)
    #x2m=np.dot(xarr*xarr,total)/np.sum(total)
    x3m=np.dot(  (xarr-xm)*(xarr-xm)*(xarr-xm) , total)/(np.sum(total))
    x4m=np.dot(  (xarr-xm)*(xarr-xm)*(xarr-xm)*(xarr-xm) , total)/np.sum(total)
    std1=np.sqrt(np.abs(x2m - xm*xm))
    ske=x3m/np.power(std1,3)
    kurto=x4m/np.power(std1,4) -3
    varr=(xarr-lambda0)*c/lambda0

    return fwhm,fwhm_low,fwhm_up, luminosity,EW,EW_low,EW_up,dlambda,dlambda_low,dlambda_up,lambda0,total,continuous,varr,xarr,std,ske,kurto,dmax,dl50,dl90,dl95,dlt