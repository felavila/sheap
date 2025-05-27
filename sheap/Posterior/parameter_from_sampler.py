from sheap.Mappers.helpers import mapping_params


def full_params_sampled_to_posterior_params(full_sample,kinds_map):
    dict_1 = {}
    for k, k_map in kinds_map.items():
         if k not in ['fe', 'continuum']:
            idx_amplitude = mapping_params(k_map.filtered_dict, "amplitude")
             

 #     #this have to be a only one runite that can be share between the samplers.
        #     dict_ = {}
        #     for k, k_map in self.kinds_map.items():
        #         if k not in ['fe', 'continuum']:
        #             idx_amplitude = mapping_params(k_map.filtered_dict, "amplitude")
        #             idx_fwhm = mapping_params(k_map.filtered_dict, "fwhm")
        #             idx_center = mapping_params(k_map.filtered_dict, "center")
                    
        #             norm_amplitude = full_samples[:, idx_amplitude]
        #             fwhm = full_samples[:, idx_fwhm]
        #             center = full_samples[:, idx_center]
        #             flux = calc_flux(norm_amplitude, fwhm)
        #             fwhm_kms = calc_fwhm_kms(fwhm, self.c, center)
        #             L_line = calc_luminosity(self.d[n], flux, center)
        #             dict_[k] = {
        #                 'lines': k_map.line_name,
        #                 "component": jnp.array(k_map.component),
        #                 'flux': flux, "fwhm": fwhm, "fwhm_kms": fwhm_kms, "L": L_line,
        #                 'center': center, 'amplitude': norm_amplitude
        #             }

        #     L_w, L_bol = {}, {}
        #     wavelenghts = [1350.0, 1450.0, 3000.0, 5100.0, 6200.0]
        #     # Assume 'continuum' is present
        #     #idx_cont = mapping_params(self.params_dict, "scale")  # Adapt if needed
        #     #profile_func = self.estimator.RegionMap.profile_functions_combine
        #     map_cont = self.kinds_map['continuum']
        #     profile_func = map_cont.profile_functions_combine
        #     idx_cont = jnp.array(list(map_cont.filtered_dict.values()))
            
        #     cont_params = full_samples[:, idx_cont]
        #     for w in wavelenghts:
        #         wave = str(int(w))
        #         hits = jnp.isclose(norm_spec[n, 0, :], jnp.array([w]), atol=1)
        #         valid = (hits & (~self.mask[n])).any()
        #         corr = BOL_CORRECTIONS.get(wave, 0.0)
        #         if valid:
        #             flux_at_w = vmap(profile_func, in_axes=(None, 0))(jnp.array([w]), cont_params).squeeze()
        #             Lw = calc_monochromatic_luminosity(self.d[n], flux_at_w, w)
        #             L_w[wave] = Lw
        #             L_bol[wave] = calc_bolometric_luminosity(Lw, corr)
        #         else:
        #             L_w[wave] = jnp.zeros(full_samples.shape[0])
        #             L_bol[wave] = jnp.zeros(full_samples.shape[0])

        #     # --- Compute black hole masses ---
        #     dict_broad = dict_.get("broad")
        #     masses = {}
        #     if dict_broad is not None:
        #         fwhm_kms = dict_broad.get('fwhm_kms')
        #         line_name_list = np.array(dict_broad["lines"])
        #         for line_name, estimator in SINGLE_EPOCH_ESTIMATORS.items():
        #             wave = estimator["wavelength"]
        #             if line_name not in line_name_list or wave not in L_w:
        #                 continue
        #             idx_broad = list(jnp.where(line_name == line_name_list)[0])
        #             Lwave = L_w[wave]
        #             fwhm_kms_ = fwhm_kms[:, idx_broad].squeeze()
        #             masses[line_name] = calc_black_hole_mass(Lwave, fwhm_kms_, estimator)
            
        #     results_L_w.append(L_w)
        #     results_L_bol.append(L_bol)
        #     results_masses.append(masses)
            
        # return results_L_w, results_L_bol, results_masses