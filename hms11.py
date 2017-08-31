#!/usr/bin/env python

#This is a Python-translated version of http://www.astro.princeton.edu/~gk/A403/hms11.f
#With some modifications to variable names for clarity.

#This is a naive Python version relying on numpy.

import numpy as np
import traceback

def sch(X,Z,M,log_Teff,log_L,log_T_center,log_rho_center,
	M_fit_frac=0.3,
	max_iterations=1000,
	max_difference=1e-4,
	perturb_log_BC=1e-4,
	N_frac=0.3,
	M_central_frac=1e-4,
	outer_boundary_T_frac=8.40896e-1,
	outer_boundary_rho=1e-12,
	outer_boundary_R_frac=1.,
	max_M_diff=1e-5,
	k_max=1000,
	perturb_max_log_Teff = 0.03,
	perturb_max_log_L = 0.1,
	perturb_max_log_T_center = 0.01,
	perturb_max_log_rho_center = 0.1,
	output_filename = None,
	roundoff_for_file = 6):
	"""Calculates chemically homogeneous stellar model in a thermal equilibrium, i.e. a model on a zero age main sequence fitting results of integrations of envelope and core at the fitting mass, M_fit = M_fit_frac*M.
	X: H content by mass fraction
	Z: Metal content by mass fraction
	M: Total stellar mass (M_sun)
	log_Teff: Guess log10 of effective temperature (K)
	log_L: Guess log10 of surface luminosity (L_sun)
	log_T_center: Guess log10 of central temperature (K)
	log_rho_center: Guess log10 of central density (g/cm**3)
	M_fit_frac: Fraction of total stellar mass at which a fitting point is used. Default = 0.3
	max_iterations: Maximum number of iterations. Default = 15
	max_difference: Maximum allowable fractional difference at fitting point. Default = 1e-4
	perturb_log_BC: Magnitude of the perturbations of the logarithms of the boundary conditions. Default = 1e-4
	N_frac: Nitrogen content, as a fraction of total metals. Z_N = N_frac*Z. Default = 0.3
	output_filename: Target filename for output of data.
	- The file output_filename.model is an ASCII file with a stellar interior for the last model run and will have columns of:
	* M [Msun]
	* T [K]
	* rho [g/cm^3]
	* R [Rsun]
	* L [Lsun]
	* X [mass fraction]
	* Z [mass fraction]
	* Z_N [mass fraction]
	For each grid point stepped through.
	- The file output_filename.converge is an ASCII file representing the intermediate models and will have columns of:
	* iteration_number
	* convergence_diff (maximum difference of state variables, convergence issues)
	* log_Teff [log K]
	* log_L [log Lsun]
	* log_T_center
	* log_rho_center [log g/cm^3]
	* log_R [log Rsun]
	* Teff [K]
	* L [erg/s]
	* T_center [K]
	* rho_center [g/cm^3]
	* R [cm]
	roundoff_for_file: The number of decimals to save for each variable in the output file.
	By default, prints to standard output.
	"""
	Z_N = N_frac*Z #Nitrogen content for CNO burning
	M_fit = M_fit_frac*M #Set the fitting point
	#diff_matrix = np.zeros((4,4))
	diff_matrix = np.zeros((4,5))
	#ord_vector = np.zeros(4)
	if output_filename is not None:
		converge_output_col_names = ["iteration_number","convergence_diff","log_Teff [log K]","log_L [log Lsun]","log_T_center","log_rho_center [log g/cm^3]","log_R [log Rsun]","Teff [K]","L [erg/s]","T_center [K]","rho_center [g/cm^3]","R [cm]"]
		converge_output_cols = np.empty((0,len(converge_output_col_names)), dtype=float)
		model_output_col_names = ["M [Msun]","T [K]","rho [g/cm^3]","R [Rsun]","L [Lsun]","X [mass fraction]","Z [mass fraction]","Z_N [mass fraction]"]
		model_output_cols = np.empty((0,len(model_output_col_names)), dtype=float)
	try:
		for iter_counter in range(max_iterations):
			envel_vec, log_R = envel(log_Teff,log_L,X,Z,Z_N,M_fit,M,outer_boundary_T_frac=outer_boundary_T_frac,outer_boundary_rho=outer_boundary_rho,outer_boundary_R_frac=outer_boundary_R_frac,k_max=k_max,max_M_diff=max_M_diff)
			core_vec = core(log_T_center, log_rho_center, X, Z, Z_N, M_fit, M_central_frac=M_central_frac,k_max=k_max,max_M_diff=max_M_diff)
			envel_vec_trimmed = envel_vec[0:4]
			core_vec_trimmed = core_vec[0:4]
			difmax = 0.
			delfit = envel_vec_trimmed/core_vec_trimmed - 1.
			difference = np.max(np.abs(delfit))
			if difference < max_difference:
				print "Iteration", iter_counter+1,": Convergence reached!",difference,"<",max_difference
				print "log_Teff, log_L, log_T_center, log_rho_center, log_R, iter_counter"
				print log_Teff, log_L, log_T_center, log_rho_center, log_R, iter_counter+1
				break
			else:
				envel_vec1, log_R = envel(log_Teff+perturb_log_BC,log_L,X,Z,Z_N,M_fit,M,outer_boundary_T_frac=outer_boundary_T_frac,outer_boundary_rho=outer_boundary_rho,outer_boundary_R_frac=outer_boundary_R_frac,k_max=k_max,max_M_diff=max_M_diff)
				envel_vec2, log_R = envel(log_Teff,log_L+perturb_log_BC,X,Z,Z_N,M_fit,M,outer_boundary_T_frac=outer_boundary_T_frac,outer_boundary_rho=outer_boundary_rho,outer_boundary_R_frac=outer_boundary_R_frac,k_max=k_max,max_M_diff=max_M_diff)
				core_vec1 = core(log_T_center+perturb_log_BC, log_rho_center, X, Z, Z_N, M_fit, M_central_frac=M_central_frac,k_max=k_max,max_M_diff=max_M_diff)
				core_vec2 = core(log_T_center, log_rho_center+perturb_log_BC, X, Z, Z_N, M_fit, M_central_frac=M_central_frac,k_max=k_max,max_M_diff=max_M_diff)
				for i in range(4):
					diff_matrix[i,0] = ((envel_vec1[i]/core_vec[i]-1.)-delfit[i])/perturb_log_BC
					diff_matrix[i,1] = ((envel_vec2[i]/core_vec[i]-1.)-delfit[i])/perturb_log_BC
					diff_matrix[i,2] = ((envel_vec[i]/core_vec1[i]-1.)-delfit[i])/perturb_log_BC
					diff_matrix[i,3] = ((envel_vec[i]/core_vec2[i]-1.)-delfit[i])/perturb_log_BC
					#ord_vector[i] = delfit[i]
					diff_matrix[i,4] = delfit[i]
				#perturb = np.linalg.solve(diff_matrix, ord_vector)
				perturb, facdel = solve(diff_matrix)
				perturb = perturb#*(np.random.random((4))*1.5+0.5)
				#print "perturb_log_Teff", perturb[0], perturb_max_log_Teff, np.sign(perturb[0])*np.min([np.abs(perturb[0]), perturb_max_log_Teff])
				log_Teff += np.sign(perturb[0])*np.min([np.abs(perturb[0]), perturb_max_log_Teff])
				log_L += np.sign(perturb[1])*np.min([np.abs(perturb[1]), perturb_max_log_L])
				log_T_center += np.sign(perturb[2])*np.min([np.abs(perturb[2]), perturb_max_log_T_center])
				log_rho_center += np.sign(perturb[3])*np.min([np.abs(perturb[3]), perturb_max_log_rho_center])
				print "Iteration", iter_counter+1,": Converging.",difference,">=",max_difference
				print "log_Teff, log_L, log_T_center, log_rho_center, log_R, iter_counter"
				print log_Teff, log_L, log_T_center, log_rho_center, log_R, iter_counter+1
				print "==="
			if output_filename is not None:
				converge_new_row = np.zeros(len(converge_output_col_names))+np.nan
				converge_new_row[0] = iter_counter+1
				converge_new_row[1] = difference
				converge_new_row[2] = log_Teff
				converge_new_row[3] = log_L
				converge_new_row[4] = log_T_center
				converge_new_row[5] = log_rho_center
				converge_new_row[6] = log_R
				converge_new_row[7] = np.power(10.,log_Teff)
				converge_new_row[8] = np.power(10.,log_L)*3.82875e33
				converge_new_row[9] = np.power(10., log_T_center)
				converge_new_row[10] = np.power(10., log_rho_center)
				converge_new_row[11] = np.power(10., log_R)*6.95700e10
				converge_output_cols = np.vstack([converge_output_cols, converge_new_row])
	except:
		traceback.print_exc()
	finally:
		if output_filename is not None:
			envel_vec, log_R, model_output_cols = envel(log_Teff,log_L,X,Z,Z_N,M_fit,M,
			outer_boundary_T_frac=outer_boundary_T_frac,outer_boundary_rho=outer_boundary_rho,outer_boundary_R_frac=outer_boundary_R_frac,k_max=k_max,max_M_diff=max_M_diff,
			model_output=True, model_array=model_output_cols)
			core_vec, model_output_cols = core(log_T_center, log_rho_center, X, Z, Z_N, M_fit, M_central_frac=M_central_frac,k_max=k_max,max_M_diff=max_M_diff,model_output=True,model_array=model_output_cols)
			model_output_cols = model_output_cols[model_output_cols[:,0].argsort()]
			with open(output_filename+".model",'w') as f:
				f.write("\t".join(model_output_col_names)+"\n")
				np.savetxt(f, model_output_cols, fmt="%."+str(roundoff_for_file)+"E",delimiter="\t")
			with open(output_filename+".converge",'w') as f:
				f.write("\t".join(converge_output_col_names)+"\n")
				np.savetxt(f, converge_output_cols, fmt="%."+str(roundoff_for_file)+"E",delimiter="\t")
			print "Finished writing to files."
		if difference > max_difference:
			raise RuntimeError("Failed, convergence problems. Difference: %s, Max Allowable Difference: %s." % (difference, max_difference))
		return log_Teff, log_L, log_T_center, log_rho_center, log_R, iter_counter+1

def envel(log_Teff,log_L,X,Z,Z_N,M_fit,M,outer_boundary_T_frac=8.40896e-1,outer_boundary_rho=1e-12,outer_boundary_R_frac=1.,k_max=1000,max_M_diff=1e-5,model_output=False,model_array=None):
	"""Integrates stellar structure equations from the surface to the fitting point for a chemically homogeneous star.
	outer_boundary_T_frac is the temperature at the outer boundary as a fraction of T_phot. Nominally it is 2^(-1./4.).
	outer_boundary_rho is the density at the outer boundary in g/cm^3. Nominally it is 10^-12 g/cm^3.
	outer_boundary_R_frac is the radius at the outer boundary as a fraction of R_phot. Nominally it is 1, if the atmosphere is geometrically thin.
	k_max is the total number of allowable steps before envel gives up.
	max_M_diff is the total absolute difference, in mass coordinate, allowed between the converged integration point and the fit mass point, in Msun. Nominally 1e-5."""
	solar_Teff = 5772.

	T_eff = np.power(10.,log_Teff)
	T_phot = T_eff #Set the photospheric temperature equal to the effective temperature
	L_phot = np.power(10.,log_L)
	R_phot = np.sqrt(L_phot)/np.square(T_phot/solar_Teff) #With L_sol/T_sol^4 = 4*pi*sigma
	log_R = np.log10(R_phot)
	in_vec = np.zeros(8)
	#Set the outer boundary BC's far from the star

	in_vec[0] = T_phot*outer_boundary_T_frac
	in_vec[1] = outer_boundary_rho
	in_vec[2] = R_phot*outer_boundary_R_frac
	in_vec[3] = L_phot
	in_vec[4] = M
	in_vec[5] = X
	in_vec[6] = Z
	in_vec[7] = Z_N
	if model_output:
		model_array = np.vstack([model_array, [in_vec[4], in_vec[0], in_vec[1], in_vec[2], in_vec[3], in_vec[5], in_vec[6], in_vec[7]]])
		for k in range(k_max):
			in_vec = step(in_vec, M_fit)
			model_array = np.vstack([model_array, [in_vec[4], in_vec[0], in_vec[1], in_vec[2], in_vec[3], in_vec[5], in_vec[6], in_vec[7]]])
			if np.abs(in_vec[4]/M_fit - 1.) < max_M_diff:
				return in_vec, log_R, model_array
	else:
		for k in range(k_max):
			in_vec = step(in_vec, M_fit)
			if np.abs(in_vec[4]/M_fit - 1.) < max_M_diff:
				return in_vec, log_R
	raise RuntimeError("envel failed because it failed to reach M_fit in the required iterations. Try increasing k_max and/or max_M_diff. np.abs(in_vec[4]/M_fit - 1.)=%s , max_M_diff = %s" % (np.abs(in_vec[4]/M_fit - 1.), max_M_diff))

def core(log_T_center, log_rho_center, X, Z, Z_N, M_fit, M_central_frac=1e-4,k_max=1000,max_M_diff=1e-5,model_output=False,model_array=None):
	"""Integrates stellar structure equations from the center to the fitting point for a chemically homogeneous star.
	M_central_frac is the fraction of M_fit that the core starts from the interior.
	k_max is the total number of allowable steps before core gives up.
	max_M_diff is the total absolute difference, in mass coordinate, allowed between the converged integration point and the fit mass point, in Msun. Nominally 1e-5."""
	sun_rho_x3 = 4.69960e-1 #3*avg density of sun. aka sunmr3
	sunml = 5.19373e-1
	T_central = np.power(10.,log_T_center)
	rho_central = np.power(10.,log_rho_center)
	M_central = M_central_frac * M_fit
	eps_tot, H_depletion_rate = nburn(rho_central,T_central,X,Z_N)
	R_central = np.power(M_central/rho_central*sun_rho_x3*3., 1./3.) #in solar radii
	L_central = sunml*M_central*eps_tot #in solar luminosities
	in_vec = np.zeros(8)
	in_vec[0] = T_central
	in_vec[1] = rho_central
	in_vec[2] = R_central
	in_vec[3] = L_central
	in_vec[4] = M_central
	in_vec[5] = X
	in_vec[6] = Z
	in_vec[7] = Z_N
	if model_output:
		model_array = np.vstack([model_array, [in_vec[4], in_vec[0], in_vec[1], in_vec[2], in_vec[3], in_vec[5], in_vec[6], in_vec[7]]])
		for k in range(k_max):
			in_vec = step(in_vec, M_fit)
			model_array = np.vstack([model_array, [in_vec[4], in_vec[0], in_vec[1], in_vec[2], in_vec[3], in_vec[5], in_vec[6], in_vec[7]]])
			if np.abs(in_vec[4]/M_fit - 1.) < max_M_diff:
				return in_vec, model_array
	else:
		for k in range(k_max):
			in_vec = step(in_vec, M_fit)
			if np.abs(in_vec[4]/M_fit - 1.) < max_M_diff:
				return in_vec
	raise RuntimeError("core failed because it failed to reach M_fit in the required iterations. Try increasing k_max and/or max_M_diff. np.abs(in_vec[4]/M_fit - 1.)=%s , max_M_diff = %s" % (np.abs(in_vec[4]/M_fit - 1.), max_M_diff))

def step(state_vec, M_fit, nuclear_on=True):
	"""Makes one integration step for a stellar core or envelope with a second-order Runge-Kutta method, the steps are carried up to the point M_fit"""
	deriv_vec = rhs(state_vec, nuclear_on=nuclear_on)
	deriv_vec_extended = np.pad(deriv_vec, (0,3), "constant")
	state_vec_trimmed = state_vec[0:5]
	acc = np.array([0.05,0.15,0.05,0.2,0.2])
	int_step_arr = deriv_vec/acc/state_vec_trimmed
	hi_step = np.max(int_step_arr)
	int_step = 1./hi_step
	if state_vec[4] > M_fit:
		int_step = -int_step
	if (state_vec[4] - M_fit)*(state_vec[4]+int_step-M_fit) < 0:
		int_step = M_fit - state_vec[4]
	half_int_step = 0.5*int_step
	midpoint_state_vec = state_vec + half_int_step * deriv_vec_extended
	deriv_vec = rhs(midpoint_state_vec, nuclear_on=nuclear_on)
	deriv_vec_extended = np.pad(deriv_vec, (0,3), "constant")
	state_vec = state_vec + int_step * deriv_vec_extended
	return state_vec #Return the new state vector.

def rhs(state_vec, nuclear_on=True):
	"""Calculates right hand sides of the stellar structure equations"""
	T, rho, R, L, M, X, Z, Z_N = state_vec #each row is one of these variables.
	P, dlnP_dlnT, dlnP_dlnrho, P_rad, P_gas, grad, Q_T, Q_R = eos(rho, T, X, Z)
	kap_tot = opact(rho,T,X,Z)
	if nuclear_on:
		eps_tot, H_depletion_rate = nburn(rho,T,X,Z_N)
	else: #If nuclear_on = False, turn nuclear burning off. TODO: add entropy, eps_grav.
		eps_tot = 0.
		H_depletion_rate = 0.
	#L_sun = 3.82875e33
	#M_sun = 1.98855e33
	#R_sun = 6.95700e10
	#C = 2.99792458e10
	#G = 6.67259e-8
	#sunmr3   = ( solar mass ) / ( 4*pi*(solar radius)**3 ) (cgs)
	const_grad_rad = 1.91485e-5 #L_sun/M_sun/(16*pi*C*G)
	sun_rho_x3 = 4.69960e-1 #3*avg density of sun. aka sunmr3
	sun_M_div_L = 5.1937316356513222331e-1 #sun mass/luminosity (cgs)
	cdpm= 8.96333428249013108715e14 #G/4/pi*(sunm/sunr**2)**2
	grad_rad = const_grad_rad * kap_tot * P / P_rad * L / M #d ln T / d ln P
	if grad > grad_rad:
		dlnT_dlnP = grad_rad
	else:
		dlnT_dlnP = grad
	dlnrho_dlnP = (1.-dlnT_dlnP*dlnP_dlnT)/dlnP_dlnrho
	dlnP_dM_coord = -cdpm*M/np.power(R,4.) #d ln P / d (Mr / Msun)
	deriv_vec = np.zeros(5)
	deriv_vec[0] = dlnT_dlnP*dlnP_dM_coord*T/P
	deriv_vec[1] = dlnrho_dlnP*dlnP_dM_coord*rho/P
	deriv_vec[2] = sun_rho_x3/rho/R/R
	deriv_vec[3] = sun_M_div_L*eps_tot
	deriv_vec[4] = 1.
	return deriv_vec

def eos(rho, T, X, Z):
	"""Calculates equation of state and thermodynamic quantities"""
	k_div_H = 8.2511e7 #k/M_H
	a_rad = 7.565912e-15 #8*pi*k^4/(15*h^3*c^3)
	K_nonrel = 9.9087e12 #non-relativistic degenerate electron coefficient: 1./20.*(3./pi)^(2./3.)*h^2/(m_e*m_H^(5/3))
	K_rel = 1.2309e15 #relativistic degenerate electron coefficient: 1./8.*(3./pi)^(1./3.)*h*c/(m_H)^(4/3)

	Y = 1.-X-Z
	mu_ion = 1./(X+Y/4.+Z/16.) #mean molecular weight of ions (Oxygen as representative of metals)
	mu_e = 2./(1.+X) #mean molecular weight per free electron for full ionization

	P_rad = a_rad * np.power(T,4.) #radiation pressure
	P_ion = k_div_H*rho*T/mu_ion #ion pressure
	P_e_rel_degen = K_rel*np.power(rho/mu_e,4./3.) #relativistic degenerate electron pressure
	P_e_nonrel_degen = K_nonrel*np.power(rho/mu_e,5./3.) #non-relativistic degenerate electron pressure
	P_e_degen = P_e_nonrel_degen/np.sqrt(1.+np.square(P_e_nonrel_degen/P_e_rel_degen))
	P_e_nondegen = k_div_H/mu_e*rho*T
	P_e_tot = P_e_nondegen*np.sqrt(1.+np.square(P_e_degen/P_e_nondegen))
	P_gas = P_ion + P_e_tot
	P = P_rad + P_gas
	f = 5./3.*np.square(P_e_degen/P_e_nonrel_degen)+4./3.*np.square(P_e_degen/P_e_rel_degen)
	dlnP_e_dlnT = np.square(P_e_nondegen/P_e_tot) #dlogP_e/dlogT at constant density
	dlnP_e_dlnrho = dlnP_e_dlnT + (1. - dlnP_e_dlnT)*f #dlogP_e/dlogrho at constant temperature

	dlnP_dlnT = (4.*P_rad + P_ion + dlnP_e_dlnT*P_e_tot)/P
	dlnP_dlnrho = (P_ion + dlnP_e_dlnrho*P_e_tot)/P
	Q_T = (12.*P_rad + 1.5 * P_ion + dlnP_e_dlnT/(f-1.)*P_e_tot)/rho #Q_T = T*C_V, C_V is specific heat at constant volume per gram.
	Q_R = P/rho*dlnP_dlnT #Q_R = Q_T * d log T / d log rho
	grad = 1./(Q_T/Q_R*dlnP_dlnrho + dlnP_dlnT) #dlnT/dlnP at constant entropy
	return P, dlnP_dlnT, dlnP_dlnrho, P_rad, P_gas, grad, Q_T, Q_R
	pass

def nburn(rho,T,X,Z_N):
	"""Calculates hydrogen burning rates for SCH and HEN programs, Oct. 22, 1984
	eps_tot: Energy generation rate due to PP and CNO burning (erg/g/s)
	H_depletion_rate: Hydrogen depletion rate (g/g/s)"""
	if T > 1e6:
		T613 = np.power(T/1e6, 1./3.)
		T623 = T613*T613
		eps_cno = X*Z_N*rho*np.exp(64.24 - 152.313/T613)/T623
		eps_pp = X*X*rho*np.exp(14.54 - 33.81/T613)/T623
		eps_tot = eps_cno + eps_pp
	else:
		eps_tot = 1e-30
	H_depletion_rate = -1.667e-19 * eps_tot
	return eps_tot, H_depletion_rate

def opact(rho,T,X,Z):
	"""Calculates opacity (radiative and conductive)"""
	A=6.
	B=1e-3
	C=1e-3

	Y = 1.-X-Z
	Z_av = X + 4.*Y + 8.*Z

	kap_e = 0.2*(1.+X)/(1.+2.7e11*rho/T/T)/(1.+(T/4.5e8)**0.86) #Electron scattering opacity (cm**2/g)
	kap_K = 12./A*(1.+X)*(1e-3+Z)*rho*(1e7/T)**3.5 #Kramer's (bound-free, free-free, bound-bound) opacity (cm**2/g)
	if T > 4e4:
		kap_H_ion = 1e10 #Negative H-ion opacity (cm**2/g)
		kap_molec = 1e-5 #Molecular opacity (cm**2/g)
	else:
		kap_H_ion = B*65.*np.sqrt(Z*rho)*np.power(T/3e3,7.7) #Negative H-ion opacity (cm**2/g)
		kap_molec = C*0.1*Z #Molecular opacity (cm**2/g)
	kap_rad = kap_molec + 1./(1./kap_H_ion + 1./(kap_e + kap_K)) #Radiative opacity (cm**2/g)
	if rho > 1e-5:
		kap_cond = Z_av * 2.6e-7 * np.square(T/rho) * (1.+np.power(rho/2e6,2./3.)) #Conductive opacity (cm**2/g)
	else:
		kap_cond = 1e10 #Conductive opacity (cm**2/g)
	kap_tot = 1./(1./kap_rad + 1./kap_cond) #Total opacity (cm**2/g)
	return kap_tot

def solve(deriv,perturb_max_log_Teff = 0.03,perturb_max_log_L = 0.1,perturb_max_log_T_center = 0.01,perturb_max_log_rho_center = 0.1):
	"""Solves n=4 linear algebraic equations"""
	delmax = np.zeros(4)
	delce = np.zeros(4)
	delmax[0] = perturb_max_log_Teff
	delmax[1] = perturb_max_log_L
	delmax[2] = perturb_max_log_T_center
	delmax[3] = perturb_max_log_rho_center
	n=3
	nm=n-1
	np_=n+1
	for k in range(0,nm+1):
	      kp=k+1
	      fac1=deriv[k,k]
	      for i in range(kp,n+1):
		      fac2=deriv[i,k]
		      for j in range(kp,np_+1):
		      	deriv[i,j]=deriv[i,j]*fac1-deriv[k,j]*fac2
	delce[n] = -deriv[n,np_]/deriv[n,n]
	for i in range(1,n+1):
		i1 = n-i
		i2 = i1+1
		delce[i1] = -deriv[i1,np_]
		for j in range(i2, n+1):
			delce[i1] = delce[i1] - deriv[i1,j]*delce[j]
		delce[i1] = delce[i1]/deriv[i1,i1]
	dm = 0.
	for i in range(0, n+1):
		d = np.abs(delce[i]/delmax[i])
		if dm < d:
			dm = d
	facdel = 1.
	if dm > 1.:
		facdel = dm
	for i in range(0, n+1):
		delce[i] = delce[i]/facdel
	facdel = dm
	return delce, facdel
