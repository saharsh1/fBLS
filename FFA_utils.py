import numpy as np
from numba import jit
from numpy import zeros, array, sqrt, mean,std, roll, log2, argmax, sin, cos
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#                         3) Brute-force folding
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@jit(nopython = True)
def brute_force_fold(time_series_flux, time_series_flux_err,
					 period_min, period_max, brute_length,
					 n_bins, dt, tolerance_bins=0.5):
	'''
	Here, we set the length for the brute-force folded profiles.
	Later, you can handle the situation in which p_max/p_min is larger than 2. for the mean-time, let it be.
	You can also transfer to frequency space later (code should be generic enough)
	'''

	# The number of chunks is the number of brute-length sections
	# that fit inside the lightcurve. The number of jumps is the number of
	# trual periods that fit in one brute-length section.
	n_chunks     = int(np.ceil(len(time_series_flux) / float(brute_length)))
	n_jumps      = int(brute_length*dt / period_min)

	# Set the phase and period resolution,
	# nad generate a period list.
	dphi         = tolerance_bins / float(n_bins)
	dp           = dphi * period_min / float(n_jumps)
	period_list  = np.arange(period_min, period_max, dp)
	n_periods    = len(period_list)

	# Initialize the data structure
	data_structure  = np.zeros((n_chunks, n_periods, 4, n_bins))
	signal_ind_list = np.arange(0, len(time_series_flux), brute_length)

	for period_ind, trial_period in enumerate(period_list):

		proper_time = np.arange(brute_length) * dt
		tmp = [(x % trial_period)/ float(trial_period)*n_bins for x in proper_time]
		ind_arr = round_arr(tmp) % n_bins
		for chunk_ind, signal_ind in enumerate(signal_ind_list):
			signal_e = time_series_flux[signal_ind:signal_ind+brute_length]
			signal_v = time_series_flux_err[signal_ind:signal_ind+brute_length]

			data_structure[chunk_ind, period_ind, :2] = fold_modulated_signal(signal_e, signal_v,
																			  ind_arr[:len(signal_e)],
																			  n_bins, trial_period, True)
	return data_structure, (period_list,)




# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
#                                 Auxil functions
# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
@jit(nopython = True)
def default_packing_function(x, iter_num):
	return x

@jit(nopython = True)
def default_addition_function(data0, data1):
	return data0 + data1

@jit(nopython = True)
def default_shift_function(data, phase_shift):
	# should do np.roll, but this is not supported in numba...
	# return np.roll(data, phase_shift, axis=1)
	output_data = np.empty_like(data)
	n = output_data.shape[1]
	phase_shift = phase_shift%n
	#jj_arr = (np.range(n) + phase_shift)%n
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			jj = j + phase_shift
			if jj >= n:
				jj = jj-n
			output_data[i,jj] = data[i,j]
	return output_data

@jit(nopython=True)
def periodicity_stepping(total_time,p_min, tolerance):
	n_jumps = (total_time / p_min)
	opt_dp = tolerance / (n_jumps/2.)
	return opt_dp

@jit(nopython=True)
def find_nearest_sorted_idx(sorted_array,value):
	# works only for sorted arrays!
	if len(sorted_array) == 1:
		return 0
	left = 0
	right = len(sorted_array)-1
	middle = (left + right)//2
	while middle != left and middle != right:
		if (sorted_array[middle] - value) > 0:
			right = middle
			middle = (left + right)//2
		else:
			left = middle
			middle = (left+right)//2
	if abs(sorted_array[middle] - value) <= abs(sorted_array[middle+1] - value):
		return middle
	else:
		return middle+1

@jit(nopython = True)
def round_arr(arr):
	output = np.zeros(len(arr),dtype=np.uint32)
	for i,x in enumerate(arr):
		output[i] = np.uint32(round(x))
	return output

def unify_data_structure_factory(resolve_params_func, shift_func, addition_func):

	# Must make jit understand that all functions are defined in it's scope when it compiles.
	@jit(nopython = True)
	def unify_iteration(data_structure_input, data_structure_output, new_param_list_cartesian, padded_param_array , iter_num, n_chunks):

		for param_set_index in range(len(new_param_list_cartesian)):
			param_set = new_param_list_cartesian[param_set_index]
			# iter_num-1 is used because we are looking for indices in the data structure from the previous iteration
			old_params_ind_0, phase_shift0 = resolve_params_func(param_set, padded_param_array, iter_num - 1, 0)
			old_params_ind_1, phase_shift1 = resolve_params_func(param_set, padded_param_array, iter_num - 1, 1)
			#tuple_coord = tuple(new_ind_list_cartesian[param_set_index])
			for pair_index in range(n_chunks / 2):
				fold0 = shift_func(data_structure_input[pair_index * 2][old_params_ind_0], phase_shift0)
				fold1 = shift_func(data_structure_input[pair_index * 2 + 1][old_params_ind_1], phase_shift1)
				#TODO - the data access to a tuple index is the last nopython problem
				data_structure_output[pair_index][param_set_index] = addition_func(fold0, fold1)
		return data_structure_output
	return unify_iteration

# Folding a Modulated signal:
@jit(nopython=True)
def fold_modulated_signal(signal_e, signal_v, proper_time, n_bins, pulsar_period, ind_arr_bool = False):
	res_e = zeros(n_bins)
	res_v = zeros(n_bins)
	ind_arr = zeros(len(signal_e), dtype=np.uint32)
	if not ind_arr_bool:
		# XXX the famous bug!  The one that can cause false results!
		ind_arr = round_arr((proper_time % pulsar_period) / pulsar_period * n_bins)%np.uint32(n_bins)
	else:
		# this option is strictly for computational efficiency purposes.
		# idiotic type conversion because of numba's incompatability with type conversion in any other way
		for i in range(len(ind_arr)):
			ind_arr[i] = np.uint32(proper_time[i])
	res_e, res_v = fold_indices(res_e, res_v, signal_e, signal_v, ind_arr)
	#if ret_ev:
	output = np.zeros((2,len(res_e)))
	output[0] = res_e
	output[1] = res_v
	return output
#else:
# Returning the result in units of sigmas. Use with caution.
#   return res_e / (sqrt(res_v))


# Correctly stepping a parameter is a non-trivial ugly question...
# TODO: Make sure this is right another time.
def range_param(vmin,vmax,dv):
	if dv > (vmax - vmin)/2:
		return np.array([(vmax+vmin)/2])
	else:
		n = int((vmax - vmin) / dv) # rounding down to get the number of points
		return np.linspace(vmin,vmax,n+2)[1:-1]


def cartesian_prod(L):
	total_length = np.prod(list(map(len,L)))
	return np.array(np.meshgrid(*L, indexing='ij')).transpose(list(range(1, len(L) + 1)) + [0]).reshape([total_length,len(L)])

# TODO replace this function with np.add.at
@jit(nopython = True)
def fold_indices(res_e_v, res_e2_v, signal_e_v, signal_e2_v, ind_arr):
	# This function folds the components of the input array signal_e_v and signal_e2_v
	# into the output arrays
	for i in range(len(signal_e_v)):
		res_e_v[ind_arr[i]] += signal_e_v[i]
		res_e2_v[ind_arr[i]] += signal_e2_v[i]
	return res_e_v, res_e2_v


def pad_with_inf(param_list):
	maxlen = np.max(list(map(len,param_list)))
	output = np.zeros([len(param_list),maxlen])
	output += np.inf
	for i,l in enumerate(param_list):
		output[i][:len(l)] = l
	return output