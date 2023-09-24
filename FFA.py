import sys
from FFA_utils import *


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#                         1) Dynamic Programming Class
#
#	This class holds the class that practically deploys the FFA procedure
# 	It is built such that extending it to a broader scope (acceleration
#	terms, for example) is possible. It therefore requires an elaborated
#   initialization stage. A short initialization script is included in this
#   file as well (see below).
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
class DynamicProgramming(object):
	# ----------------------------------------------------------------------------
	#                   Initialize DynamicProgramming object
	# ----------------------------------------------------------------------------
	def __init__(self, data, initialize_func, param_limits, param_stepping_func, resolve_params_func, addition_func,
				 shift_func, packing_func, brute_length,dt, data_type):
		'''
		The Dynamic programming class initialization. Input is described in the code below.
		The input to this preliminary initialization stage is defined in the initialization script.
		'''

		# Store the input data and data type:
		self.data = data
		self.data_type = data_type

		# A function that receives the data and builds the FFA data structure.
		# This function can be simple (wrapper for the data) or more complicated (RFI cleaning?).
		# This allows to use acceleration and periodicity with the same class,
		# and also allows for packing in the initializing function.
		# Default: brute-force folding (see below).
		self.initialize_func = initialize_func

		# The addition rule.
		# Default: simple addition: x1 + x2.
		self.addition_func = addition_func

		# This will allow for bit packing when possible.
		# Will be able to enforce a repacking every iteration.
		# Default: naive identity: x --> x without any packing.
		self.packing_func = packing_func

		# The shifting operation may not be that trivial:
		# for example, how do you handle shifts by non-integer amounts of bins?
		# therefore, it might be important to leave this for outside implementation.
		# Default: a jitted equivalent of the numpy roll about axis function.
		self.shift_func = shift_func

		# this function will determine the required precision of the parameters used.
		# for example, dp, da, dP, de, ...
		# Default: we only use the period (see periodicity_stepping below)
		self.param_stepping_func = param_stepping_func

		# For a given parameter, we need to obtain the index of the next one
		# for the iterations to progress. Note that in this case the 'params'
		# is actually only the period.
		self.resolve_params_func = resolve_params_func

		# a list of tuples with min and max values for all the enumerated params
		# An (incorrect) assumption is that the parameter resolution is independent of the parameter value.
		# This assumption is required for computational efficiency
		self.param_limits = param_limits

		# n_params is assumed to be non-changing during the expansion procedure.
		# If you wish to change parameter set, or introduce another param, make another instance of the class.
		# Use the previous class for initialization.
		self.n_params = len(self.param_limits)

		# We need to add two sections of data and refine the period list.
		# This is done using the unify data structure, is a function that depends on the resolve_params, shift, and
		# addition funtions. In order to geterate and apply Nubma, we need to define a `factory' that will generate
		# the jitted funtion outside and send it in...
		self.unify_data_structure_func = unify_data_structure_factory(resolve_params_func, shift_func, addition_func)

		# data structure is assumed to be completely held in memory throughout the algorithm.
		# if large data structures are required to handle, it is only with the pruning class.
		# This is good assuming the speed difference between the classes is less than the time it takes to load data
		# from slow memory
		self.initialized    = False
		self.data_structure = np.zeros(0, self.data_type)
		self.iter_num       = False
		self.param_list     = []
		self.brute_length   = brute_length
		self.dt = dt

	# ----------------------------------------------------------------------------
	#              Initialize the DynamicProgramming data structure
	# ----------------------------------------------------------------------------
	def init_data_structure(self):
		# will be useful also for non-destructive start overs (after changing one of the logic functions)
		self.iter_num = 0

		# Well, this practically sends the input data to the
		# brute-force folding routine.
		tmp = self.packing_func(
			self.initialize_func(self.data),
			self.iter_num)

		# Now from the initialilzed (brute-force folded) data
		# we read the data structure, list of trial periods used
		# and track the duration of the analyzed lightcurve chunk.
		self.data_structure = tmp[0].astype(self.data_type)
		self.param_list     = tmp[1]
		self.chunk_duration = self.brute_length * self.dt

	# ----------------------------------------------------------------------------
	#                           Do a single iteration
	# ----------------------------------------------------------------------------
	def perform_iteration(self):
		self.iter_num += 1

		# Read the relevant info from
		n_chunks             = len(self.data_structure)
		param_steppings      = self.param_stepping_func(self.iter_num)
		self.param_steppings = param_steppings

		n_params = len(param_steppings)
		# Avoiding recursion because numba does not support recursion

		new_param_list = [range_param(self.param_limits[i][0],self.param_limits[i][1],param_steppings[i]) for i in range(n_params)]
		self.new_param_list = new_param_list
		# zero is assumed to be the chunk_param index. then n_params regular params (acc,period, etc)
		# then the phase param, which is handled differently (matched filtering, binning, half-bin shifting and constant precision)
		other_dims = self.data_structure.shape[n_params+1:]

		final_shape = [n_chunks//2]+ list(map(len,new_param_list)) + list(other_dims)
		n_param_sets = np.prod(list(map(len,new_param_list)))
		# print(list(map(len,new_param_list))) !!! Commented: Sahar Shahaf 8/4/2019
		# accessing using an array is not supported by numba...
		new_data_structure = np.zeros([n_chunks//2]+ [n_param_sets] + list(other_dims),dtype=self.data_type)

		new_param_length_list = np.array([np.arange(len(l)) for l in new_param_list])
		# The reason that this function is made in a factory and used this way is for jit...

		new_param_list_cartesian = cartesian_prod(new_param_list)
		new_ind_list_cartesian = cartesian_prod(new_param_length_list)
		# numba cannot receive a list of arrays with different lengths as an argument.
		# the arrays are used only for lookup, and therefore padding them with inf is not destructive.
		padded_param_array = pad_with_inf(self.param_list)
		# print("final shape =",final_shape) !!! Commented: Sahar Shahaf 8/4/2019
		new_data_structure = self.unify_data_structure_func(self.data_structure, new_data_structure, new_param_list_cartesian, padded_param_array, self.iter_num, n_chunks)
		# make sure the copy happens correctly.
		self.data_structure = new_data_structure.reshape(final_shape)
		self.param_list = new_param_list
		self.chunk_duration *= 2

	def do_iterations(self, n_iters = 'max'):
		#t = time.time()
		if n_iters == 'max':
			n_iters = int(np.log2(len(self.data_structure)))
		sys.stdout.write("\n")
		for i in range(n_iters):
			n = int((10+1) * float(i) / n_iters)
			sys.stdout.write("    Iterating: [{0}{1}] - Data Dimensions: {2} \r".format('#' * n, ' ' * (10 - n), self.data_structure.shape))
			# print("performing iteration: ", i, flush=True)
			# print("data structure dimensions", self.data_structure.shape)
			self.perform_iteration()
		# print ("elapsed time:", time.time()-t)
		# sys.stdout.flush()
		sys.stdout.write("\r\x1b[2K\r\x1b[1A\r\x1b[2K\r")  
		return


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#                          2) Initialization script
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def FFA_init(time_series_e, time_series_v, dt, p_min, p_max,
			 n_bins, tolerance_bins, data_type,
			 addition_function, shift_function):
	'''
	Initialization script for FFA:
	------------------------------
	This function is merely a configuration script, configuring the dynamicProgramming class

	periodicity_search_dynamic_programming(time_series_e, time_series_v, dt, p_min, p_max, n_bins, tolerance_bins,
										   data_type = np.float32, brute_force_fold_func=None,
										   addition_function=default_addition_function, shift_function = default_shift_function)

	time_series_e should already be "E/V" weighted
	time_series_v should already represent "E^2/V". these are anyway the e and v parts of the signal...
	Moreover, time_series_e should already be bias subtracted (note that this causes negative numbers in folding)
	To mitigate this, you can aim at having an "unbiased" pulsar at the detection limit.
	'''

	# Preliminary initialization
	original_data = np.array([time_series_e, time_series_v])
	packing_func  = default_packing_function
	param_limits  = [[p_min, p_max]]

	# Given the number of points in the light curve and the lenfth of
	# the lightcurve guven the `typical' timestep. W use this to set
	# the integer k, representing the nearest log2-length lightcurve.
	total_length = len(time_series_e)
	k            = int(np.log2(total_length*dt/p_max) - 1)
	brute_length = int(total_length / 2**k)

	# this function will initialize the data structure
	@jit(nopython=True)
	def initialize_func(data):
		return brute_force_fold(data[0], data[1], p_min, p_max, brute_length, int(n_bins), dt, tolerance_bins)

	# this function will determine the required precision of the period grid, dP
	@jit(nopython=True)
	def param_stepping_func(iter_num):
		chunk_duration = 2**iter_num*brute_length * dt
		return (periodicity_stepping(chunk_duration, p_min, tolerance_bins*dt), )

	# For a given parameter, we need to obtain the index of the next one
	# for the iterations to progress. Note that in this case the 'params'
	# is actually only the period. This function is defined in here, and
	# not somewhere below because we need the brute length value.
	@jit(nopython = True)
	def resolve_params_func(param_set, param_list, iter_num, latter):
		new_p       = param_set[0]
		old_p_index = find_nearest_sorted_idx(param_list[0], new_p)

		# this calc corresponds to a shift right. (as performed by the roll function):
		# multiplication by latter is for nulling the phase. it is a different implementation of if.
		relative_phase = int(round(((2**(iter_num) * brute_length * dt) % new_p) / new_p * n_bins))%n_bins

		return old_p_index, relative_phase*latter

	# Now, initialize the DynamicProgramming structure to do FFA:
	ffa = DynamicProgramming(original_data, initialize_func, param_limits, param_stepping_func, resolve_params_func,
							 addition_function, shift_function, packing_func, brute_length,dt, data_type)
	return ffa

