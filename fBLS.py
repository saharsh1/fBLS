# *****************************************************************************
#                   ------------------------------
#                          TransitSearch.py (class)
#                   ------------------------------
# This file defines the "FastBLS" class. An object of this class stores
# the measured flux, their corresponding uncertainties, and measurement times.
# See more details under __init__.
#
# A FastBLS class stores the following methods:
# ---------------------------------------------
# GenerateTrialPeriods - Create a list of trial periods.
# RunBLS - Run BLS search for the trial periods.

# External Dependencies:
# os, sys, numpy, matplotlib, numba and
# dynamic_programming_pulsar_search as dpps
# ******************************************************************************

import matplotlib.pyplot as plt, matplotlib.cm as cm, sys
import time

from FFA import *
from numba import njit
from tqdm import tqdm as tqdm_terminal
from tqdm.notebook import tqdm as tqdm_notebook

# Check if the code is running inside a Jupyter notebook
is_notebook = 'ipykernel' in sys.modules

# Use the appropriate tqdm version based on the environment
tqdm = tqdm_notebook if is_notebook else tqdm_terminal

# warnings.filterwarnings("ignore", category=RuntimeWarning)


#           ******************************************************************************
#           ******************************************************************************
#           ******************************************************************************
#                          1)     Definition of the fBLS class.
#           ******************************************************************************
#           ******************************************************************************
#           ******************************************************************************
class fBLS:

    def __init__(self, time_axis, flxs, flux_errs):
        '''
        Input: time_axis - Numpy array. Times of measurements.
               fluxes - Numpy array. Measured fluxes.
               flux_errs - Numpy array. Estimated uncertainties in the flux
               TargetName - string (optional)

        Output: All the input data, along with an evenly sampled grid data
                and the grid resolution dt.
        '''

        # We need to arrange the data on an evenly sampled grid.
        # Set a small enough dt, and bin the corresponding datapoints into it.
        # We assume that dt is small enough so that only one point falls at each bin,
        # at most (we can have empty bins)

        dt = np.median(np.diff(time_axis))
        grid_time = np.arange(np.min(time_axis), np.max(time_axis)+dt, dt)
        grid_flux = np.zeros(len(grid_time))
        grid_var = np.zeros(len(grid_time))

        # IMPORTANT: Subtract mean from the flux time series
        # IMPORTANT2: Missing points must have zeros!
        weights = flux_errs**(-2) / np.nansum(flux_errs**(-2))
        fluxes = (flxs-np.mean(flxs))*weights
        fluxes -= np.mean(fluxes)

        # Assign the fluxes & variances according to the time grid:
        for i in range(len(fluxes)):
            ind = int(np.round((time_axis[i] - grid_time[0]) / dt))
            grid_flux[ind] = fluxes[i]
            grid_var[ind] = weights[i]

        # Add the input to the class structure
        self.time_axis = time_axis
        self.fluxes = fluxes
        self.flux_errs = flux_errs

        # Add the grid to the class structure
        self.grid_time = grid_time
        self.grid_flux = grid_flux
        self.grid_var  = grid_var
        self.dt = dt

# =========================================================================================================
    def BLS(self, PeriodRange, NumberOfPeriodChunks=1,
            DutyCycle=0.03, over_sampling=4, ToleranceDenom=4,
            minWidth=0, maxWidth=None, TransitDuration=0.0,
            PeriodChunks='freq', arrayInitSize=6e5):
        '''
        Input: PeriodRange - Numpy array, [Pmin, Pmax].
               NumberOfPeriodChunks - int. Divide the analysis into several
                                      period chunks to avoid memory issues.
                                      The chunks will be evenly spaced in freq.
                plotFlag - if True, plots the periodogram
                TransitDuration - Duration of the transit, in units of the time_axis
                                  if provided, the DutyCycle will be taken as the maximum
                                  value of DutyCycle, TransitDuration/MaxPeriodInChunk

               Parameters of the dynamical programming code:
               (see GenerateTrialPeriod below for more info)
               DutyCycle - float. the estimated fraction of the transit in phase
               over_sampling - integer. The number of points inside the transit
               ToleranceDenom - float. divides the tolerance
        '''
        # Generate chuncks of period to analyze:
        # Since the analysis is demanding in terms of memory, in order to scan a large range
        # of orbital periods we need to divide into bins.
        if PeriodChunks == 'power':
            # Option 1: Steps that are evensly spaces in P^(1/3)
            print(f"Cutting {NumberOfPeriodChunks} chunks, evenly spaced in P**(1/3).")
            start = np.power(PeriodRange[0], 1/3)
            stop = np.power(PeriodRange[1], 1/3)
            pVec = np.power(np.linspace(start, stop,
                                        num=NumberOfPeriodChunks+1,
                                        endpoint=True), 3)

        elif PeriodChunks == 'freq':
            # Option 2: chose to divide to bins that are evenly spaced in the frequency
            print(f"Cutting {NumberOfPeriodChunks} chunks, evenly spaced in frequency.")
            freqVec = np.linspace(1/PeriodRange[1], 1/PeriodRange[0],
                                  num=NumberOfPeriodChunks+1,
                                  endpoint=True)
            pVec = 1/freqVec
            pVec.sort()

        # initialize:
        start_time = time.time()

        p = np.full(int(arrayInitSize), np.nan)
        score = np.full(int(arrayInitSize), np.nan)
        width = np.full(int(arrayInitSize), np.nan)
        Nbin = np.full(int(arrayInitSize), np.nan)
        counterInd = 0

        with tqdm(total=NumberOfPeriodChunks) as pbar:
            for ChunkN in range(NumberOfPeriodChunks):
                # TD = TransitDuration * (pVec[ChunkN]/pVec[0])**(1/3)
                pt, st, wt, Nt = __BLS_PeriodChunk__(self.grid_flux, self.grid_var, self.dt,
                                                          [pVec[ChunkN], pVec[ChunkN+1]],
                                                          DutyCycle, over_sampling, ToleranceDenom,
                                                          minWidth, maxWidth, TransitDuration)

                p[counterInd:counterInd+len(pt)] = pt
                score[counterInd:counterInd+len(pt)] = st
                width[counterInd:counterInd+len(pt)] = wt
                Nbin[counterInd:counterInd+len(pt)] = Nt
                counterInd += len(pt)
                pbar.update(1)

        ind = ~np.isnan(p)

        # shahaf: 20211111:
        # return p[ind], score[ind], width[ind], Nbin[ind], dds
        return p[ind], score[ind], width[ind], Nbin[ind]

# =========================================================================================================
    def GenerateTrialPeriods(self, PeriodRange, DutyCycle=0.03,
                             over_sampling=4, ToleranceDenom=2, plotFlag=False):
        '''
        Input: PeriodRange - Numpy array, [Pmin, Pmax].
               DutyCycle - float. the estimated fraction of the transit in phase
               over_sampling - integer. The number of points inside the transit
               ToleranceDenom - float. divides the tolerance (see below)

        Parameters of the periodicity search code
        ------------------------------------------
         * duty_cycle: is the estimated duration of the transit in the phase
         * over_sampling: is the number of points inside the transit.
         * ToleranceDenom: The tolerance determines how far can an each point
                           be (in units of bins) from its correct location.
                          Here, this is half the duty cycle in bins.
                          This one affects the resultion in period for the code
                          output. Here, this is half the duty cycle in bins,
                          divided by ToleranceDenom.
        '''

        self.TrialPeriods, self.dyp_data_structure, dyp = __GenerateTrialPeriods__(self.grid_flux, self.grid_var, self.dt,
                                                                                   PeriodRange, DutyCycle, over_sampling,
                                                                                   ToleranceDenom)

        # Plotting the 2D map:
        if plotFlag:
            plt.figure(figsize=(9, 14))
            plt.grid()
            plt.imshow(dyp.data_structure[0, :, 0, :]/dyp.data_structure[0, :, 1, :]**0.5,
                       interpolation='nearest', cmap=cm.Greys_r)
            plt.xlabel('Bin #', fontsize=16)
            plt.ylabel('Trial Period', fontsize=16)
            plt.title('FastFoldIntensityMap', fontsize=16)
            plt.show()

        return self

# =========================================================================================================


#           ******************************************************************************
#           ******************************************************************************
#           ******************************************************************************
#                       2) Auxiliary routines, and jitted subfunctions
#           ******************************************************************************
#           ******************************************************************************
#           ******************************************************************************
def __GenerateTrialPeriods__(grid_flux, grid_var, dt, PeriodRange, DutyCycle,
                             over_sampling, ToleranceDenom):
    '''
    Input: PeriodRange - Numpy array, [Pmin, Pmax].
           DutyCycle - float. the estimated fraction of the transit in phase
           over_sampling - integer. The number of points inside the transit
           ToleranceDenom - float. divides the tolerance (see below)

    Parameters of the periodicity search code
    ------------------------------------------
     * duty_cycle: is the estimated duration of the transit in the phase
     * over_sampling: is the number of points inside the transit.
     * ToleranceDenom: The tolerance determines how far can an each point
                       be (in units of bins) from its correct location.
                      Here, this is half the duty cycle in bins.
                      This one affects the resultion in period for the code
                      output. Here, this is half the duty cycle in bins,
                      divided by ToleranceDenom.
    '''
    # The number of bins in each folded profile
    n_fold_bins = int(over_sampling * 1./DutyCycle)

    # IMPORTANT PARAMETER: When folding, how far an each pointbe (in units of bins)
    # from its correct location. Here, this is half the duty cycle in bins.
    # This one affects the resultion in period for the code output.
    P0 = 0.5 * (PeriodRange[0] + PeriodRange[1])
    tolerance = P0*DutyCycle / dt / ToleranceDenom

    # If the egress is important, one should cosider having a smaller tolderance.

    # Summary: the resolution in the bin axis is determined bu the n_fold_bins parameter
    #          the period resolution is determined by the tolerance.
    #          the two are linked via the duty cycle.

    # Run periodicity search
    # ----------------------
    # The periodicity search is initialilzed by the periodicity_search_dynamic_programming routine.
    # retyrn as class the contains the required information and routines (see inside) for the analysis.
    dyp = FFA_init(grid_flux, grid_var, dt, PeriodRange[0], PeriodRange[1],
                   n_fold_bins, tolerance, np.float32,
                   default_addition_function, default_shift_function)

    # init_data_structure - does a few folding trials, to stars with.
    # used to aviois accuracy losses in the initial point of the algorithm
    dyp.init_data_structure()

    # Run the iterative analysis, number of iterations can be limited to some number.
    # Output after do_iterations: dyp.data_structure shape:
    # Option 1:
    # [number of time chuncks , number of trial periods, 0-folded flux 1-BinNpts, n_fold_bins]
    # Option 2, with variance (required replacement of the brute force folding):
    # [number of time chuncks , numer of trial periods, 0-folded flux 1-folded variance 2-folded suqred flux 3-BinNpts, n_fold_bins]

    dyp.do_iterations()
    # When do iterations is over:
    # the number of time chucks is 1, the number of trial periods is large.

    # In case the iterations are to be stopped at some time:
    # Example, to n_iters = maximum - 3, we can use:
    # max number of iterations: Niter = int(np.log2(dyp.data_structure.shape[0]))
    # And the input to the code: dyp.do_iterations(n_iters=Niter-3)

    data_structure = dyp.data_structure
    TrialPeriods = dyp.param_list[0]

    return TrialPeriods, data_structure, dyp


# ******************************************************************************
# ******************************************************************************
def __BLS_PeriodChunk__(grid_flux, grid_var, dt, PeriodRange, DutyCycle,
                        over_sampling, ToleranceDenom, minWidth,
                        maxWidth, TransitDuration):

    # Disable output
    # sys.stdout = open(os.devnull, 'w')
    DebugFlag = False

    DutyCycle = np.maximum(DutyCycle,
                           TransitDuration/((PeriodRange[0]+PeriodRange[1])/2))

    # For preformance checks, we can track the time the fold and BLS calculation:
    if DebugFlag:
        start_time = time.time()

    p, dds, _ = __GenerateTrialPeriods__(grid_flux, grid_var, dt,
                                         PeriodRange, DutyCycle, over_sampling,
                                         ToleranceDenom)

    if DebugFlag:
        print('Folding time = {0:.4f} sec.'.format(time.time()-start_time), flush=True)
        start_time = time.time()

    # Set the maximum width of the box to be half of the data
    if maxWidth is None:
        maxWidth = dds.shape[3]//3

    # Calculate the BLS score
    # Read the (total) flux per bin
    FoldFlux = np.squeeze(np.asarray(dds[0, :, 0, :], dtype='f4'))

    # Read the number of points per bin
    FoldNpts = np.squeeze(np.asarray(dds[0, :, 1, :], dtype='f4'))

    # Remove dds to save memory (commented out Sahar shahaf 20211111).
    dds = None

    s, w, N = __BLS_score__(FoldNpts, FoldFlux,
                            minWidth, maxWidth)

    if DebugFlag:
        print('BLS time = {0:.4f} sec.'.format(time.time()-start_time), flush=True)

    # Enable output:
    # sys.stdout = sys.__stdout__

    # Shahaf 20211111:
    # return p, s, w, N, dds
    return p, s, w, N


# ******************************************************************************
# ******************************************************************************
def __BLS_score__(FoldNpts, FoldFlux, minWidth, maxWidth):
    '''
    Input:  FoldNpts, FoldFlux - numpy 2D arrays (N folds X N phase bins)
            containing the number of points in rach bin, the total flux in each bin,
            It is assumed that the mean flus was subtracted from each fold
            minWidth, maxWidth - minimal maximal transit width in bins.

    Output: Score, Width, NbinInFold. Four numpy arrays with
            the orbital period, its BLS score and corresponding width
            and the number of bins in the trial period fold.
    '''

    # Total number of points in all bins
    twiceS = np.concatenate((FoldFlux, FoldFlux[:, 0:maxWidth+1]), axis=1).astype('f4')

    # Calculate the r value (see Section (2) in the BLS paper):
    twiceR = np.concatenate((FoldNpts, FoldNpts[:, 0:maxWidth+1]), axis=1).astype('f4')

    # Initialize data structures, according to BLS 2002 convention
    s = np.zeros_like(FoldFlux, dtype='f4')
    r = np.zeros_like(FoldNpts, dtype='f4')

    Score = np.zeros(FoldFlux.shape[0], dtype='f4')
    Width = np.zeros(FoldFlux.shape[0], dtype='f4')
    NbinInFold = FoldFlux.shape[1]

    # Convolve with a box. Define a rectangle pulse with a given window_length
    epsilon = 1e-16
    for i in np.arange(maxWidth):

        s += twiceS[:, i:i+NbinInFold]
        r += twiceR[:, i:i+NbinInFold]

        if minWidth <= i+1:
            # SR is the score defined in the BLS paper: SR = (s**2/(r*(1-r)))**0.5
            SR = np.divide(np.abs(s), np.sqrt(r*(1-r) + epsilon), dtype='f4')
            SRmax = np.max(SR, axis=1)
            new_score_ind = SRmax > Score
            Score[new_score_ind] = SRmax[new_score_ind]
            Width[new_score_ind] = np.divide(i+1, NbinInFold, dtype='f4')

    NbinInFold = np.ones(Score.shape)*NbinInFold

    return Score, Width, NbinInFold
