# fBLS

An implementation of fBLS, a fast-folding-based BLS periodogram, described in [ArXiv: 2204.02398](https://arxiv.org/abs/2204.02398).

The core of **fBLS** consists of a dynamic programming algorithm, akin to the Fast Folding Analysis, that was proposed for [pulsar search](https://ui.adsabs.harvard.edu/abs/1977IrAJ...13..142D/abstract). This algorithm simultaneously generates folded profiles for a dense grid of trial periods *in linear time*. The grid of trial periods is analyzed in a search for box-shaped signals ([BLS](https://ui.adsabs.harvard.edu/abs/2002A%26A...391..369K/abstract)), thus producing a BLS-periodogram.

Use Conda to set up the required environment from the yml file. The fBLS_illustration notebook aims to describe the algorithm. fBLS_example executes the code for a real example from Kepler.
