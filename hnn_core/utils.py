"""Utility functions."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <cjb@cfin.au.dk>

import logging
import numpy as np

from .externals.mne import _validate_type


logger = logging.getLogger('hnn-core')
logger.addHandler(logging.StreamHandler())

try:
    from tqdm.auto import tqdm
    has_tqdm = True
except ImportError:
    logger.warning(
        "tqdm library not found. Falling back to non-interactive progress "
        "visualization.")
    has_tqdm = False

def set_log_level(verbose):
    """Convenience function for setting the log level.

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
    """
    if isinstance(verbose, bool):
        if verbose is True:
            verbose = 'INFO'
        else:
            verbose = 'WARNING'
    if isinstance(verbose, str):
        verbose = verbose.upper()
        logging_types = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                             WARNING=logging.WARNING, ERROR=logging.ERROR,
                             CRITICAL=logging.CRITICAL)
        if verbose not in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]
    logger.setLevel(verbose)


def _verbose_iterable(data):
    """Wrap an iterable object with tqdm.

    If tqdm is not available or if we did not set the appropriate
    log level, then we fall back to the classical method.

    Parameters
    ----------
    data: range, list
        This will be data which will wrapped by tqdm (if available).
    Returns
    -------
    wrapped_data: tqdm object
        Data object wrapped with tqdm.
    """
    wrapped_data = data
    if logger.getEffectiveLevel() == logging.INFO:
        if has_tqdm:
            wrapped_data = tqdm(data, leave=False)
    return wrapped_data


def _tqdm_log(msg):
    """Log a message by using either tqdm or the system logger.

    Parameters
    ----------
    msg: string
        Message which will be printed
    """
    if has_tqdm:
        tqdm.write(msg)
    else:
        logger.info(msg)


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    assert len(x) > winsz
    win = np.hamming(winsz)
    win /= sum(win)
    return np.convolve(x, win, 'same')


# Savitzky-Golay filtering, lifted and adapted from mne-python (0.22)
def _savgol_filter(data, h_freq, sfreq):
    """Filter the data using Savitzky-Golay polynomial method.

    Parameters
    ----------
    data : array-like
        The data to filter (1D)
    h_freq : float
        Approximate high cutoff frequency in Hz. Note that this
        is not an exact cutoff, since Savitzky-Golay filtering
        is done using polynomial fits
        instead of FIR/IIR filtering. This parameter is thus used to
        determine the length of the window over which a 5th-order
        polynomial smoothing is applied.
    sfreq : float
        The sampling frequency (in Hz)

    Returns
    -------
    filt_data : array-like
        The filtered data
    """  # noqa: E501
    from scipy.signal import savgol_filter

    _validate_type(sfreq, (float, int), 'sfreq')
    assert sfreq > 0.
    _validate_type(h_freq, (float, int), 'h_freq')
    assert h_freq > 0.

    h_freq = float(h_freq)
    if h_freq >= sfreq / 2.:
        raise ValueError('h_freq must be less than half the sample rate')

    # savitzky-golay filtering
    window_length = (int(np.round(sfreq / h_freq)) // 2) * 2 + 1
    # loop over 'agg', 'L2', and 'L5'
    filt_data = savgol_filter(data, axis=-1, polyorder=5,
                              window_length=window_length)
    return filt_data


def smooth_waveform(data, window_len, sfreq):
    """Smooth an arbitrary waveform using Hamming-windowed convolution

    Parameters
    ----------
    data : list | np.ndarray
        The data to filter
    window_len : float
        The length (in ms) of a `~numpy.hamming` window to convolve the
        data with.
    sfreq : float
        The data sampling rate.

    Returns
    -------
    data_filt : np.ndarray
        The filtered data
    """
    if ((isinstance(data, np.ndarray) and data.ndim > 1) or
            (isinstance(data, list) and isinstance(data[0], list))):
        raise RuntimeError('smoothing currently only supported for 1D-arrays')

    if not isinstance(window_len, (float, int)) or window_len < 0:
        raise ValueError('Window length must be a non-negative number')
    elif 0 < window_len < 1:
        raise ValueError('Window length less than 1 ms is not supported')

    _validate_type(sfreq, (float, int), 'sfreq')
    assert sfreq > 0.
    # convolutional filter length is given in samples
    winsz = np.round(1e-3 * window_len * sfreq)
    if winsz > len(data):
        raise ValueError(
            f'Window length too long: {winsz} samples; data length is '
            f'{len(data)} samples')

    return _hammfilt(data, winsz)
