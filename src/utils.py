from obci_readmanager.signal_processing.read_manager import ReadManager
import pandas as pd
import numpy as np
from scipy import signal
from typing import Optional, Dict, Any, List, Tuple, Literal

EEG_SCALING_FACTOR = 0.0715
BASELINE = 2


def download_signal(
    bin_file_path: str,
    xml_file_path: str,
    tag_file_path: Optional[str] = None,
    csv_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """Loads EEG data and metadata from specified files."""

    eeg_data: Dict[str, Any] = {}

    mgr = ReadManager(xml_file_path, bin_file_path, tag_file_path)

    # Retrieve metadata
    eeg_data['sampling'] = float(mgr.get_param("sampling_frequency"))
    eeg_data['channels_names'] = mgr.get_param("channels_names")

    # Retrieve and scale EEG data
    raw_samples = mgr.get_samples()
    eeg_data['data'] = raw_samples * EEG_SCALING_FACTOR

    # Retrieve tags if tag file was provided
    if tag_file_path:
        eeg_data['tags'] = mgr.get_tags()

    # Handle optional CSV data - initialize key first
    eeg_data['data_csv'] = None
    if csv_file_path:
        try:
            eeg_data['data_csv'] = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"Warning: Optional CSV file not found: {csv_file_path}")
        except pd.errors.EmptyDataError:
             print(f"Warning: Optional CSV file is empty: {csv_file_path}")
        except Exception as e:
             print(f"Warning: Error reading optional CSV file {csv_file_path}: {e}")

    return eeg_data


def _apply_iir_filter(
    syg: np.ndarray,
    fs: float,
    params: Dict[str, Any],
    btype: str,
    axis: int = -1
) -> np.ndarray:
    """Helper function to design and apply a single IIR filter (SOS)."""
    # Basic check for required parameters
    required_keys = ['N', 'Wn', 'ftype']
    if not all(key in params for key in required_keys):
        raise ValueError(f"Filter params for {btype} must include {required_keys}")

    # Specific validation for bandpass Wn
    if btype in ['bandpass', 'bandstop']:
        wn_val = params['Wn']
        if not isinstance(wn_val, (list, tuple)) or len(wn_val) != 2:
            raise ValueError(f"{btype} 'Wn' parameter must be a list/tuple of two frequencies [low, high]")


    # Extract optional parameters relevant to iirfilter (e.g., rp, rs for certain ftypes)
    optional_filter_params = {k: v for k, v in params.items() if k in ['rp', 'rs']}

    sos = signal.iirfilter(
        N=params['N'],
        Wn=params['Wn'],
        btype=btype,
        analog=False,
        ftype=params['ftype'],
        fs=fs,
        output='sos',
        **optional_filter_params
    )
    # Apply the filter using Second-Order Sections for stability
    return signal.sosfiltfilt(sos, syg, axis=axis)

def _apply_notch_filter(
    syg: np.ndarray,
    fs: float,
    params: Dict[str, Any],
    axis: int = -1
) -> np.ndarray:
    """Helper function to design and apply a notch filter (SOS)."""
    required_keys = ['w0', 'Q']
    if not all(key in params for key in required_keys):
        raise ValueError(f"Notch filter params must include {required_keys}")

    w0 = params['w0'] # Frequency to remove
    Q = params['Q']   # Quality factor

    # Design notch filter using b, a coefficients first
    b_notch, a_notch = signal.iirnotch(w0=w0, Q=Q, fs=fs)
    # Convert to SOS format for stability with sosfiltfilt
    sos_notch = signal.tf2sos(b_notch, a_notch)
    # Apply the filter
    return signal.sosfiltfilt(sos_notch, syg, axis=axis)


def filter_signal(
    syg: np.ndarray,
    fs: float,
    lowpass_params: Optional[Dict[str, Any]] = None,
    highpass_params: Optional[Dict[str, Any]] = None,
    bandpass_params: Optional[Dict[str, Any]] = None,
    notch_params: Optional[Dict[str, Any]] = None,
    axis: int = -1
) -> np.ndarray:
    """
    Applies a sequence of digital filters (lowpass, highpass, bandpass, notch)
    to a signal using zero-phase filtering (sosfiltfilt).

    Args:
        syg: Input signal (numpy array).
        fs: Sampling frequency.
        lowpass_params: Dictionary with parameters for signal.iirfilter (btype='lowpass').
                        Requires 'N', 'Wn', 'ftype'. Optional: 'rp', 'rs'.
        highpass_params: Dictionary with parameters for signal.iirfilter (btype='highpass').
                         Requires 'N', 'Wn', 'ftype'. Optional: 'rp', 'rs'.
        bandpass_params: Dictionary with parameters for signal.iirfilter (btype='bandpass').
                         Requires 'N', 'Wn' (as [low, high]), 'ftype'. Optional: 'rp', 'rs'.
        notch_params: Dictionary with parameters for signal.iirnotch.
                      Requires 'w0' (notch frequency), 'Q' (quality factor).
        axis: Axis along which to apply the filter.

    Returns:
        Filtered signal (numpy array).
    """
    syg_filtered = syg.copy() # Work on a copy

    # Apply filters in a specific order if needed, or as provided.
    if bandpass_params is not None:
        syg_filtered = _apply_iir_filter(syg_filtered, fs, bandpass_params, 'bandpass', axis)

    if notch_params is not None:
        syg_filtered = _apply_notch_filter(syg_filtered, fs, notch_params, axis)

    if lowpass_params is not None:
        syg_filtered = _apply_iir_filter(syg_filtered, fs, lowpass_params, 'lowpass', axis)

    if highpass_params is not None:
        syg_filtered = _apply_iir_filter(syg_filtered, fs, highpass_params, 'highpass', axis)


    return syg_filtered


def apply_montage(
    syg: np.ndarray,
    channels_names: List[str],
    montage_type: Literal['common_average', 'linked_ears', 'channel'],
    reference_channel: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Applies a specified referencing montage to EEG signal data.

    Args:
        syg: Input EEG signal data (numpy array, channels x time).
        channels_names: List of channel names corresponding to the rows of syg.
        montage_type: The type of montage to apply. Options:
                      'common_average': Subtract the mean of all channels.
                      'linked_ears': Subtract the mean of 'M1' and 'M2' channels.
                                     'M1' and 'M2' are removed from the output.
                      'channel': Subtract a specific reference channel.
                                 The reference channel is removed from the output.
        reference_channel: The name of the channel to use as reference
                           (required only if montage_type='channel').

    Returns:
        A tuple containing:
        - syg_montaged: The re-referenced signal data (numpy array).
        - channels_names_montaged: The updated list of channel names.

    Raises:
        ValueError: If inputs are invalid (e.g., missing channels, invalid type).
    """
    # Validate input dimensions
    if syg.ndim != 2:
        raise ValueError("Input signal 'syg' must be a 2D array (channels x time).")
    if syg.shape[0] != len(channels_names):
        raise ValueError("Number of rows in 'syg' must match the number of channel names.")

    syg_montaged: np.ndarray
    channels_names_montaged: List[str]

    if montage_type == 'common_average':
        if syg.shape[0] < 2:
             raise ValueError("Common average montage requires at least 2 channels.")
        # Compute the mean across channels
        average_reference = np.mean(syg, axis=0, keepdims=True)
        # Subtract the average reference from all channels
        syg_montaged = syg - average_reference
        channels_names_montaged = channels_names # Channel names remain the same

    elif montage_type == 'linked_ears':
        mastoid_labels = {'M1', 'M2'}
        mastoid_indices = [i for i, name in enumerate(channels_names) if name in mastoid_labels]

        if len(mastoid_indices) != 2:
            raise ValueError("Both 'M1' and 'M2' must be present in channels_names for 'linked_ears' montage.")

        # Calculate the average of M1 and M2 signals
        linked_ear_reference = np.mean(syg[mastoid_indices, :], axis=0, keepdims=True)

        # Identify indices of channels *excluding* M1 and M2
        non_mastoid_indices = [i for i, name in enumerate(channels_names) if name not in mastoid_labels]

        if not non_mastoid_indices:
             raise ValueError("No channels remaining after excluding M1 and M2.")

        # Select non-mastoid channels and subtract the reference
        syg_montaged = syg[non_mastoid_indices, :] - linked_ear_reference
        # Update channel names list
        channels_names_montaged = [channels_names[i] for i in non_mastoid_indices]

    elif montage_type == 'channel':
        if reference_channel is None:
            raise ValueError("`reference_channel` must be specified for montage_type='channel'.")
        if reference_channel not in channels_names:
            raise ValueError(f"Reference channel '{reference_channel}' not found in channels_names.")

        try:
            ref_index = channels_names.index(reference_channel)
        except ValueError: # Should be caught by the 'in' check above, but belt-and-suspenders
             raise ValueError(f"Reference channel '{reference_channel}' not found.")

        # Get the reference signal
        reference_signal = syg[ref_index, :].reshape(1, -1) # Keep dims for broadcasting

        # Identify indices of channels *excluding* the reference channel
        non_ref_indices = [i for i, name in enumerate(channels_names) if i != ref_index]

        if not non_ref_indices:
             raise ValueError(f"No channels remaining after excluding reference '{reference_channel}'.")

        # Select non-reference channels and subtract the reference signal
        syg_montaged = syg[non_ref_indices, :] - reference_signal
        # Update channel names list
        channels_names_montaged = [channels_names[i] for i in non_ref_indices]

    else:
        # Use Literal type hint to help catch this earlier, but keep runtime check
        valid_types = ['common_average', 'linked_ears', 'channel']
        raise ValueError(f"Invalid montage_type '{montage_type}'. Choose from {valid_types}.")

    return syg_montaged, channels_names_montaged


def cut_signal(syg, tags, sampling):
    # Initialize parameters
    dlugosc = int((6 + BASELINE) * sampling)
    lewa_list = []
    prawa_list = []
    lewa, prawa = ('lewa', 'prawa')

    for tag in tags:
        # Determine if the tag is frequent or rare based on the timestamp
        t0 = int(sampling * (tag['start_timestamp'] - BASELINE))

        # Slice the signal segment
        segment = syg[:, t0:t0 + dlugosc]

        if tag['desc']['strona'] == lewa:
            lewa_list.append(segment)
        elif tag['desc']['strona'] == prawa:
            prawa_list.append(segment)

    lewa_array = np.array(lewa_list)

    prawa_array = np.array(prawa_list)

    return lewa_array, prawa_array