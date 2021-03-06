
import numpy as np





# For BPM and plotting
def pos_based_method_improved(_time_series, _fps):

    # sliding window size
    window_size = 48
    overlap = 1
    signal_length = len(_time_series)
    windows_counter = int(signal_length / overlap)-window_size

    H = np.zeros(signal_length, dtype='float64')

    n = window_size
    m = n - window_size
    for i in range(windows_counter):

        window = _time_series[m:n]

        # 5 temporal normalization
        mean_window = np.average(window, axis=0)
        norm_window = window / mean_window

        # 6 projection
        S1 = np.dot(norm_window, [-1, 1, 0])
        S2 = np.dot(norm_window, [1, 1, -2])

        # 7 tuning
        S1_std = np.std(S1)
        S2_std = np.std(S2)

        alpha = S1_std / S2_std

        h = S1 + alpha * S2

        # Overlap-adding
        H[m:n] += h
        n += 1
        m = n - window_size

    bpm, pruned_fft, heart_rates, fft, raw = get_bpm(H, _fps)

    # bandpass filter for pulse
    bandpassed_fft = fft.copy()
    bound_low = (np.abs(heart_rates - 40)).argmin()
    bound_high = (np.abs(heart_rates - 170)).argmin()

    bandpassed_fft[:bound_low] = 0
    bandpassed_fft[bound_high:] = 0

    return bpm, bandpassed_fft, fft, heart_rates, raw, H, h, norm_window, _time_series[m:n],


# Complete extraction function with POS-Algorithm and FFT
def extract_pos_based_method_improved(_time_series, _fps):

    # sliding window size
    window_size = 48
    hann_window = np.hanning(window_size)
    overlap = 24
    signal_length = len(_time_series)-overlap
    H = np.zeros(signal_length, dtype='float64')

    windows_counter = int((len(_time_series) - window_size) / overlap)
    # windows_counter = int(signal_length / window_size) * 2

    # Windowing
    n = window_size
    m = n - window_size
    for i in range(windows_counter):

        window = _time_series[m:n]

        # Call signal transforamtion function
        hann_windowed_signal = rgb_into_pulse_signal(window, hann_window)

        # Overlap-adding
        H[m:n] += hann_windowed_signal
        n += overlap
        m = n - window_size

    # last window is splitted by half and added at the end and front of H
    last_hann_windowed_signal = rgb_into_pulse_signal(_time_series[m:n], hann_window)

    # 1st half added at the back
    H[-overlap:] += last_hann_windowed_signal[:overlap]
    # 2nd half added at the front
    H[0:overlap] += last_hann_windowed_signal[overlap:]

    bpm, pruned_fft, heart_rates, fft, raw = get_bpm(H, _fps)

    return bpm, pruned_fft


# application of FFT to get BPM and human frequencies from 40 to 170 BPM
def get_bpm(_H, _fps):

    # Fourier Transform
    raw = np.fft.fft(_H, 512)
    L = int(len(raw) / 2 + 1)
    fft = np.abs(raw[:L])

    frequencies = np.linspace(0, _fps/2.0, L, endpoint=True)

    heart_rates = frequencies * 60.0

    # bandpass filter for pulse
    bandpassed_fft = fft.copy()
    bound_low = (np.abs(heart_rates - 40)).argmin()
    bound_high = (np.abs(heart_rates - 170)).argmin()

    pruned_fft = bandpassed_fft[bound_low:bound_high]
    bandpassed_fft[:bound_low] = 0
    bandpassed_fft[bound_high:] = 0

    # get position of strongest frequency
    max_freq_pos = np.argmax(bandpassed_fft)

    # bpm is strongest frequnecy
    bpm = heart_rates[max_freq_pos]

    return bpm, pruned_fft, heart_rates, fft, raw


# signal transforamtion function with POS-Algorithm and application of Hann-window-signal
def rgb_into_pulse_signal(_window, _hann_window):

    # Make float64 Array and add 1.0 to avoid means that are zero
    _window = _window.astype('float64')
    _window += 1.0

    # 5 temporal normalization
    mean_window = np.average(_window, axis=0)
    norm_window = _window / mean_window

    # if np.isnan(norm_window).any():
    #     print('nan')

    # 6 apply projection matrice
    S1 = np.dot(norm_window, [-1, 1, 0])
    S2 = np.dot(norm_window, [1, 1, -2])

    # 7 alpha-tuning
    S1_std = np.std(S1)
    S2_std = np.std(S2)

    alpha = S1_std / S2_std

    h = S1 + alpha * S2

    # apply Hann-window-signal
    _hann_windowed_signal = _hann_window * h

    return _hann_windowed_signal


