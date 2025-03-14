import numpy as np
import scipy.stats
import scipy.signal
from scipy.signal import find_peaks
import scipy as sp


def shannon_entropy(signal):
    """Calculate the Shannon entropy of a signal."""
    probability_distribution, _ = np.histogram(signal, bins='fd', density=True)
    probability_distribution = probability_distribution[probability_distribution > 0]
    entropy = -np.sum(probability_distribution * np.log2(probability_distribution))
    return entropy


def higuchi_fd(signal, kmax=10):
    """Calculate the Higuchi Fractal Dimension of a signal."""
    N = len(signal)
    Lk = np.zeros((kmax,))

    for k in range(1, kmax + 1):
        Lmk = np.zeros((k,))
        for m in range(k):
            Lm = 0
            n_max = int(np.floor((N - m - 1) / k))
            for j in range(1, n_max):
                Lm += np.abs(signal[m + j * k] - signal[m + (j - 1) * k])
            Lmk[m] = (Lm * (N - 1) / (k * n_max)) / k

        Lk[k - 1] = np.mean(Lmk)

    Lk = np.log(Lk)
    ln_k = np.log(np.arange(1, kmax + 1))
    higuchi, _ = np.polyfit(ln_k, Lk, 1)

    return higuchi

def hjorth_parameters(y):
    """Calculate the Hjorth parameters of a signal."""
    first_deriv = np.diff(y)
    second_deriv = np.diff(y, 2)

    activity = np.var(y)
    mobility = np.sqrt(np.var(first_deriv) / activity)
    complexity = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility

    return activity, mobility, complexity


def zero_crossing_rate(signal):
    """Calculate the zero-crossing rate of a signal."""
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    return len(zero_crossings) / len(signal)


def root_mean_square(signal):
    """Calculate the root mean square (RMS) of a signal."""
    return np.sqrt(np.mean(signal ** 2))


def energy(signal):
    """Calculate the energy of a signal."""
    return np.sum(signal ** 2)


def envelope(signal):
    """Calculate the envelope of a signal using Hilbert transform."""
    analytic_signal = scipy.signal.hilbert(signal)
    return np.abs(analytic_signal)


def autocorrelation(signal):
    """Calculate the autocorrelation of a signal."""
    result = np.correlate(signal, signal, mode='full')
    return result[result.size // 2:]


def peak_analysis(signal):
    """Calculate the peak analysis features of a signal."""
    peaks, _ = find_peaks(signal)
    peak_count = len(peaks)
    peak_to_peak_val = np.ptp(signal)
    return peak_count, peak_to_peak_val


def spectral_entropy(signal, sf):
    """Calculate the spectral entropy of a signal."""
    _, psd = scipy.signal.welch(signal, sf)
    psd_norm = psd / psd.sum()
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return spectral_entropy


def hurst_exponent(signal):
    """Calculate the Hurst exponent of a signal."""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(signal[lag:], signal[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def approximate_entropy(signal):
    """Calculate the approximate entropy of a signal."""
    N = len(signal)
    m = 2
    r = 0.2 * np.std(signal)

    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.abs(x[:, None] - x[None, :]).max(axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)

    return _phi(m) - _phi(m + 1)


def FeatureExtract(y, sf=512):
    # f, t, Zxx = sp.signal.stft(y, 512, nperseg=15 * 512, noverlap=14 * 512)
    # delta = np.array([], dtype=float)
    # theta = np.array([], dtype=float)
    # alpha = np.array([], dtype=float)
    # beta = np.array([], dtype=float)
    # for i in range(0, int(t[-1])):
    #     indices = np.where((f >= 0.5) & (f <= 4))[0]
    #     delta = np.append(delta, np.sum(np.abs(Zxx[indices, i])))

    #     indices = np.where((f >= 4) & (f <= 8))[0]
    #     theta = np.append(theta, np.sum(np.abs(Zxx[indices, i])))

    #     indices = np.where((f >= 8) & (f <= 13))[0]
    #     alpha = np.append(alpha, np.sum(np.abs(Zxx[indices, i])))

    #     indices = np.where((f >= 13) & (f <= 30))[0]
    #     beta = np.append(beta, np.sum(np.abs(Zxx[indices, i])))

    # abr = alpha / beta
    # tbr = theta / beta
    # dbr = delta / beta
    # tar = theta / alpha
    # dar = delta / alpha
    # dtabr = (alpha + beta) / (delta + theta)

    L = len(y)  # Length of signal

    Y = np.fft.fft(y)  # Perform FFT
    Y[0] = 0  # Set DC component to zero
    P2 = np.abs(Y / L)  # Two-sided spectrum
    P1 = P2[:L // 2 + 1]  # One-sided spectrum
    P1[1:-1] = 2 * P1[1:-1]  # Adjust FFT spectrum

    # Frequency ranges
    f1 = np.arange(len(P1)) * sf / len(P1)
    indices1 = np.where((f1 >= 0.5) & (f1 <= 4))[0]
    delta = np.sum(P1[indices1])
    indices1 = np.where((f1 >= 4) & (f1 <= 8))[0]
    theta = np.sum(P1[indices1])
    indices1 = np.where((f1 >= 8) & (f1 <= 13))[0]
    alpha = np.sum(P1[indices1])
    indices1 = np.where((f1 >= 13) & (f1 <= 40))[0]
    beta = np.sum(P1[indices1])

    # Feature ratios
    abr = alpha / beta
    tbr = theta / beta
    dbr = delta / beta
    tar = theta / alpha
    dar = delta / alpha
    dtabr = (delta + theta) / (alpha + beta)

    # Basic statistical features
    mean_val = np.mean(y)
    variance_val = np.var(y)
    min_val = np.min(y)
    max_val = np.max(y)
    skewness_val = scipy.stats.skew(y)
    kurtosis_val = scipy.stats.kurtosis(y)

    # Hjorth parameters
    activity, mobility, complexity = hjorth_parameters(y)

    # Non-linear features
    entropy_val = shannon_entropy(y)
    # fractal_dim_val = higuchi_fd(y)

    # Time-domain features
    # zero_cross_rate = zero_crossing_rate(y)
    # rms_val = root_mean_square(y)
    # energy_val = energy(y)
    # envelope_val = np.mean(envelope(y))
    # autocorr_val = np.mean(autocorrelation(y))
    #
    # # Peak analysis
    # peak_count, peak_to_peak_val = peak_analysis(y)
    #
    # # Spectral features
    # spectral_entropy_val = spectral_entropy(y, sf)
    #
    # # Additional non-linear features
    # hurst_val = hurst_exponent(y)
    # approx_entropy_val = approximate_entropy(y)

    # Create feature dictionary
    features_dict = {
        "delta": delta,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "abr": abr,
        "tbr": tbr,
        "dbr": dbr,
        "tar": tar,
        "dar": dar,
        "dtabr": dtabr,
        "mean": mean_val,
        "variance": variance_val,
        "min": min_val,
        "max": max_val,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "entropy": entropy_val,
        # "fractal_dimension": fractal_dim_val,
        # "zero_crossing_rate": zero_cross_rate,
        # "rms": rms_val,
        # "energy": energy_val,
        # "envelope": envelope_val,
        # "autocorrelation": autocorr_val,
        # "peak_count": peak_count,
        # "peak_to_peak": peak_to_peak_val,
        # "spectral_entropy": spectral_entropy_val,
        # "hurst_exponent": hurst_val,
        # "approximate_entropy": approx_entropy_val,
    }

    return features_dict

def STFT_feature(data):
    f, t, Zxx = sp.signal.stft(data, 512, nperseg=15 * 512, noverlap=14 * 512)
    delta = np.array([], dtype=float)
    theta = np.array([], dtype=float)
    alpha = np.array([], dtype=float)
    beta = np.array([], dtype=float)
    for i in range(0, int(t[-1])):
        indices = np.where((f >= 0.5) & (f <= 4))[0]
        delta = np.append(delta, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 4) & (f <= 8))[0]
        theta = np.append(theta, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 8) & (f <= 13))[0]
        alpha = np.append(alpha, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 13) & (f <= 30))[0]
        beta = np.append(beta, np.sum(np.abs(Zxx[indices, i])))

    abr = alpha / beta
    tbr = theta / beta
    dbr = delta / beta
    tar = theta / alpha
    dar = delta / alpha
    dtabr = (alpha + beta) / (delta + theta)

    diction = {"delta": delta,
               "theta": theta,
               "alpha": alpha,
               "beta": beta,
               "abr": abr,
               "tbr": tbr,
               "dbr": dbr,
               "tar": tar,
               "dar": dar,
               "dtabr": dtabr
               }
    return (t,f,Zxx)



