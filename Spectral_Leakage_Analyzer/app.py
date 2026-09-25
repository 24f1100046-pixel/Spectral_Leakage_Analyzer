import streamlit as st
import numpy as np
import scipy.signal as sig
from scipy.signal.windows import get_window
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Dict

st.set_page_config(page_title="Spectral Leakage Analyzer", layout="wide")
plt.style.use('dark_background')

def generate_signal(fs: float, N: int, sig_type: str, f1: float, a1: float, f2: float, a2: float, noise_snr: float = None) -> np.ndarray:
    t = np.arange(N) / fs
    if sig_type == "Pure Sinusoid":
        signal = a1 * np.sin(2 * np.pi * f1 * t)
    elif sig_type == "Sum of Sinusoids":
        signal = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    elif sig_type == "Chirp Signal":
        signal = sig.chirp(t, f0=f1, t1=t[-1], f1=f2)
    else:
        signal = np.zeros(N)
    
    if noise_snr is not None:
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(noise_snr/10))
        noise = np.sqrt(noise_power) * np.random.randn(N)
        signal += noise
        
    return signal

@st.cache_data
def compute_spectrum(signal: np.ndarray, window_name: str, fs: float, pad_factor: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(signal)
    
    # Generate window
    if window_name == "Rectangular":
        w = np.ones(N)
    elif window_name.startswith("Kaiser"):
        beta = float(window_name.split(" ")[-1])
        w = sig.windows.kaiser(N, beta)
    else:
        # Convert name for scipy
        win_dict = {
            "Hann": "hann",
            "Hamming": "hamming",
            "Blackman": "blackman",
            "Bartlett": "bartlett",
            "Flat Top": "flattop"
        }
        w = sig.windows.get_window(win_dict.get(window_name, "boxcar"), N)
        
    windowed_signal = signal * w
    
    # Zero padding
    N_pad = N * pad_factor
    
    # FFT
    spectrum = np.fft.fft(windowed_signal, N_pad)
    freqs = np.fft.fftfreq(N_pad, 1/fs)
    
    # Take positive frequencies only
    pos_idx = freqs >= 0
    freqs = freqs[pos_idx]
    spectrum = spectrum[pos_idx]
    
    # Normalize to 0 dB max
    mag = np.abs(spectrum)
    # Avoid log of zero
    mag[mag == 0] = 1e-12
    mag_db = 20 * np.log10(mag)
    mag_db -= np.max(mag_db)
    
    return freqs, mag_db, w

def compute_metrics(w: np.ndarray, mag_db: np.ndarray, freqs: np.ndarray, fs: float) -> Dict[str, float]:
    N = len(w)
    # 3dB Bandwidth (Main lobe width)
    peak_idx = np.argmax(mag_db)
    
    left_idx = peak_idx
    while left_idx > 0 and mag_db[left_idx] > -3:
        left_idx -= 1
        
    right_idx = peak_idx
    while right_idx < len(mag_db) - 1 and mag_db[right_idx] > -3:
        right_idx += 1
        
    main_lobe_hz = freqs[right_idx] - freqs[left_idx]
    main_lobe_bins = main_lobe_hz * (N / fs)
    
    # Sidelobe level
    # Find local maxima
    peaks, _ = sig.find_peaks(mag_db)
    if len(peaks) > 1:
        # Exclude main lobe peak and close ones
        valid_peaks = [p for p in peaks if p < left_idx or p > right_idx]
        if valid_peaks:
            sidelobe_level = np.max(mag_db[valid_peaks])
        else:
            sidelobe_level = -np.inf
    else:
        sidelobe_level = -np.inf
        
    # Processing Gain
    pg = 10 * np.log10((np.sum(w)**2) / (N * np.sum(w**2))) if np.sum(w) > 0 else 0
    
    # Scalloping loss (Approximate)
    # Calculate fft of window padded
    W = np.fft.fft(w, 2048)
    W_mag = np.abs(W)
    W_mag /= np.max(W_mag)
    W_db = 20 * np.log10(np.maximum(W_mag, 1e-12))
    bin_spacing = 2048 / N
    half_bin_idx = int(bin_spacing / 2)
    scalloping_loss = W_db[half_bin_idx] if half_bin_idx < len(W_db) else 0

    return {
        "Main-lobe width (Hz)": main_lobe_hz,
        "Main-lobe width (bins)": main_lobe_bins,
        "Highest side-lobe level (dB)": sidelobe_level,
        "Processing gain (dB)": pg,
        "Scalloping loss (dB)": scalloping_loss
    }

def main():
    st.title("Spectral Leakage Analyzer")
    
    with st.expander("What is Spectral Leakage?"):
        st.write("Spectral leakage occurs when a signal's frequency is not perfectly periodic within the observation window. This spreads the signal's energy across multiple frequencies. Windowing functions help mitigate this by tapering the signal at the edges, at the cost of main-lobe widening.")
        
    st.sidebar.header("Signal Controls")
    sig_type = st.sidebar.selectbox("Signal Type", ["Pure Sinusoid", "Sum of Sinusoids", "Chirp Signal"])
    
    fs = st.sidebar.number_input("Sampling Rate (Hz)", 100, 48000, 1000)
    N = st.sidebar.number_input("Number of Samples", 64, 8192, 256)
    
    f1 = st.sidebar.slider("Freq 1 (Hz)", 1.0, float(fs/2), 50.0)
    a1 = st.sidebar.slider("Amp 1", 0.1, 10.0, 1.0)
    
    f2, a2 = 0.0, 0.0
    if sig_type in ["Sum of Sinusoids", "Chirp Signal"]:
        f2 = st.sidebar.slider("Freq 2 (Hz)", 1.0, float(fs/2), 120.0)
        if sig_type == "Sum of Sinusoids":
            a2 = st.sidebar.slider("Amp 2", 0.1, 10.0, 0.5)
            
    add_noise = st.sidebar.checkbox("Add Noise")
    noise_snr = st.sidebar.slider("SNR (dB)", -20, 50, 20) if add_noise else None
    
    signal = generate_signal(fs, N, sig_type, f1, a1, f2, a2, noise_snr)
    
    st.sidebar.header("Window Selection")
    st.sidebar.write("Select at least 2 windows to compare:")
    
    windows_opts = ["Rectangular", "Hann", "Hamming", "Blackman", "Bartlett", "Flat Top"]
    
    selected_windows = []
    for w in windows_opts:
        if st.sidebar.checkbox(w, value=(w in ["Rectangular", "Hann", "Hamming", "Blackman"])):
            selected_windows.append(w)
            
    kaiser = st.sidebar.checkbox("Kaiser")
    if kaiser:
        beta = st.sidebar.slider("Kaiser Beta", 0.0, 20.0, 14.0)
        selected_windows.append(f"Kaiser {beta}")
        
    if len(selected_windows) < 2:
        st.warning("Please select at least 2 windows.")
        return
        
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_windows)))
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Time Domain", "Window Shapes", "Magnitude Spectra", 
        "Metrics", "Leakage Vis"
    ])
    
    # Common variables
    t = np.arange(N) / fs
    
    with tab1:
        st.subheader("Time Domain Signals")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, signal, 'w--', label="Original Signal", alpha=0.5)
        for i, w_name in enumerate(selected_windows):
            _, _, w = compute_spectrum(signal, w_name, fs)
            ax.plot(t, signal * w, label=w_name, color=colors[i], alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Window Shapes and Frequency Responses")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, w_name in enumerate(selected_windows):
                _, _, w = compute_spectrum(signal, w_name, fs)
                ax.plot(w, label=w_name, color=colors[i])
            ax.set_title("Window Shapes (Time Domain)")
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, w_name in enumerate(selected_windows):
                _, _, w = compute_spectrum(signal, w_name, fs)
                W = np.fft.fft(w, 2048)
                W_mag = np.abs(W)
                W_mag /= np.max(W_mag)
                W_db = 20 * np.log10(np.maximum(W_mag, 1e-12))
                freqs_w = np.fft.fftfreq(2048, 1/N)[:1024]
                ax.plot(freqs_w, W_db[:1024], label=w_name, color=colors[i])
            ax.set_title("Window Frequency Responses")
            ax.set_xlabel("Frequency Bins")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_ylim([-120, 5])
            ax.set_xlim([0, 10])
            ax.legend()
            st.pyplot(fig)
            
    with tab3:
        st.subheader("Magnitude Spectra Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, w_name in enumerate(selected_windows):
            freqs, mag_db, _ = compute_spectrum(signal, w_name, fs)
            ax.plot(freqs, mag_db, label=w_name, color=colors[i], alpha=0.8)
            
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized Magnitude (dB)")
        ax.set_ylim([-120, 5])
        ax.legend()
        
        # Zoomed view option
        zoom = st.checkbox("Zoom around Freq 1")
        if zoom:
            span = 10 * (fs/N)
            ax.set_xlim([max(0, f1 - span), min(fs/2, f1 + span)])
            
        st.pyplot(fig)
        
    with tab4:
        st.subheader("Spectral Leakage Metrics")
        metrics_data = []
        for w_name in selected_windows:
            freqs, mag_db, w = compute_spectrum(signal, w_name, fs)
            m = compute_metrics(w, mag_db, freqs, fs)
            m["Window"] = w_name
            metrics_data.append(m)
            
        df = pd.DataFrame(metrics_data).set_index("Window")
        
        # Style to highlight best values
        def highlight_best(s):
            if s.name in ["Main-lobe width (Hz)", "Main-lobe width (bins)", "Scalloping loss (dB)"]:
                is_best = s == s.abs().min() if 'loss' in s.name else s == s.min()
            elif s.name in ["Highest side-lobe level (dB)", "Processing gain (dB)"]:
                is_best = s == s.min() if 'side-lobe' in s.name else s == s.max()
            else:
                return [''] * len(s)
            return ['background-color: darkgreen' if v else '' for v in is_best]
            
        st.dataframe(df.style.apply(highlight_best), use_container_width=True)
        
    with tab5:
        st.subheader("Leakage Visualization")
        cols = st.columns(min(len(selected_windows), 3))
        
        for i, w_name in enumerate(selected_windows):
            col_idx = i % 3
            with cols[col_idx]:
                fig, ax = plt.subplots(figsize=(6, 4))
                freqs, mag_db, _ = compute_spectrum(signal, w_name, fs)
                
                # Identify main lobe approx
                peak_idx = np.argmax(mag_db)
                left_idx = peak_idx
                while left_idx > 0 and mag_db[left_idx-1] < mag_db[left_idx]:
                    left_idx -= 1
                right_idx = peak_idx
                while right_idx < len(mag_db) - 1 and mag_db[right_idx+1] < mag_db[right_idx]:
                    right_idx += 1
                    
                ax.plot(freqs, mag_db, color=colors[i])
                ax.fill_between(freqs[left_idx:right_idx+1], -150, mag_db[left_idx:right_idx+1], 
                                color='lime', alpha=0.5, label='Main Lobe')
                ax.fill_between(freqs[:left_idx], -150, mag_db[:left_idx], 
                                color='red', alpha=0.3, label='Leakage/Side Lobes')
                ax.fill_between(freqs[right_idx+1:], -150, mag_db[right_idx+1:], 
                                color='red', alpha=0.3)
                
                span = 20 * (fs/N)
                ax.set_xlim([max(0, f1 - span), min(fs/2, f1 + span)])
                ax.set_ylim([-100, 5])
                ax.set_title(w_name)
                if i == 0:
                    ax.legend()
                st.pyplot(fig)

if __name__ == "__main__":
    main()
