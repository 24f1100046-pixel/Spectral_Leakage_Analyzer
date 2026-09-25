# Spectral Leakage Analyzer

A complete Streamlit web application for analyzing and visualizing the effects of different window functions on spectral leakage in signal processing.

## Features
- **Signal Generation**: Pure Sinusoid, Sum of Sinusoids, Chirp Signal, Custom.
- **Window Selection**: Compare various window functions like Rectangular, Hann, Hamming, Blackman, Bartlett, Kaiser, Flat Top.
- **Visualizations**: 
  - Time Domain
  - Window Shapes & Frequency Responses
  - Magnitude Spectra Comparison
  - Spectral Leakage Metrics Table
  - Leakage Visualization

## Installation

1. Clone or download this repository.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment
You can easily deploy this application on [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting your GitHub repository and deploying `app.py`.
