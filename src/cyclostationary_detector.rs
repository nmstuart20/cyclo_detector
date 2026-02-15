use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// Configuration for the cyclostationary detector
#[derive(Clone, Debug)]
pub struct CycloDetectorConfig {
    /// FFT size for the channelizer (N')
    /// Controls frequency resolution: df = fs / fft_size
    pub fft_size: usize,

    /// Number of FFT blocks to accumulate for SCF estimation (P)
    /// More blocks = better SCF estimate, but longer observation time
    /// Total samples needed: fft_size * num_blocks (with no overlap)
    pub num_blocks: usize,

    /// Hop size between successive FFT blocks
    /// Typically fft_size / 4 for 75% overlap (Hann window)
    pub hop_size: usize,

    /// Candidate cycle frequencies (alpha) to test
    /// For known modulations:
    ///   - BPSK/QPSK: alpha = symbol_rate, 2*carrier_freq
    ///   - OFDM: alpha = 1/symbol_duration
    ///   - AM: alpha = 2*carrier_freq
    /// Set to None to search all resolvable alpha values
    pub alpha_candidates: Option<Vec<f64>>,

    /// Detection threshold for the SCF magnitude
    /// Peaks above this (relative to noise floor) indicate cyclostationary signal
    pub threshold: f64,

    /// Sample rate in Hz (needed to map bin indices to physical frequencies)
    pub sample_rate: f64,

    /// Window function to apply before each FFT
    pub window_type: WindowType,
}

impl Default for CycloDetectorConfig {
    fn default() -> Self {
        Self {
            fft_size: 256,
            num_blocks: 64,
            hop_size: 64, // 75% overlap with fft_size=256
            alpha_candidates: None,
            threshold: 6.0, // ~6 dB above noise floor
            sample_rate: 1.0,
            window_type: WindowType::Hann,
        }
    }
}

#[derive(Clone, Debug)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

/// Result of a cyclostationary detection
#[derive(Clone, Debug)]
pub struct CycloDetection {
    /// Detected cycle frequency in Hz
    pub alpha: f64,
    /// Spectral frequency where peak was found (Hz)
    pub freq: f64,
    /// SCF magnitude at the peak
    pub magnitude: f64,
    /// SNR of the peak relative to the estimated noise floor (dB)
    pub snr_db: f64,
    /// Sample index (center of the observation window)
    pub sample_index: usize,
}

/// The Spectral Correlation Function (2D: freq x cycle_freq)
#[derive(Clone, Debug)]
pub struct SpectralCorrelation {
    /// SCF magnitude matrix [alpha_index][freq_index]
    pub scf_mag: Vec<Vec<f64>>,
    /// Frequency axis values in Hz
    pub freq_axis: Vec<f64>,
    /// Cycle frequency axis values in Hz
    pub alpha_axis: Vec<f64>,
}

/// Cyclostationary detector using the FFT Accumulation Method (FAM)
///
/// The FAM computes the Spectral Correlation Function efficiently:
///   1. Segment input into overlapping blocks
///   2. Window and FFT each block -> channelized complex spectrogram
///   3. For each candidate cycle frequency alpha:
///      - Compute cross-spectral products: X_t(f) * conj(X_t(f - alpha))
///      - Average across time blocks
///      - The result is the SCF estimate at that alpha
///
/// Reference: Roberts, Brown, Loomis - "Computationally Efficient Algorithms
///            for Cyclic Spectral Analysis," IEEE SP Magazine, 1991
pub struct CycloDetector {
    config: CycloDetectorConfig,
    window: Vec<f64>,
    fft_planner: FftPlanner<f64>,
}

impl CycloDetector {
    pub fn new(config: CycloDetectorConfig) -> Self {
        let window = generate_window(&config.window_type, config.fft_size);
        let fft_planner = FftPlanner::new();

        Self {
            config,
            window,
            fft_planner,
        }
    }

    /// Run detection on a block of I&Q samples.
    /// Returns a list of detections (cycle frequencies with significant SCF peaks).
    pub fn detect(&mut self, iq_samples: &[Complex<f64>]) -> Vec<CycloDetection> {
        let scf = self.compute_scf(iq_samples);
        self.extract_detections(&scf, 0)
    }

    /// Run detection with a known sample offset (for snippet indexing)
    pub fn detect_at_offset(
        &mut self,
        iq_samples: &[Complex<f64>],
        sample_offset: usize,
    ) -> Vec<CycloDetection> {
        let scf = self.compute_scf(iq_samples);
        self.extract_detections(&scf, sample_offset)
    }

    /// Compute the full Spectral Correlation Function using the FAM.
    ///
    /// This is the core algorithm:
    ///   1. Channelize: sliding-window FFT to produce a time-frequency matrix
    ///   2. Cross-correlate: for each alpha, multiply shifted frequency bins
    ///   3. Accumulate: average across time to estimate SCF
    pub fn compute_scf(&mut self, iq_samples: &[Complex<f64>]) -> SpectralCorrelation {
        let n = self.config.fft_size;
        let hop = self.config.hop_size;

        // --- Step 1: Channelize (Short-Time FFT) ---
        // Produce a spectrogram: columns are time, rows are frequency
        let num_segments = (iq_samples.len().saturating_sub(n)) / hop + 1;
        let num_segments = num_segments.min(self.config.num_blocks);

        let fft = self.fft_planner.plan_fft_forward(n);

        // spectrogram[time_block][freq_bin] = complex value
        let mut spectrogram: Vec<Vec<Complex<f64>>> = Vec::with_capacity(num_segments);

        for t in 0..num_segments {
            let start = t * hop;
            if start + n > iq_samples.len() {
                break;
            }

            // Apply window and copy into FFT buffer
            let mut fft_buf: Vec<Complex<f64>> = iq_samples[start..start + n]
                .iter()
                .enumerate()
                .map(|(i, &s)| s * self.window[i])
                .collect();

            fft.process(&mut fft_buf);

            // Normalize by FFT size
            let norm = 1.0 / (n as f64);
            for val in &mut fft_buf {
                *val *= norm;
            }

            spectrogram.push(fft_buf);
        }

        let actual_segments = spectrogram.len();
        if actual_segments == 0 {
            return SpectralCorrelation {
                scf_mag: vec![],
                freq_axis: vec![],
                alpha_axis: vec![],
            };
        }

        // --- Step 2 & 3: Cross-spectral products and accumulation ---
        let df = self.config.sample_rate / n as f64; // frequency resolution
        let dalpha = self.config.sample_rate / (actual_segments as f64 * hop as f64); // cycle freq resolution

        // Build alpha axis: either from candidates or full search
        let alpha_axis = self.build_alpha_axis(dalpha);
        let freq_axis: Vec<f64> = (0..n)
            .map(|k| {
                // FFT bin to centered frequency
                let k_centered = if k < n / 2 { k as f64 } else { k as f64 - n as f64 };
                k_centered * df
            })
            .collect();

        // SCF[alpha_idx][freq_idx]
        let mut scf_mag: Vec<Vec<f64>> = vec![vec![0.0; n]; alpha_axis.len()];

        for (ai, &alpha) in alpha_axis.iter().enumerate() {
            // Convert alpha to a bin shift
            // shift = alpha / df (how many frequency bins to shift)
            let shift_bins = (alpha / df).round() as i64;

            // For each frequency bin f, compute:
            //   SCF(f, alpha) = (1/P) * sum_t [ X_t(f + alpha/2) * conj(X_t(f - alpha/2)) ]
            //
            // Using the shift approach:
            //   SCF(f, alpha) ≈ (1/P) * sum_t [ X_t(k + shift/2) * conj(X_t(k - shift/2)) ]
            let half_shift_pos = shift_bins / 2;
            let half_shift_neg = shift_bins - half_shift_pos; // handles odd shifts

            let mut accum: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];

            for seg in &spectrogram {
                for k in 0..n {
                    let k_pos = ((k as i64 + half_shift_pos).rem_euclid(n as i64)) as usize;
                    let k_neg = ((k as i64 - half_shift_neg).rem_euclid(n as i64)) as usize;

                    accum[k] += seg[k_pos] * seg[k_neg].conj();
                }
            }

            // Normalize by number of segments
            let norm = 1.0 / actual_segments as f64;
            for k in 0..n {
                scf_mag[ai][k] = (accum[k] * norm).norm();
            }
        }

        SpectralCorrelation {
            scf_mag,
            freq_axis,
            alpha_axis,
        }
    }

    /// Build the cycle frequency axis to search over
    fn build_alpha_axis(&self, dalpha: f64) -> Vec<f64> {
        match &self.config.alpha_candidates {
            Some(candidates) => candidates.clone(),
            None => {
                // Full search: alpha from -fs/2 to fs/2 in steps of dalpha
                // In practice, alpha=0 is the regular PSD, so we skip it
                // and focus on positive alpha values (SCF is conjugate symmetric)
                let max_alpha = self.config.sample_rate / 2.0;
                let num_alpha = (max_alpha / dalpha).floor() as usize;

                (1..=num_alpha).map(|i| i as f64 * dalpha).collect()
            }
        }
    }

    /// Extract detections by thresholding the SCF
    fn extract_detections(
        &self,
        scf: &SpectralCorrelation,
        sample_offset: usize,
    ) -> Vec<CycloDetection> {
        if scf.scf_mag.is_empty() {
            return vec![];
        }

        let mut detections = Vec::new();
        let n = scf.freq_axis.len();

        for (ai, alpha) in scf.alpha_axis.iter().enumerate() {
            let row = &scf.scf_mag[ai];

            // Estimate noise floor for this alpha slice using median
            let mut sorted_row: Vec<f64> = row.clone();
            sorted_row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let noise_floor = sorted_row[sorted_row.len() / 2]; // median as robust noise estimate

            if noise_floor <= 0.0 {
                continue;
            }

            // Find peaks above threshold
            for k in 0..n {
                let snr_linear = row[k] / noise_floor;
                let snr_db = 10.0 * snr_linear.log10();

                if snr_db > self.config.threshold {
                    // Simple peak check: must be local maximum
                    let left = if k > 0 { row[k - 1] } else { 0.0 };
                    let right = if k < n - 1 { row[k + 1] } else { 0.0 };

                    if row[k] >= left && row[k] >= right {
                        detections.push(CycloDetection {
                            alpha: *alpha,
                            freq: scf.freq_axis[k],
                            magnitude: row[k],
                            snr_db,
                            sample_index: sample_offset
                                + (self.config.num_blocks * self.config.hop_size) / 2,
                        });
                    }
                }
            }
        }

        // Sort by SNR descending
        detections.sort_by(|a, b| b.snr_db.partial_cmp(&a.snr_db).unwrap());
        detections
    }

    /// Convenience: compute the Spectral Coherence Function (normalized SCF)
    /// Useful for detection thresholding as it normalizes out the PSD shape.
    ///
    ///   C(f, alpha) = SCF(f, alpha) / sqrt(S(f + alpha/2) * S(f - alpha/2))
    ///
    /// where S(f) is the PSD estimate (i.e., the SCF at alpha=0).
    pub fn compute_coherence(&mut self, iq_samples: &[Complex<f64>]) -> SpectralCorrelation {
        let mut scf = self.compute_scf(iq_samples);

        // Estimate PSD: SCF at alpha=0
        let n = self.config.fft_size;
        let hop = self.config.hop_size;
        let fft = self.fft_planner.plan_fft_forward(n);
        let num_segments = (iq_samples.len().saturating_sub(n)) / hop + 1;
        let num_segments = num_segments.min(self.config.num_blocks);

        let mut psd: Vec<f64> = vec![0.0; n];
        for t in 0..num_segments {
            let start = t * hop;
            if start + n > iq_samples.len() {
                break;
            }

            let mut fft_buf: Vec<Complex<f64>> = iq_samples[start..start + n]
                .iter()
                .enumerate()
                .map(|(i, &s)| s * self.window[i])
                .collect();

            fft.process(&mut fft_buf);

            let norm = 1.0 / n as f64;
            for (k, val) in fft_buf.iter().enumerate() {
                psd[k] += val.norm_sqr() * norm * norm;
            }
        }
        let seg_norm = 1.0 / num_segments as f64;
        for val in &mut psd {
            *val *= seg_norm;
        }

        // Normalize SCF by PSD to get coherence
        let df = self.config.sample_rate / n as f64;
        for (ai, &alpha) in scf.alpha_axis.iter().enumerate() {
            let shift_bins = (alpha / df).round() as i64;
            let half_shift_pos = shift_bins / 2;
            let half_shift_neg = shift_bins - half_shift_pos;

            for k in 0..n {
                let k_pos = ((k as i64 + half_shift_pos).rem_euclid(n as i64)) as usize;
                let k_neg = ((k as i64 - half_shift_neg).rem_euclid(n as i64)) as usize;

                let denom = (psd[k_pos] * psd[k_neg]).sqrt();
                if denom > 1e-20 {
                    scf.scf_mag[ai][k] /= denom;
                } else {
                    scf.scf_mag[ai][k] = 0.0;
                }
            }
        }

        scf
    }
}

// --- Window Functions ---

fn generate_window(window_type: &WindowType, size: usize) -> Vec<f64> {
    match window_type {
        WindowType::Hann => (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos()))
            .collect(),
        WindowType::Hamming => (0..size)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (size - 1) as f64).cos())
            .collect(),
        WindowType::Blackman => (0..size)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / (size - 1) as f64;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),
        WindowType::Rectangular => vec![1.0; size],
    }
}

// --- Trait Integration Example ---
// This shows how the cyclostationary detector could plug into your existing
// detector trait system alongside power, MSK, NPSK detectors.
pub struct DetectionEvent {
    pub start_sample: usize,
    pub end_sample: usize,
    pub center_freq: f64,
    pub detector_type: String,
    pub metadata: DetectionMetadata,
}

pub enum DetectionMetadata {
    Power { power_db: f64 },
    Cyclostationary { alpha: f64, snr_db: f64 },
    Msk { /* ... */ },
}

/// Your existing detector trait (example)
pub trait Detector {
    fn detect_events(&mut self, iq_samples: &[Complex<f64>]) -> Vec<DetectionEvent>;
    fn name(&self) -> &str;
}

impl Detector for CycloDetector {
    fn detect_events(&mut self, iq_samples: &[Complex<f64>]) -> Vec<DetectionEvent> {
        let detections = self.detect(iq_samples);

        detections
            .into_iter()
            .map(|d| {
                // Convert cyclo detection to a detection event
                // The "event" spans the entire observation window since
                // cyclostationary analysis needs the full block
                let window_len = self.config.fft_size * self.config.num_blocks;
                let center = d.sample_index;
                let start = center.saturating_sub(window_len / 2);
                let end = start + window_len;

                DetectionEvent {
                    start_sample: start,
                    end_sample: end.min(iq_samples.len()),
                    center_freq: d.freq,
                    detector_type: "cyclostationary".to_string(),
                    metadata: DetectionMetadata::Cyclostationary {
                        alpha: d.alpha,
                        snr_db: d.snr_db,
                    },
                }
            })
            .collect()
    }

    fn name(&self) -> &str {
        "cyclostationary"
    }
}

// --- Usage Example ---

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a BPSK signal with known symbol rate for testing
    fn generate_bpsk(
        num_samples: usize,
        symbol_rate: f64,
        carrier_freq: f64,
        sample_rate: f64,
        snr_db: f64,
    ) -> Vec<Complex<f64>> {
        let samples_per_symbol = (sample_rate / symbol_rate).round() as usize;
        let num_symbols = num_samples / samples_per_symbol + 1;

        // Random BPSK symbols: +1 or -1
        let mut rng_state: u64 = 42;
        let symbols: Vec<f64> = (0..num_symbols)
            .map(|_| {
                // Simple LCG PRNG
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                if rng_state & 0x8000_0000_0000_0000 != 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        let mut signal: Vec<Complex<f64>> = Vec::with_capacity(num_samples);

        // Signal power
        let noise_power = 10.0_f64.powf(-snr_db / 10.0);
        let noise_std = (noise_power / 2.0).sqrt();

        let mut noise_rng: u64 = 123;

        for i in 0..num_samples {
            let sym_idx = i / samples_per_symbol;
            let sym = symbols[sym_idx.min(symbols.len() - 1)];

            let t = i as f64 / sample_rate;
            let carrier = Complex::new(
                (2.0 * PI * carrier_freq * t).cos(),
                (2.0 * PI * carrier_freq * t).sin(),
            );

            // Box-Muller for Gaussian noise (simplified)
            noise_rng = noise_rng.wrapping_mul(6364136223846793005).wrapping_add(3);
            let u1 = (noise_rng & 0xFFFF) as f64 / 65536.0 + 1e-10;
            noise_rng = noise_rng.wrapping_mul(6364136223846793005).wrapping_add(7);
            let u2 = (noise_rng & 0xFFFF) as f64 / 65536.0;
            let noise = Complex::new(
                noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos(),
                noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin(),
            );

            signal.push(carrier * sym + noise);
        }

        signal
    }

    #[test]
    fn test_bpsk_detection() {
        let sample_rate = 100_000.0; // 100 kHz
        let symbol_rate = 10_000.0;  // 10 kbaud
        let carrier_freq = 25_000.0; // 25 kHz
        let num_samples = 65_536;

        let signal = generate_bpsk(num_samples, symbol_rate, carrier_freq, sample_rate, 10.0);

        let config = CycloDetectorConfig {
            fft_size: 256,
            num_blocks: 128,
            hop_size: 64,
            // BPSK has cyclic features at alpha = symbol_rate and alpha = 2*carrier
            alpha_candidates: Some(vec![
                symbol_rate,
                2.0 * carrier_freq,
                5000.0,  // decoy - should not detect
                15000.0, // decoy - should not detect
            ]),
            threshold: 3.0,
            sample_rate,
            window_type: WindowType::Hann,
        };

        let mut detector = CycloDetector::new(config);
        let detections = detector.detect(&signal);

        println!("=== BPSK Cyclostationary Detections ===");
        for d in &detections {
            println!(
                "  alpha={:.0} Hz, freq={:.0} Hz, SNR={:.1} dB, mag={:.6}",
                d.alpha, d.freq, d.snr_db, d.magnitude
            );
        }

        // We should detect features at the symbol rate and/or 2*carrier
        assert!(!detections.is_empty(), "Should detect cyclostationary features in BPSK signal");

        // At least one detection should be at or near the symbol rate or 2*carrier
        let has_symbol_rate = detections.iter().any(|d| (d.alpha - symbol_rate).abs() < 100.0);
        let has_carrier = detections.iter().any(|d| (d.alpha - 2.0 * carrier_freq).abs() < 100.0);

        assert!(
            has_symbol_rate || has_carrier,
            "Should detect cyclic feature at symbol rate or 2x carrier"
        );
    }

    #[test]
    fn test_noise_only() {
        // Pure noise should produce no detections (or very few false alarms)
        let sample_rate = 100_000.0;
        let num_samples = 65_536;

        let mut rng: u64 = 999;
        let noise: Vec<Complex<f64>> = (0..num_samples)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u1 = (rng & 0xFFFF) as f64 / 65536.0 + 1e-10;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(3);
                let u2 = (rng & 0xFFFF) as f64 / 65536.0;
                Complex::new(
                    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos(),
                    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin(),
                )
            })
            .collect();

        let config = CycloDetectorConfig {
            fft_size: 256,
            num_blocks: 128,
            hop_size: 64,
            alpha_candidates: Some(vec![10_000.0, 25_000.0, 50_000.0]),
            threshold: 6.0,
            sample_rate,
            window_type: WindowType::Hann,
        };

        let mut detector = CycloDetector::new(config);
        let detections = detector.detect(&noise);

        println!("=== Noise-Only Detections (should be few/none) ===");
        for d in &detections {
            println!(
                "  alpha={:.0} Hz, freq={:.0} Hz, SNR={:.1} dB",
                d.alpha, d.freq, d.snr_db
            );
        }

        // With a reasonable threshold, noise should produce very few false alarms
        assert!(
            detections.len() <= 3,
            "Too many false alarms on pure noise: {}",
            detections.len()
        );
    }
}
