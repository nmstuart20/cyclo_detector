%% Cyclostationary Detector vs Energy Detector Performance Comparison
% Compares detection performance (Pd vs SNR, ROC curves) for:
%   1. Energy Detector (radiometer)
%   2. Cyclostationary Feature Detector (spectral correlation based)
%
% Signal model: Wideband BPSK signal with RRC pulse shaping and known
% symbol rate (exhibits cyclostationary features at the symbol rate),
% embedded in AWGN. Signal occupies ~54% of Nyquist bandwidth.
%
% Author: Nick | Generated with Claude
% Date: 2025

clear; clc; close all;

%% ===================== Simulation Parameters =====================
fs = 1e6;               % Sampling frequency (Hz) — 1 MHz
fb = 500e3;             % Symbol rate (Hz) — 500 ksym/s wideband BPSK
fc = 250e3;             % Carrier frequency (Hz)
rolloff = 0.35;         % RRC pulse shaping rolloff factor
span = 8;               % RRC filter span in symbols
N = 8192;               % Number of samples per observation
Pfa_target = 0.01;      % Target probability of false alarm
num_trials = 5000;      % Monte Carlo trials per SNR point
SNR_dB = -20:1:5;       % SNR range to sweep (dB)

% Cyclostationary detector parameters
% BPSK with RRC pulse shaping exhibits cyclic features at:
%   alpha = 1/T (symbol rate) and alpha = 2*fc (double carrier)
% We target alpha0 = fb (the symbol rate) which is the strongest feature
alpha0 = fb;            % Cyclic frequency: symbol rate for RRC-shaped BPSK
Nw = 256;               % FFT window length for spectral correlation
overlap = 192;          % Window overlap (75%)
delta_alpha = fs / N;   % Cyclic frequency resolution

% Derived parameters
sps = round(fs / fb);   % Samples per symbol
bw_null2null = fb * (1 + rolloff);  % Occupied bandwidth (one-sided)

fprintf('=== Cyclostationary vs Energy Detector Comparison ===\n');
fprintf('Signal: Wideband BPSK (RRC, beta=%.2f)\n', rolloff);
fprintf('fs=%.0f kHz | fb=%.0f kHz | fc=%.0f kHz | sps=%d\n', fs/1e3, fb/1e3, fc/1e3, sps);
fprintf('Occupied BW: ~%.0f kHz (%.0f%% of Nyquist)\n', 2*bw_null2null/1e3, 100*2*bw_null2null/(fs/2));
fprintf('Samples: %d | Trials: %d | Pfa target: %.3f\n', N, num_trials, Pfa_target);
fprintf('Cyclic freq (alpha0): %.0f Hz\n\n', alpha0);

%% ===================== Design RRC Pulse Shape Filter =====================
% Root-raised-cosine filter for wideband BPSK
rrc_filt = rcosdesign(rolloff, span, sps, 'sqrt');
rrc_filt = rrc_filt / norm(rrc_filt);  % Normalize energy
fprintf('RRC filter: %d taps | span=%d symbols | rolloff=%.2f\n', length(rrc_filt), span, rolloff);

%% ===================== Generate Wideband BPSK Signal =====================
function x = generate_bpsk_signal(N, fs, fb, fc, snr_db, sps, rrc_filt)
    % Generate wideband RRC-shaped BPSK signal at given SNR in AWGN
    filt_delay = (length(rrc_filt) - 1) / 2;
    num_symbols = ceil((N + 2 * filt_delay) / sps) + 10;
    bits = 2 * randi([0 1], num_symbols, 1) - 1;  % {-1, +1}

    % Upsample
    baseband_up = upsample(bits, sps);

    % RRC pulse shape filtering
    baseband = conv(baseband_up, rrc_filt, 'same');

    % Trim to N samples (account for filter transients)
    start_idx = filt_delay + 1;
    baseband = baseband(start_idx:start_idx + N - 1);

    % Modulate to carrier
    t = (0:N-1)' / fs;
    signal = baseband .* cos(2 * pi * fc * t);

    % Normalize signal power to 1
    signal = signal / sqrt(mean(signal.^2));

    % Add noise at specified SNR
    noise_power = 10^(-snr_db / 10);
    noise = sqrt(noise_power) * randn(N, 1);
    x = signal + noise;
end

%% ===================== Energy Detector =====================
function T = energy_detector(x)
    % Test statistic: normalized energy
    T = sum(abs(x).^2) / length(x);
end

%% ===================== Cyclostationary Detector =====================
function T = cyclo_detector(x, fs, alpha0, Nw, overlap)
    % Spectral Correlation Function (SCF) based detector
    % Estimates the magnitude of the spectral correlation at cyclic
    % frequency alpha0, normalized by the zero-cycle PSD.
    %
    % Uses the frequency-smoothing method (FFT Accumulation Method).

    N = length(x);
    hop = Nw - overlap;
    num_blocks = floor((N - Nw) / hop) + 1;

    % Window
    win = hamming(Nw);

    % Frequency shift by +/- alpha0/2
    t_win = (0:Nw-1)' / fs;
    shift_pos = exp(+1j * pi * alpha0 * t_win);
    shift_neg = exp(-1j * pi * alpha0 * t_win);

    Sxa = zeros(Nw, 1);   % Cross-spectral estimate at alpha0
    Sx0 = zeros(Nw, 1);   % PSD estimate at alpha=0

    for k = 0:num_blocks-1
        idx = (1:Nw)' + k * hop;
        seg = x(idx) .* win;

        % Shifted segments
        seg_pos = seg .* shift_pos;
        seg_neg = seg .* shift_neg;

        X_pos = fft(seg_pos, Nw);
        X_neg = fft(seg_neg, Nw);
        X0    = fft(seg, Nw);

        % Accumulate cross-spectral density at alpha0
        Sxa = Sxa + X_pos .* conj(X_neg);
        % Accumulate PSD at alpha=0
        Sx0 = Sx0 + X0 .* conj(X0);
    end

    Sxa = Sxa / num_blocks;
    Sx0 = Sx0 / num_blocks;

    % Spectral coherence (degree of cyclostationarity)
    % Test statistic: max of |S_x^alpha(f)|^2 / S_x^0(f) over frequency
    % This is related to the cyclic domain profile
    coherence = abs(Sxa).^2 ./ (Sx0 + eps);
    T = max(coherence);
end

%% ===================== Threshold Calibration (H0: noise only) =====================
fprintf('Calibrating thresholds under H0 (noise only)...\n');

T_energy_h0 = zeros(num_trials, 1);
T_cyclo_h0  = zeros(num_trials, 1);

for trial = 1:num_trials
    noise = randn(N, 1);

    T_energy_h0(trial) = energy_detector(noise);
    T_cyclo_h0(trial)  = cyclo_detector(noise, fs, alpha0, Nw, overlap);
end

% Set thresholds for target Pfa
T_energy_h0_sorted = sort(T_energy_h0, 'descend');
T_cyclo_h0_sorted  = sort(T_cyclo_h0, 'descend');

thresh_idx = ceil(Pfa_target * num_trials);
threshold_energy = T_energy_h0_sorted(thresh_idx);
threshold_cyclo  = T_cyclo_h0_sorted(thresh_idx);

% Verify Pfa
Pfa_energy_actual = mean(T_energy_h0 > threshold_energy);
Pfa_cyclo_actual  = mean(T_cyclo_h0 > threshold_cyclo);

fprintf('  Energy Detector  — threshold: %.4f | actual Pfa: %.4f\n', threshold_energy, Pfa_energy_actual);
fprintf('  Cyclo Detector   — threshold: %.4f | actual Pfa: %.4f\n', threshold_cyclo, Pfa_cyclo_actual);

%% ===================== Pd vs SNR Sweep =====================
fprintf('\nRunning Pd vs SNR sweep...\n');

Pd_energy = zeros(length(SNR_dB), 1);
Pd_cyclo  = zeros(length(SNR_dB), 1);

for s = 1:length(SNR_dB)
    snr = SNR_dB(s);
    det_energy = 0;
    det_cyclo  = 0;

    for trial = 1:num_trials
        x = generate_bpsk_signal(N, fs, fb, fc, snr, sps, rrc_filt);

        % Energy detector
        if energy_detector(x) > threshold_energy
            det_energy = det_energy + 1;
        end

        % Cyclostationary detector
        if cyclo_detector(x, fs, alpha0, Nw, overlap) > threshold_cyclo
            det_cyclo = det_cyclo + 1;
        end
    end

    Pd_energy(s) = det_energy / num_trials;
    Pd_cyclo(s)  = det_cyclo / num_trials;

    fprintf('  SNR = %+3d dB | Pd_energy = %.3f | Pd_cyclo = %.3f\n', snr, Pd_energy(s), Pd_cyclo(s));
end

%% ===================== ROC Curves at Select SNRs =====================
fprintf('\nGenerating ROC curves...\n');

SNR_roc = [-15, -10, -5, 0];  % SNR values for ROC curves
num_roc_trials = 3000;
num_thresh_pts = 200;

ROC = struct();

for r = 1:length(SNR_roc)
    snr = SNR_roc(r);
    fprintf('  ROC at SNR = %+d dB...\n', snr);

    % Collect test statistics under H0 and H1
    T_e_h0 = zeros(num_roc_trials, 1);
    T_e_h1 = zeros(num_roc_trials, 1);
    T_c_h0 = zeros(num_roc_trials, 1);
    T_c_h1 = zeros(num_roc_trials, 1);

    for trial = 1:num_roc_trials
        % H0: noise only
        noise = randn(N, 1);
        T_e_h0(trial) = energy_detector(noise);
        T_c_h0(trial) = cyclo_detector(noise, fs, alpha0, Nw, overlap);

        % H1: signal + noise
        x = generate_bpsk_signal(N, fs, fb, fc, snr, sps, rrc_filt);
        T_e_h1(trial) = energy_detector(x);
        T_c_h1(trial) = cyclo_detector(x, fs, alpha0, Nw, overlap);
    end

    % Sweep thresholds to generate ROC
    % Energy detector
    e_min = min([T_e_h0; T_e_h1]);
    e_max = max([T_e_h0; T_e_h1]);
    thresholds_e = linspace(e_min, e_max, num_thresh_pts);
    Pfa_e = zeros(num_thresh_pts, 1);
    Pd_e  = zeros(num_thresh_pts, 1);
    for k = 1:num_thresh_pts
        Pfa_e(k) = mean(T_e_h0 > thresholds_e(k));
        Pd_e(k)  = mean(T_e_h1 > thresholds_e(k));
    end

    % Cyclo detector
    c_min = min([T_c_h0; T_c_h1]);
    c_max = max([T_c_h0; T_c_h1]);
    thresholds_c = linspace(c_min, c_max, num_thresh_pts);
    Pfa_c = zeros(num_thresh_pts, 1);
    Pd_c  = zeros(num_thresh_pts, 1);
    for k = 1:num_thresh_pts
        Pfa_c(k) = mean(T_c_h0 > thresholds_c(k));
        Pd_c(k)  = mean(T_c_h1 > thresholds_c(k));
    end

    ROC(r).snr = snr;
    ROC(r).Pfa_e = Pfa_e; ROC(r).Pd_e = Pd_e;
    ROC(r).Pfa_c = Pfa_c; ROC(r).Pd_c = Pd_c;
end

%% ===================== Spectral Correlation Visualization =====================
fprintf('\nComputing SCF surface for visualization...\n');

snr_viz = 0;  % SNR for visualization
x_viz = generate_bpsk_signal(N, fs, fb, fc, snr_viz, sps, rrc_filt);

% --- Figure 3a: PSD showing wideband signal ---
figure('Position', [100 100 800 400], 'Color', 'w');
[pxx, f_psd] = pwelch(x_viz, hamming(512), 256, 1024, fs, 'centered');
plot(f_psd/1e3, 10*log10(pxx), 'b', 'LineWidth', 1.5);
hold on;
% Mark signal bandwidth
xline((fc - bw_null2null)/1e3, 'r--', 'LineWidth', 1);
xline((fc + bw_null2null)/1e3, 'r--', 'LineWidth', 1);
xline(-(fc - bw_null2null)/1e3, 'r--', 'LineWidth', 1);
xline(-(fc + bw_null2null)/1e3, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off;
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('PSD (dB/Hz)', 'FontSize', 12);
title(sprintf('Power Spectral Density — Wideband BPSK (RRC \\beta=%.2f)\nf_b=%.0f kHz | f_c=%.0f kHz | BW_{null}=%.0f kHz | SNR=%+d dB', ...
    rolloff, fb/1e3, fc/1e3, 2*bw_null2null/1e3, snr_viz), 'FontSize', 13);
legend('PSD', sprintf('Signal BW (±%.0f kHz)', bw_null2null/1e3), 'Location', 'south', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);
saveas(gcf, 'psd_wideband.png');

% Compute SCF over a range of cyclic frequencies
alpha_range = linspace(0, 3 * fb, 150);
f_axis = (-Nw/2:Nw/2-1) * (fs / Nw);

SCF_surface = zeros(length(alpha_range), Nw);
win = hamming(Nw);
hop = Nw - overlap;
num_blocks = floor((N - Nw) / hop) + 1;
t_win = (0:Nw-1)' / fs;

for ai = 1:length(alpha_range)
    alpha = alpha_range(ai);
    shift_pos = exp(+1j * pi * alpha * t_win);
    shift_neg = exp(-1j * pi * alpha * t_win);

    Sxa = zeros(Nw, 1);
    for k = 0:num_blocks-1
        idx = (1:Nw)' + k * hop;
        seg = x_viz(idx) .* win;
        X_pos = fft(seg .* shift_pos, Nw);
        X_neg = fft(seg .* shift_neg, Nw);
        Sxa = Sxa + X_pos .* conj(X_neg);
    end
    Sxa = Sxa / num_blocks;
    SCF_surface(ai, :) = fftshift(abs(Sxa));
end

%% ===================== Plotting =====================
fprintf('\nGenerating plots...\n');

% --- Figure 1: Pd vs SNR ---
figure('Position', [100 100 800 500], 'Color', 'w');
plot(SNR_dB, Pd_energy, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', 'Energy Detector');
hold on;
plot(SNR_dB, Pd_cyclo, 'r-s', 'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', 'Cyclostationary Detector');
yline(Pfa_target, 'k--', 'LineWidth', 1, 'DisplayName', sprintf('P_{fa} = %.2f', Pfa_target));
hold off;
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Probability of Detection (P_d)', 'FontSize', 12);
title(sprintf('Detection Performance: Cyclostationary vs Energy Detector\nWideband BPSK (RRC \\beta=%.2f) | N=%d | P_{fa}=%.2f | %d trials', rolloff, N, Pfa_target, num_trials), 'FontSize', 13);
legend('Location', 'southeast', 'FontSize', 11);
grid on;
xlim([SNR_dB(1) SNR_dB(end)]);
ylim([0 1.05]);
set(gca, 'FontSize', 11);

saveas(gcf, 'pd_vs_snr.png');

% --- Figure 2: ROC Curves ---
figure('Position', [100 100 900 700], 'Color', 'w');
colors = lines(length(SNR_roc));
for r = 1:length(SNR_roc)
    subplot(2, 2, r);
    plot(ROC(r).Pfa_e, ROC(r).Pd_e, 'b-', 'LineWidth', 2, 'DisplayName', 'Energy');
    hold on;
    plot(ROC(r).Pfa_c, ROC(r).Pd_c, 'r-', 'LineWidth', 2, 'DisplayName', 'Cyclostationary');
    plot([0 1], [0 1], 'k--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    hold off;
    xlabel('P_{fa}', 'FontSize', 10);
    ylabel('P_d', 'FontSize', 10);
    title(sprintf('ROC @ SNR = %+d dB', SNR_roc(r)), 'FontSize', 11);
    legend('Location', 'southeast', 'FontSize', 9);
    grid on;
    xlim([0 1]); ylim([0 1]);
    set(gca, 'FontSize', 10);
end
sgtitle('Receiver Operating Characteristic Curves', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, 'roc_curves.png');

% --- Figure 3: Spectral Correlation Function Surface ---
figure('Position', [100 100 800 500], 'Color', 'w');
imagesc(f_axis/1e3, alpha_range/1e3, 10*log10(SCF_surface + eps));
set(gca, 'YDir', 'normal');
colorbar;
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Cyclic Frequency \alpha (kHz)', 'FontSize', 12);
title(sprintf('Spectral Correlation Function |S_x^\\alpha(f)| (dB)\nWideband BPSK (RRC \\beta=%.2f) @ SNR = %+d dB | \\alpha_0 = %.0f kHz (1/T)', rolloff, snr_viz, alpha0/1e3), 'FontSize', 13);
colormap(jet);

% Mark the expected cyclic frequency
hold on;
yline(alpha0/1e3, 'w--', 'LineWidth', 2);
text(f_axis(end)/1e3 * 0.6, alpha0/1e3 + 15, '\alpha_0 = 1/T', 'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');
hold off;
set(gca, 'FontSize', 11);

saveas(gcf, 'scf_surface.png');

% --- Figure 4: Test Statistic Distributions ---
figure('Position', [100 100 900 400], 'Color', 'w');

% Regenerate stats at a representative SNR for visualization
snr_dist = -8;
T_e_h0_dist = zeros(num_trials, 1);
T_e_h1_dist = zeros(num_trials, 1);
T_c_h0_dist = zeros(num_trials, 1);
T_c_h1_dist = zeros(num_trials, 1);

fprintf('  Computing test statistic distributions at SNR=%+d dB...\n', snr_dist);
for trial = 1:num_trials
    noise = randn(N, 1);
    T_e_h0_dist(trial) = energy_detector(noise);
    T_c_h0_dist(trial) = cyclo_detector(noise, fs, alpha0, Nw, overlap);

    x = generate_bpsk_signal(N, fs, fb, fc, snr_dist, sps, rrc_filt);
    T_e_h1_dist(trial) = energy_detector(x);
    T_c_h1_dist(trial) = cyclo_detector(x, fs, alpha0, Nw, overlap);
end

subplot(1, 2, 1);
histogram(T_e_h0_dist, 50, 'FaceColor', [0.3 0.3 0.8], 'FaceAlpha', 0.6, 'DisplayName', 'H_0 (noise)');
hold on;
histogram(T_e_h1_dist, 50, 'FaceColor', [0.8 0.3 0.3], 'FaceAlpha', 0.6, 'DisplayName', sprintf('H_1 (SNR=%+ddB)', snr_dist));
xline(threshold_energy, 'k--', 'LineWidth', 2, 'DisplayName', 'Threshold');
hold off;
xlabel('Test Statistic', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('Energy Detector', 'FontSize', 12);
legend('FontSize', 9);
grid on;

subplot(1, 2, 2);
histogram(T_c_h0_dist, 50, 'FaceColor', [0.3 0.3 0.8], 'FaceAlpha', 0.6, 'DisplayName', 'H_0 (noise)');
hold on;
histogram(T_c_h1_dist, 50, 'FaceColor', [0.8 0.3 0.3], 'FaceAlpha', 0.6, 'DisplayName', sprintf('H_1 (SNR=%+ddB)', snr_dist));
xline(threshold_cyclo, 'k--', 'LineWidth', 2, 'DisplayName', 'Threshold');
hold off;
xlabel('Test Statistic', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('Cyclostationary Detector', 'FontSize', 12);
legend('FontSize', 9);
grid on;

sgtitle(sprintf('Test Statistic Distributions (SNR = %+d dB)', snr_dist), 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, 'test_stat_distributions.png');

%% ===================== Summary Statistics =====================
fprintf('\n========== RESULTS SUMMARY ==========\n');
fprintf('%-8s | %-12s | %-12s | %-6s\n', 'SNR(dB)', 'Pd_Energy', 'Pd_Cyclo', 'Gain');
fprintf('%s\n', repmat('-', 1, 48));

for s = 1:length(SNR_dB)
    gain_str = '';
    if Pd_cyclo(s) > Pd_energy(s) + 0.01
        gain_str = 'CYCLO';
    elseif Pd_energy(s) > Pd_cyclo(s) + 0.01
        gain_str = 'ENERGY';
    else
        gain_str = 'TIE';
    end
    fprintf('%+4d     | %10.3f   | %10.3f   | %s\n', SNR_dB(s), Pd_energy(s), Pd_cyclo(s), gain_str);
end

% Find approximate SNR for Pd=0.9
interp_snr_e = interp1(Pd_energy(Pd_energy > 0.05 & Pd_energy < 0.99), ...
                       SNR_dB(Pd_energy > 0.05 & Pd_energy < 0.99), 0.9, 'linear');
interp_snr_c = interp1(Pd_cyclo(Pd_cyclo > 0.05 & Pd_cyclo < 0.99), ...
                       SNR_dB(Pd_cyclo > 0.05 & Pd_cyclo < 0.99), 0.9, 'linear');

fprintf('\nSNR required for Pd = 0.9:\n');
if ~isnan(interp_snr_e)
    fprintf('  Energy Detector:         %.1f dB\n', interp_snr_e);
end
if ~isnan(interp_snr_c)
    fprintf('  Cyclostationary Detector: %.1f dB\n', interp_snr_c);
end
if ~isnan(interp_snr_e) && ~isnan(interp_snr_c)
    fprintf('  Cyclo advantage:          %.1f dB\n', interp_snr_e - interp_snr_c);
end

fprintf('\nAll plots saved. Done!\n');
