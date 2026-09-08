# MigraineAI Pro — Synthetic Training Datasets (v2, 50K rows)

⚠️ **These are SYNTHETIC datasets, not real patient data.** They were generated
with realistic statistical correlations (sleep deficit, barometric pressure
drops, stress, trigger count, hormonal shifts, dehydration → higher migraine
probability) so you can build and test the full ML pipeline end-to-end. Do
**not** present model outputs trained on this data as clinically validated —
swap in real user-collected or clinically sourced data before making any
health claims.

**v2 update:** regenerated at 50,000 rows per dataset (100,000 total) with a
stronger, cleaner signal-to-noise ratio than the original 20K batch — a
sanity-check logistic regression on the tabular set now hits **AUC 0.747**
(up from ~0.54 on v1), so your DL models have an actual learnable pattern to
find rather than mostly noise. The older 20K files are still in this folder
for comparison but the 50K files below are the ones to use going forward.

## 1. `tabular_sessions_50k.csv` (50,000 rows, 52 columns)
Session-level data — one row per symptom-logging session.
**Powers:** Module 2 (XGBoost classifier), Module 13 (Trigger Severity Index),
Module 4 (SHAP feature importance).

Key columns:
- `patient_id`, `session_id`, `timestamp`
- Symptom fields: `pain_intensity`, `pain_character`, `onset_speed`,
  `duration_hrs`, `phase`, plus binary flags (`nausea`, `vomiting`,
  `photophobia`, `phonophobia`, `visual_aura`, `neck_stiffness`,
  `dizziness`, `fatigue`)
- Lifestyle: `sleep_duration_hrs`, `sleep_quality_1to5`, `stress_level_0to10`,
  `water_intake_glasses`, `caffeine_mg`, `hormonal_cycle_day` (NaN for males)
- Vitals: `heart_rate_bpm`, `spo2_pct`, `skin_temp_c`, `barometric_pressure_hpa`
- Environment: `humidity_pct`, `ambient_light_lux`, `ambient_noise_db`,
  `weather_state`
- 20 binary trigger columns: `trigger_<name>` (matches your Trigger Matrix UI),
  plus `n_triggers_checked`
- **Label:** `migraine_occurred` (0/1) — target for the XGBoost classifier

## 2. `timeseries_daily_50k.csv` (50,000 rows, 18 columns)
Patient-day time-series — 1,000 synthetic patients × 50 consecutive days.
**Powers:** Module 1 (Onset Forecast Window), the LSTM half of Module 2,
Module 3 (Fusion Score calibration).

Key columns:
- `patient_id`, `date`, `day_index` (0–39, use to build rolling windows)
- Daily vitals/environment: `heart_rate_bpm`, `spo2_pct`, `skin_temp_c`,
  `barometric_pressure_hpa`, `pressure_delta_24h_hpa`, `humidity_pct`,
  `ambient_light_lux`, `ambient_noise_db`
- Lifestyle: `sleep_duration_hrs`, `stress_level_0to10`,
  `water_intake_glasses`, `n_triggers_today`
- **Labels (3 forecast horizons):** `migraine_onset_next_6h`,
  `migraine_onset_next_12h`, `migraine_onset_next_24h`

To train the LSTM, group by `patient_id`, sort by `day_index`, and build
sliding windows (e.g., 7-day lookback → predict next-day onset columns).

## Suggested next steps
1. Train XGBoost on `tabular_sessions_20k.csv` → feeds Module 2 + SHAP (Module 4)
2. Train LSTM on `timeseries_daily_20k.csv` (windowed) → feeds Module 1
3. Fit a simple logistic/weighted blend of both models' outputs on a held-out
   split → calibrates Module 3's Fusion Score weights
4. Once real user data starts flowing in from your app, retrain periodically
   and phase out the synthetic data — treat it as scaffolding, not the final
   model.
