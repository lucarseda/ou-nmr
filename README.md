# Ornstein-Uhlenbeck Modeling of Pulsed NMR Relaxation

This project analyzes pulsed NMR relaxation data from light and heavy mineral oil using an Ornstein-Uhlenbeck stochastic magnetic-field model.

The analysis extracts $T_1$ from inversion-recovery measurements and $T_2$ from Hahn spin-echo measurements. These relaxation times are then used to infer parameters of a stochastic fluctuating magnetic field and compare the resulting OU echo-envelope prediction with the experimental spin-echo decay.

## Physical Background

In pulsed NMR, the longitudinal relaxation time $T_1$ characterizes recovery of magnetization along the external magnetic field, while the transverse relaxation time $T_2$ characterizes decay of coherent magnetization in the transverse plane.

For the Hahn spin-echo sequence,

$$
90^\circ - \tau - 180^\circ - \tau - \mathrm{echo},
$$

the measured echo amplitude is modeled as

$$
A_{\mathrm{echo}}(\tau) = A_0 e^{-2\tau/T_2} + C.
$$

The inversion-recovery signal is modeled as

$$
S(\tau) = A\left|1 - 2e^{-\tau/T_1}\right| + C.
$$

The OU model treats local magnetic-field fluctuations as a stationary stochastic process with correlation time \(\tau_c\). The inferred OU parameters are used to simulate fluctuating fields, stochastic phase accumulation, and echo-envelope decay.

## Repository Structure

```text
ou-nmr/
├── data/                    # Experimental T1 and T2 data
├── src/
│   └── ounmr/               # Reusable analysis code
├── scripts/                 # Reproducible analysis pipeline
├── results/
│   ├── figures/             # Generated figures
│   └── tables/              # Generated fit and parameter tables
├── notebooks/               # Original exploratory notebook
├── paper/                   # Written report
├── presentation/            # Presentation slides
├── supplement/              # Supplementary derivations
│   ├── ou-update-rule/             
│   └── hahn-echo-dephasing/  
└── README.md
