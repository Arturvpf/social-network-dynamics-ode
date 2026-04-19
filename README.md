# Social Network Dynamics via Ordinary Differential Equations

A mathematical and numerical study of user behavior in social networks modeled as a system of nonlinear ordinary differential equations (ODEs). The project formulates an interaction model between three coupled state variables — user engagement, platform usage time, and content diversity — and analyzes its qualitative behavior through numerical simulation, parameter sensitivity analysis, and bifurcation diagrams.

This work was developed for the Computational Numerical Methods course at CIn/UFPE, in the context of modeling recommendation-algorithm dynamics from a continuous-time perspective.

## Motivation

Digital platforms shape user behavior through feedback loops between engagement, exposure time, and content variety. These dynamics are notoriously hard to study with purely data-driven methods when empirical calibration data is unavailable. A continuous ODE model provides an interpretable lens: each term represents a hypothesis about how one variable reinforces or suppresses another, and its qualitative behavior (fixed points, oscillations, bifurcations) can be compared against theoretical expectations.

## Model

Three state variables are tracked over time:

- **E(t)** — user engagement.
- **T(t)** — time spent on the platform.
- **D(t)** — diversity of consumed content.

The dynamics are governed by a nonlinear coupled system of the form:

```
dE/dt = alpha * tanh(T) - beta * E - kappa * D
dT/dt = gamma * E * (1 - T / T_max) - delta * T
dD/dt = eta * T - theta * D
```

With non-negativity guards on the state variables for numerical stability. The parameters `alpha, beta, kappa, gamma, delta, eta, theta, T_max` control the strength of each interaction and are chosen to produce behavior qualitatively consistent with the theoretical hypotheses.

## Methodology

1. **Formulation** — derivation of the ODE system from behavioral hypotheses, with a review of supporting literature.
2. **Numerical integration** — the system is integrated using SciPy's `solve_ivp` (adaptive Runge-Kutta), for a range of initial conditions.
3. **Parameter sensitivity analysis** — each parameter is swept across a reasonable interval to identify regions of the parameter space where the qualitative behavior is preserved.
4. **Bifurcation diagrams** — peaks of the time-series are extracted with `scipy.signal.find_peaks` to visualize how the asymptotic behavior changes with each parameter.
5. **Discussion** — interpretation of stable equilibria, oscillatory regimes, and connections to the recommendation-system literature.

## Repository Layout

```
.
├── README.md
├── .gitignore
└── notebooks/
    └── social_network_ode_model.ipynb   # Formulation, simulation, and analysis
```

## How to Run

The notebook runs on Python 3.10+ with the scientific stack:

```bash
pip install numpy scipy matplotlib jupyter
jupyter notebook notebooks/social_network_ode_model.ipynb
```

All simulations are self-contained; no external data is required.

## Authors

- Andreywid Yago Lima de Souza (ayls@cin.ufpe.br)
- Artur Vinicius Pereira Fernandes (arturvpf3@gmail.com)
- Breno Ramos de Souza Gomes (brsg@cin.ufpe.br)
- Valter Sanches Martins Neto (vsmn@cin.ufpe.br)
- Manoel Lira de Carvalho (mlc6@cin.ufpe.br)

Centro de Informatica (CIn), Universidade Federal de Pernambuco (UFPE).

## References

- Virtanen et al. *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.* Nature Methods, 2020.
- Strogatz, S. H. *Nonlinear Dynamics and Chaos.* Westview Press.
