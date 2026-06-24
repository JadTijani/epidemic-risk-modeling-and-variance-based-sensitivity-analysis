```markdown
# Beyond Classical Epidemiology: A Dynamic SEIQRL-F Model

An advanced computational framework that extends the classical SEIR architecture by integrating continuous public health interventions, logistics constraints, and behavioral variables through ordinary differential equations (ODEs). The framework incorporates a dynamic force of infection alongside a global sensitivity analysis driven by variance decomposition.

## Mathematical Structure

The system state is governed by eight coupled non-linear ordinary differential equations:

```latex
dS/dt = -λ(t)S - α(t)S + μ(t)L + ωR - φ_vax(t) + φ_evasion(t)
dL/dt = α(t)S - μ(t)L - ηλ(t)L
dE/dt = λ(t)S + ηλ(t)L - σE
dI/dt = σE - (γ + δ_eff(t))I
dQ/dt = δ_eff(t)I - γ_Q Q
dR/dt = γI + γ_Q Q - ωR + φ_vax(t) - φ_evasion(t)
dF_acc/dt = r_fatigue(L/N) - r_recov F_acc
dEcon/dt = C_L(L/N) + C_I(I/N)
## Methodology

* **Global Sensitivity Analysis:** Evaluation of a 12-dimensional parameter space using the Sobol method to decompose output variance, identifying standalone main effects ($S_1$) and high-order parameter couplings ($S_2$).
* **Logistical Constraints:** Integration of hospital infrastructure saturation limits and policy enforcement delays modeled via smooth, continuous sigmoids.
* **Execution:** Multiprocessed Monte Carlo sampling designed to evaluate thousands of coupled simulation tracks simultaneously.

## Installation & Usage

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
