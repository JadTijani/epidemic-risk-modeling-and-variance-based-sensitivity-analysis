# Beyond Classical Epidemiology: A Dynamic SEIQRL-F Model

An advanced computational framework that extends the classical SEIR architecture by integrating continuous public health interventions, logistics constraints, and behavioral variables through ordinary differential equations (ODEs). The framework incorporates a dynamic force of infection alongside a global sensitivity analysis driven by variance decomposition.

## Mathematical Structure

The system state is governed by eight coupled non-linear ordinary differential equations:

$$\begin{aligned}
\frac{dS}{dt} &= -\lambda(t)S -\alpha(t)S + \mu(t)L + \omega R - \phi_{vax}(t) + \phi_{evasion}(t) \\
\frac{dL}{dt} &= \alpha(t)S - \mu(t)L - \eta\lambda(t)L \\
\frac{dE}{dt} &= \lambda(t)S + \eta\lambda(t)L - \sigma E \\
\frac{dI}{dt} &= \sigma E - (\gamma + \delta_{eff}(t))I \\
\frac{dQ}{dt} &= \delta_{eff}(t)I - \gamma_Q Q \\
\frac{dR}{dt} &= \gamma I + \gamma_Q Q - \omega R + \phi_{vax}(t) - \phi_{evasion}(t) \\
\frac{dF_{acc}}{dt} &= r_{fatigue}\frac{L}{N} - r_{recov}F_{acc} \\
\frac{dEcon}{dt} &= C_L\frac{L}{N} + C_I\frac{I}{N}
\end{aligned}$$

## Methodology

* Global Sensitivity Analysis: Evaluation of a 12-dimensional parameter space using the Sobol method to decompose output variance, identifying standalone main effects ($S_1$) and high-order parameter couplings ($S_2$).
* Logistical Constraints: Integration of hospital infrastructure saturation limits and policy enforcement delays modeled via smooth, continuous sigmoids.
* Execution: Multiprocessed Monte Carlo sampling designed to evaluate thousands of coupled simulation tracks simultaneously.

## Installation & Usage

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```
