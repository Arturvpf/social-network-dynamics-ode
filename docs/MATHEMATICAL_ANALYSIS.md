# Mathematical & Numerical Analysis of the Coupled Social Network ODE System

This document provides the complete analytical derivations and numerical theory underlying the `social-network-dynamics-ode` package.

---

## 1. Governing Equations and State Space

Let $\mathbf{y}(t) = [E(t),\; T(t),\; D(t)]^\top \in \mathbb{R}_{\ge 0}^3$ denote user engagement $E(t)$, daily platform usage time $T(t)$, and consumed content diversity $D(t)$. For positive parameters $\alpha, \beta, \kappa, \gamma, \delta, \eta, \theta, T_{\max} > 0$, the autonomous vector field $\mathbf{f}(\mathbf{y})$ is:

$$
\begin{cases}
\dfrac{dE}{dt} = \alpha \tanh(T) - \beta E - \kappa D \\[8pt]
\dfrac{dT}{dt} = \gamma E \left(1 - \dfrac{T}{T_{\max}}\right) - \delta T \\[8pt]
\dfrac{dD}{dt} = \eta E - \theta D
\end{cases}
$$

---

## 2. Positivity (Filippov Projection) and Compact Absorbing Box

### Proposition 1 (Boundary Behavior on $\partial \mathbb{R}_{\ge 0}^3$)
On the boundary planes $T = 0$ and $D = 0$ with $E, T, D \ge 0$:

$$
\left.\frac{dT}{dt}\right|_{T=0} = \gamma E \ge 0, \qquad \left.\frac{dD}{dt}\right|_{D=0} = \eta E \ge 0.
$$

On the upper time plane $T = T_{\max}$:

$$
\left.\frac{dT}{dt}\right|_{T=T_{\max}} = -\delta T_{\max} < 0.
$$

On the face $E = 0$, however, $\left.\frac{dE}{dt}\right|_{E=0} = \alpha \tanh(T) - \kappa D$, which becomes negative when initial content overload $D$ exceeds $\frac{\alpha}{\kappa}\tanh(T)$. Applying the standard Filippov tangent-cone projection $\dot{E} = \max(0, \alpha\tanh(T) - \kappa D)$ on $E = 0$ renders $\mathbb{R}_{\ge 0}^3$ strictly positively invariant.

### Proposition 2 (Global Boundedness & Absorbing Set)
Since $\tanh(T) < 1$ and $D(t) \ge 0$ for all $t \ge 0$:
1. $\dot{E} \le \alpha - \beta E \implies \limsup_{t \to \infty} E(t) \le E_{\sup} := \dfrac{\alpha}{\beta}$.
2. $\dot{T} \le \gamma E_{\sup}\left(1 - \dfrac{T}{T_{\max}}\right) - \delta T \implies \limsup_{t \to \infty} T(t) \le T_{\sup} := \dfrac{\gamma E_{\sup} T_{\max}}{\gamma E_{\sup} + \delta T_{\max}} < T_{\max}$.
3. $\dot{D} \le \eta E_{\sup} - \theta D \implies \limsup_{t \to \infty} D(t) \le D_{\sup} := \dfrac{\eta}{\theta} E_{\sup}$.

Hence $\Omega = [0, E_{\sup}] \times [0, T_{\sup}] \times [0, D_{\sup}]$ is a compact globally absorbing positively invariant box.

---

## 3. Exact Activation Number $\mathcal{R}_0$ and Uniqueness of the Interior Equilibrium

### Theorem 1 (Threshold Dichotomy & Transcritical Bifurcation)
Define the dimensionless **Platform Activation Number**:

$$
\mathcal{R}_0 := \frac{\alpha \gamma \theta}{\delta (\beta \theta + \kappa \eta)}.
$$

1. **If $\mathcal{R}_0 \le 1$:** The trivial state $(0, 0, 0)$ is the **unique** equilibrium in $\mathbb{R}_{\ge 0}^3$ and is locally asymptotically stable for $\mathcal{R}_0 < 1$.
2. **If $\mathcal{R}_0 > 1$:** $(0, 0, 0)$ is an unstable hyperbolic saddle ($\det J(0,0,0) > 0$), and there exists a **unique** strictly positive interior equilibrium $(E^*, T^*, D^*) \in (0, E_{\sup}) \times (0, T_{\max}) \times (0, D_{\sup})$.

#### Proof
Setting $\dot{D} = 0$ gives $D^* = \frac{\eta}{\theta} E^*$. Substituting into $\dot{E} = 0$ yields:

$$
E^*(T^*) = \frac{\alpha \tanh(T^*)}{\beta + \kappa \eta / \theta} = \frac{\alpha \theta \tanh(T^*)}{\beta \theta + \kappa \eta}.
$$

For $T^* > 0$, substituting $E^*(T^*)$ into $\dot{T} = 0$ and dividing by $\delta T^*$ yields:

$$
\mathcal{R}_0 \, \phi(T^*) = 1, \qquad \text{where} \quad \phi(T) := \frac{\tanh(T)}{T}\left(1 - \frac{T}{T_{\max}}\right).
$$

For all $T \in (0, T_{\max})$, both $\frac{\tanh(T)}{T}$ and $\left(1 - \frac{T}{T_{\max}}\right)$ are positive and strictly decreasing, with $\lim_{T \to 0^+} \phi(T) = 1$ and $\phi(T_{\max}) = 0$. Therefore $\phi(T) \in (0, 1)$ is a strictly decreasing bijection from $(0, T_{\max})$ onto $(0, 1)$. Consequently, $\phi(T^*) = 1/\mathcal{R}_0$ admits a solution $T^* \in (0, T_{\max})$ **if and only if** $\mathcal{R}_0 > 1$, and when it exists, it is unique. $\blacksquare$

### Closed-Form Critical Bifurcation Thresholds ($\mathcal{R}_0 = 1$)
Solving $\mathcal{R}_0 = 1$ for each individual parameter yields the exact transcritical bifurcation points:

$$
\alpha_c = \frac{\delta(\beta\theta + \kappa\eta)}{\gamma\theta}, \quad
\beta_c = \frac{\alpha\gamma\theta - \kappa\delta\eta}{\delta\theta}, \quad
\kappa_c = \frac{\theta(\alpha\gamma - \beta\delta)}{\delta\eta}, \quad
\gamma_c = \frac{\delta(\beta\theta + \kappa\eta)}{\alpha\theta},
$$

$$
\delta_c = \frac{\alpha\gamma\theta}{\beta\theta + \kappa\eta}, \quad
\eta_c = \frac{\theta(\alpha\gamma - \beta\delta)}{\kappa\delta}, \quad
\theta_c = \frac{\kappa\delta\eta}{\alpha\gamma - \beta\delta}.
$$

---

## 4. Jacobian Spectrum and Routh–Hurwitz Stability

The analytical Jacobian at any state $(E, T, D)$ is:

$$
J(E, T, D) = \begin{pmatrix}
-\beta & \alpha \operatorname{sech}^2(T) & -\kappa \\[6pt]
\gamma \left(1 - \dfrac{T}{T_{\max}}\right) & -\dfrac{\gamma E}{T_{\max}} - \delta & 0 \\[6pt]
\eta & 0 & -\theta
\end{pmatrix}.
$$

Its characteristic polynomial $\det(\lambda I - J) = \lambda^3 + c_1 \lambda^2 + c_2 \lambda + c_3 = 0$ has coefficients:
- $c_1 = -\operatorname{tr}(J) = \beta + a_{22} + \theta > 0$, where $a_{22} = \frac{\gamma E}{T_{\max}} + \delta$,
- $c_2 = \beta a_{22} + \beta\theta + a_{22}\theta - a_{12} a_{21}$, where $a_{12} = \alpha\operatorname{sech}^2(T)$ and $a_{21} = \gamma(1 - T/T_{\max})$,
- $c_3 = -\det(J) = \theta(\beta a_{22} - a_{12}a_{21}) + \kappa\eta a_{22}$.

At the Origin $(0, 0, 0)$, $c_3 = \delta(\beta\theta + \kappa\eta)(1 - \mathcal{R}_0)$, so $c_3 < 0 \iff \mathcal{R}_0 > 1$.

At the baseline parameters ($\mathcal{R}_0 = 45/7 \approx 6.4286$):
- **Origin $(0, 0, 0)$:** $\lambda_1 = +0.076517$, $\lambda_2 = -0.167064$, $\lambda_3 = -0.059453$ (**unstable saddle**).
- **Interior Equilibrium $(4.499645,\; 5.070107,\; 5.999526)$:**
  - $c_1 = 0.168748 > 0$, $c_2 = 0.009184 > 0$, $c_3 = 0.000177 > 0$,
  - $\Delta_2 = c_1 c_2 - c_3 = +0.001373 > 0$,
  - $\lambda_{1,2} = -0.039990 \pm 0.019977i$, $\lambda_3 = -0.088768$ (**locally asymptotically stable spiral sink**).

*(Note: An earlier draft of the notebook inadvertently evaluated `J0_simpl` instead of `J(E_eq, T_eq, D_eq)` when inspecting the interior equilibrium, producing positive eigenvalues there by mistake. Evaluating the true Jacobian resolves that discrepancy completely.)*
