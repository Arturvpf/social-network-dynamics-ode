# Dinâmica de Redes Sociais via Equações Diferenciais Ordinárias (EDOs)

> **English Documentation:** [README.md](README.md) | **Provas Matemáticas:** [docs/MATHEMATICAL_ANALYSIS.md](docs/MATHEMATICAL_ANALYSIS.md) | **Notebook Acadêmico Completo:** [notebooks/social_network_ode_model.ipynb](notebooks/social_network_ode_model.ipynb)

Estudo matemático, numérico e computacional do comportamento de usuários em redes sociais modelado por um sistema acoplado de **Equações Diferenciais Ordinárias (EDOs) não lineares**, desenvolvido no contexto da disciplina de **Métodos Numéricos Computacionais** no **Centro de Informática da UFPE (CIn/UFPE)**.

---

## Visão Geral do Modelo

O sistema acompanha três variáveis de estado acopladas ao longo do tempo:
- **$E(t)$ — Engajamento do usuário**
- **$T(t)$ — Tempo diário de uso da plataforma** (limitado por $T_{\max} = 24\text{ h}$)
- **$D(t)$ — Diversidade do conteúdo consumido**

$$
\begin{cases}
\dfrac{dE}{dt} = \alpha \tanh(T) - \beta E - \kappa D \\[8pt]
\dfrac{dT}{dt} = \gamma E \left(1 - \dfrac{T}{T_{\max}}\right) - \delta T \\[8pt]
\dfrac{dD}{dt} = \eta E - \theta D
\end{cases}
$$

### Principais Resultados Analíticos e Numéricos

1. **Limiar Exato de Ativação ($\mathcal{R}_0$):** Demonstramos analiticamente que o número adimensional
   $$\mathcal{R}_0 = \frac{\alpha \gamma \theta}{\delta (\beta \theta + \kappa \eta)}$$
   governa uma **bifurcação transcrítica exata** em $\mathcal{R}_0 = 1$. Para $\mathcal{R}_0 \le 1$, a plataforma sofre abandono completo $(0,0,0)$; para $\mathcal{R}_0 > 1$, emerge um único equilíbrio interno positivo $(E^*, T^*, D^*)$.
2. **Estabilidade Assintótica via Routh–Hurwitz:** Nos parâmetros base ($\mathcal{R}_0 \approx 6{,}43$), a origem $(0,0,0)$ é um ponto de sela instável ($\lambda_1 = +0{,}0765$), enquanto o equilíbrio interno $(4{,}500;\; 5{,}070;\; 6{,}000)$ é um **foco espiral assintoticamente estável** ($\lambda_{1,2} = -0{,}0400 \pm 0{,}0200i$, $\lambda_3 = -0{,}0888$).
3. **Métodos Numéricos Implementados do Zero:** Euler Explícito $O(h)$, Heun/RK2 $O(h^2)$, Runge–Kutta Clássico RK4 $O(h^4)$ e Dormand–Prince adaptativo RK4(5), com comprovação empírica das ordens de convergência frente ao `DOP853` de 8ª ordem.
4. **Calibração Estatística de Parâmetros:** Estimação em duas etapas (colocação + *Trust-Region Reflective*) com intervalos de confiança de 95% e $R^2 = 0{,}999$.
5. **Extensões e Otimização de Recomendação:** Otimização do peso de exploração algorítmica $u^* \in [0, 1]$, extensão 4D com fadiga cognitiva $F(t)$ e modelo acoplado em rede social *Small-World* ($N = 15$ nós).
6. **Simulador Web Interativo (`simulator/index.html`):** Laboratório interativo em HTML5 Canvas (bilíngue `PT-BR` / `EN`) com retrato de fase clicável, órbita 3D rotacionável e comparação ao vivo entre Euler e RK4.

---

## Como Executar

```bash
pip install -e ".[dev]"

# Executar suíte de 33 testes automatizados
pytest -v

# Relatório analítico de equilíbrios e estabilidade
social-ode equilibrium

# Benchmark de convergência numérica (Euler, RK2, RK4)
social-ode convergence

# Calibração de parâmetros e otimização de políticas
social-ode calibrate
social-ode policies

# Iniciar o Simulador Web Interativo em http://127.0.0.1:8080/index.html
social-ode serve-simulator

# Abrir o Notebook Acadêmico (100% em Português)
jupyter notebook notebooks/social_network_ode_model.ipynb
```

---

## Autores (Grupo 05 — CIn/UFPE)

- Andreywid Yago Lima de Souza (`ayls@cin.ufpe.br`)
- Artur Vinicius Pereira Fernandes (`arturvpf3@gmail.com`)
- Breno Ramos de Souza Gomes (`brsg@cin.ufpe.br`)
- Valter Sanches Martins Neto (`vsmn@cin.ufpe.br`)
- Manoel Lira de Carvalho (`mlc6@cin.ufpe.br`)
