# Multi-stream flow in a gas with a sinusoidal velocity profile

A gas of rigid spheres (mass $m$, radius $r$, number density $n$, temperature $T$) is
spatially uniform at $t=0$ but carries a macroscopic velocity

$$v_x(x)\big|_{t=0}=v_0\sin(kx),\qquad v_0\gg v_T\equiv\sqrt{k_BT/m}.$$

**When does the local velocity distribution $f(v_x)$ develop two local maxima, each at least
twice the minimum between them?**

## Answer

$$\boxed{\;v_0\;\gtrsim\;\frac{\nu}{k}\;\sim\;\frac{n\sigma}{k}\sqrt{\frac{k_BT}{m}}
\;\sim\;\frac{n r^{2}}{k}\sqrt{\frac{k_BT}{m}}\;}
\qquad \sigma=4\pi r^{2},\quad \nu=\sqrt2\,n\sigma\bar v$$

equivalently $k v_0\gtrsim\nu$ (steepening beats collisions), or
$\mathrm{Ma}\cdot k\lambda\gtrsim1$, or $\lambda\gtrsim v_T/(kv_0)$ (a molecule crosses the
whole incipient fold without colliding). Combined with the given $v_0\gg v_T$:
$v_0\gg v_T\max\!\left(1,1/(k\lambda)\right)$.

Reasoning in one paragraph: collisionlessly $f(x,v_x,t)=f_0(x-v_xt,v_x)$, so the profile
steepens like a Burgers wave and folds over in phase space at $t_b=1/(kv_0)$ at $x=\pi/k$. The
fold is the multi-stream region; since $v_0\gg v_T$ it is resolved into separate peaks after
only $t_*=t_b[1+O((v_T/v_0)^{2/3})]\simeq t_b$, and it is then $\Delta x\sim v_T/(kv_0)$
thick. Collisions restore a single-peaked local Maxwellian, so multi-stream flow needs
$\nu t_*\lesssim1$.

## Files

| file | what it is |
|---|---|
| `multi_stream_flow.ipynb` | the full solution — **already executed**, so all figures and printed output render directly in GitHub's file preview (no kernel needed) |
| `multi_stream_flow.py` | the same physics as a standalone script/module: every check and figure, importable functions |
| `build_notebook.py` | utility that produced the executed notebook from `notebook_source.txt` without Jupyter installed |
| `notebook_source.txt` | notebook source in `# %%` percent format (editable, diff-friendly) |

## What is in the notebook

1. Notation and the three dimensionless groups $\theta=v_T/v_0$, $\Gamma=kv_0/\nu$, $k\lambda$
2. The exact collisionless solution, verified against $\partial_\tau F+V\partial_\xi F=0$ and
   against conservation of mass, momentum and energy
3. Lagrangian map, Jacobian $1+\tau\cos\eta$, breaking time $t_b=1/(kv_0)$
4. Closed-form fold geometry — half-width $A(\tau)=\sqrt{\tau^2-1}-\arccos(1/\tau)$ and dip
   depths $(X\mp A)/\tau$ — checked to eight digits against brute force
5. The factor-of-two criterion $\Rightarrow$ $\varepsilon_*=k v_0t_*-1=c\,\theta^{2/3}$ with
   $c=1.160$ (fold centre) or $0.731$ (best point), and the $2/3$ exponent measured numerically
6. Fold thickness $\Delta x\sim v_T/(kv_0)$
7. The collision criterion, the boxed answer, a table of thresholds for air, regime diagram
8. Which relative speed sets $\nu$, and how sharp the criterion is
9. Verification: 4M-particle collisionless Monte Carlo against the analytic $f(V)$, then a
   BGK-collision Monte Carlo showing the contrast fall monotonically from $\sim400$
   (collisionless) to below $2$ at $\Gamma\approx0.3$
10. **Alternative solution 2** — the same answer from two rates / two lengths, with no kinetic
    equation: multi-stream flow occurs when the incipient shock front is thinner than a mean
    free path
11. Caveat: the interior of a *collisional* strong shock is itself weakly bimodal
    (Mott–Smith), evaluated numerically
12. Summary table

## Running the script

```bash
python3 multi_stream_flow.py                 # all checks + 7 figures as PNG
python3 multi_stream_flow.py --no-bgk        # skip the slow Monte-Carlo section
python3 multi_stream_flow.py --show          # display figures instead of saving
python3 multi_stream_flow.py --outdir figs   # choose where PNGs go
```

Requires only `numpy` and `matplotlib`. The full run (including 11 particle simulations of
$10^6$ molecules) takes about two minutes.

## Rebuilding the notebook

```bash
python3 build_notebook.py notebook_source.txt multi_stream_flow.ipynb
```

`build_notebook.py` executes the percent-format cells in one shared namespace, captures stdout
as stream outputs and every open matplotlib figure as a base64 PNG, and writes valid
nbformat 4.5 — so the notebook committed to the repository already contains its outputs.
