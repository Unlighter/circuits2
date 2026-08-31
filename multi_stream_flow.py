"""
Multi-stream flow in a gas with a sinusoidal velocity profile
============================================================

A gas of rigid spheres (mass m, radius r, density n, temperature T) is uniform at t = 0 but
carries a macroscopic velocity v_x(x) = v0 sin(kx), with v0 >> v_T = sqrt(kB T / m).
Question: when does f(v_x) develop at least two local maxima, each at least twice the local
minimum between them ("multi-stream flow")?

ANSWER
------
    v0  >~  nu / k  ~  n sigma v_T / k  ~  (n r^2 / k) sqrt(kB T / m)

equivalently   k v0 >~ nu     (steepening beats collisions)
               Ma * k lambda >~ 1
               lambda >~ v_T/(k v0) = thickness of the incipient fold.
Together with the given v0 >> v_T:   v0 >> v_T * max(1, 1/(k lambda)).

Physics: collisionlessly the profile breaks (folds in phase space) at t_b = 1/(k v0) at
x = pi/k; because v0 >> v_T the fold is resolved into separate peaks at
t_* = t_b [1 + O((v_T/v0)^(2/3))] ~ t_b.  Collisions Maxwellianise f, so multi-stream flow
requires nu t_* <~ 1.

This module reproduces every number and figure of the companion notebook
`multi_stream_flow.ipynb`.

Run:
    python3 multi_stream_flow.py            # all checks, figures saved as PNG
    python3 multi_stream_flow.py --no-bgk   # skip the (slow) Monte-Carlo section
    python3 multi_stream_flow.py --show     # show figures instead of saving
"""

from __future__ import annotations

import argparse
import os

import numpy as np

kB = 1.380649e-23
C2 = np.sqrt(2.0 * np.log(2.0))          # |H|/theta needed for a factor-2 dip
AIR = dict(n=2.5e25, r=1.5e-10, T=300.0, m=4.8e-26)   # air-like, 300 K, 1 atm


# --------------------------------------------------------------------- kinetics
def gas(n: float, r: float, T: float, m: float) -> dict:
    """Hard-sphere kinetic quantities (SI units)."""
    sigma = 4.0 * np.pi * r ** 2                     # pi (2r)^2
    vT = np.sqrt(kB * T / m)
    vbar = np.sqrt(8.0 * kB * T / (np.pi * m))
    lam = 1.0 / (np.sqrt(2.0) * n * sigma)
    return dict(sigma=sigma, vT=vT, vbar=vbar, lam=lam, nu=vbar / lam)


def v0_threshold(k: float, **gaskw) -> float:
    """Minimum v0 for multi-stream flow at wavenumber k: v0 ~ nu/k."""
    return gas(**gaskw)["nu"] / k


# ------------------------------------------------- exact collisionless solution
def H(V, xi, tau):
    """f = n/(sqrt(2 pi) vT) exp(-H^2/2 theta^2) with H = V - sin(xi - V tau)."""
    return V - np.sin(xi - V * tau)


def F(V, xi, tau, theta):
    """Dimensionless distribution F = f v0 / n."""
    return np.exp(-H(V, xi, tau) ** 2 / (2 * theta ** 2)) / (np.sqrt(2 * np.pi) * theta)


def A_fold(tau):
    """Half-width in xi of the three-stream window; tau * (central dip depth in |H|).

    A(tau) = sqrt(tau^2 - 1) - arccos(1/tau)  ~  (2 sqrt2 / 3) (tau-1)^{3/2}.
    """
    tau = np.asarray(tau, dtype=float)
    return np.where(tau > 1.0,
                    np.sqrt(np.maximum(tau ** 2 - 1.0, 0.0))
                    - np.arccos(np.clip(1.0 / tau, -1.0, 1.0)), 0.0)


def dips(X, tau):
    """The two extremal values of H at xi = pi + X:  (X -+ A)/tau."""
    A = float(A_fold(tau))
    return (X + A) / tau, -(A - X) / tau


def tau_star(theta: float, coef: float = 1.0) -> float:
    """First tau at which a factor-2 dip exists (coef=1: at xi=pi, coef=2: best point)."""
    lo, hi = 1.0, 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if coef * float(A_fold(mid)) / mid < C2 * theta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def moments(xig, tau, theta, vmax=6.0, nV=120001, chunk=40):
    """(density, momentum, energy) densities of the exact F, integrated over V."""
    Vg = np.linspace(-vmax, vmax, nV)
    out = np.empty((len(xig), 3))
    for i in range(0, len(xig), chunk):
        sl = slice(i, i + chunk)
        Fm = F(Vg[None, :], np.asarray(xig)[sl, None], tau, theta)
        out[sl, 0] = np.trapezoid(Fm, Vg, axis=1)
        out[sl, 1] = np.trapezoid(Fm * Vg[None, :], Vg, axis=1)
        out[sl, 2] = np.trapezoid(Fm * Vg[None, :] ** 2, Vg, axis=1)
    return out.T


def extrema(V, f):
    """Indices of local maxima and minima of a sampled 1-D array."""
    d = np.diff(f)
    return (np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1,
            np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1)


# -------------------------------------------------------------- particle models
def free_stream(theta=0.05, npart=4_000_000, seed=2024):
    rng = np.random.default_rng(seed)
    xi = rng.random(npart) * 2 * np.pi
    return xi, np.sin(xi) + theta * rng.standard_normal(npart)


def bgk_run(Gamma, theta=0.05, npart=1_000_000, dtau=0.0075, tau_end=1.5,
            ncell=128, seed=1):
    """Free streaming + stochastic BGK collisions.  Units: 1/k, v0, 1/(k v0).

    Gamma = k v0 / nu.  Each step, a particle relaxes with probability
    1 - exp(-dtau/Gamma) to a Maxwellian built from its cell's density/mean/temperature.
    """
    rng = np.random.default_rng(seed)
    xi = rng.random(npart) * 2 * np.pi
    V = np.sin(xi) + theta * rng.standard_normal(npart)
    p = 1.0 - np.exp(-dtau / Gamma) if np.isfinite(Gamma) else 0.0
    for _ in range(int(round(tau_end / dtau))):
        xi = (xi + V * dtau) % (2 * np.pi)
        if p > 0.0:
            cell = np.minimum((xi / (2 * np.pi) * ncell).astype(np.int64), ncell - 1)
            cnt = np.bincount(cell, minlength=ncell).astype(float)
            s1 = np.bincount(cell, weights=V, minlength=ncell)
            s2 = np.bincount(cell, weights=V * V, minlength=ncell)
            cnt[cnt == 0] = 1.0
            mean = s1 / cnt
            var = np.maximum(s2 / cnt - mean ** 2, 1e-14)
            hit = rng.random(npart) < p
            idx = cell[hit]
            V[hit] = mean[idx] + np.sqrt(var[idx]) * rng.standard_normal(int(hit.sum()))
    return xi, V


def probe_points(tau, theta, xc=np.pi):
    """Velocities of the theory stream peaks and of the dips between them."""
    Vg = np.linspace(-1.3, 1.3, 400001)
    imax, imin = extrema(Vg, F(Vg, xc, tau, theta))
    return Vg[imax], Vg[imin]


def contrast(xi, V, Vpk, Vdp, xc=np.pi, half=np.pi / 64, w=0.03):
    """min(peak density)/max(dip density), sampled at theory-given velocities."""
    m = np.abs((xi - xc + np.pi) % (2 * np.pi) - np.pi) < half
    Vc = V[m]

    def dens(qs):
        return np.array([np.count_nonzero(np.abs(Vc - q) < w) / (2 * w) for q in qs])

    pk, dp = dens(Vpk), dens(Vdp)
    return pk.min() / max(dp.max(), 1.0 / (2 * w)), Vc.size, pk, dp


def mott_smith_contrast(Ma, w=0.5, gam=5.0 / 3.0):
    """Bimodality of the two-Maxwellian (Mott-Smith) shock interior; v in units of u1."""
    n2 = (gam + 1) * Ma ** 2 / ((gam - 1) * Ma ** 2 + 2)
    u2 = 1.0 / n2
    T1 = 1.0 / (gam * Ma ** 2)
    T2 = T1 * (2 * gam * Ma ** 2 - (gam - 1)) * ((gam - 1) * Ma ** 2 + 2) \
        / ((gam + 1) ** 2 * Ma ** 2)
    V = np.linspace(-0.6, 2.0, 400001)
    f = ((1 - w) / np.sqrt(2 * np.pi * T1) * np.exp(-(V - 1) ** 2 / (2 * T1))
         + w * n2 / np.sqrt(2 * np.pi * T2) * np.exp(-(V - u2) ** 2 / (2 * T2)))
    imax, imin = extrema(V, f)
    if len(imax) < 2 or len(imin) < 1:
        return n2, u2, np.sqrt(T2), 1.0
    mid = imin[(imin > imax[0]) & (imin < imax[-1])]
    return n2, u2, np.sqrt(T2), min(f[imax[0]], f[imax[-1]]) / f[mid].min()


# --------------------------------------------------------------------- checks
def check_gas():
    g = gas(**AIR)
    print("== reference gas (air-like, 300 K, 1 atm) ==")
    print(f"   sigma = {g['sigma']:.3e} m^2   v_T = {g['vT']:.1f} m/s   "
          f"v_bar = {g['vbar']:.1f} m/s")
    print(f"   lambda = {g['lam'] * 1e9:.1f} nm   nu = {g['nu']:.3e} 1/s   "
          f"nu/(n r^2 v_T) = {g['nu'] / (AIR['n'] * AIR['r'] ** 2 * g['vT']):.1f}")
    return g


def check_solution(theta=0.05):
    print("\n== exact collisionless solution ==")
    h, rng, pts = 1e-5, np.random.default_rng(0), []
    while len(pts) < 6:
        p = (rng.uniform(0, 2 * np.pi), rng.uniform(-1.2, 1.2), rng.uniform(0.2, 1.8))
        if F(p[1], p[0], p[2], theta) > 1e-3:
            pts.append(p)
    res = []
    for xi, V, tau in pts:
        dt = (F(V, xi, tau + h, theta) - F(V, xi, tau - h, theta)) / (2 * h)
        dx = (F(V, xi + h, tau, theta) - F(V, xi - h, tau, theta)) / (2 * h)
        res.append(abs(dt + V * dx) / max(abs(dt), abs(V * dx), 1e-30))
    print(f"   max relative residual of F_tau + V F_xi = {max(res):.2e}")
    xig = np.linspace(0, 2 * np.pi, 721)[:-1]
    for tau in (0.0, 0.7, 1.4):
        d, p, e = moments(xig, tau, theta)
        print(f"   tau={tau:4.1f}: <n>={d.mean():.6f}  <p>={p.mean():+.1e}  "
              f"<E>={e.mean():.6f} (exact {0.5 + theta ** 2:.6f})")


def check_fold():
    print("\n== fold geometry: roots of H and dip depths ==")
    Vg = np.linspace(-1.45, 1.45, 600001)
    for tau in (1.02, 1.10, 1.50, 2.00):
        A = float(A_fold(tau))
        for frac in (0.0, 0.5, 0.99, 1.01):
            X = frac * A
            hh = H(Vg, np.pi + X, tau)
            ns = int((np.diff(np.sign(hh)) != 0).sum())
            d = np.diff(hh)
            ext = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
            num = np.sort(np.abs(hh[ext]))[:2]
            pred = np.sort(np.abs(dips(X, tau)))
            print(f"   tau={tau:4.2f} X/A={frac:4.2f} streams={ns} "
                  f"numeric={np.round(num, 7)} formula={np.round(pred, 7)}")
        print(f"      A={A:.6f}  (2sqrt2/3)eps^1.5={2 * np.sqrt(2) / 3 * (tau - 1) ** 1.5:.6f}")


def check_threshold():
    p1 = (1.5 * np.sqrt(np.log(2))) ** (2 / 3)
    p2 = (0.75 * np.sqrt(np.log(2))) ** (2 / 3)
    print(f"\n== detection time: eps_* = c theta^(2/3), c = {p1:.4f} (centre) / "
          f"{p2:.4f} (fold edge) ==")
    print(f"   {'theta':>8} {'eps* centre':>12} {'asympt':>9} {'eps* edge':>10} "
          f"{'asympt':>9} {'2A/theta':>9}")
    for th in (0.2, 0.1, 0.05, 0.02, 0.01, 1e-3, 1e-4):
        ts = tau_star(th)
        print(f"   {th:8.4f} {ts - 1:12.5f} {p1 * th ** (2 / 3):9.5f} "
              f"{tau_star(th, 2) - 1:10.5f} {p2 * th ** (2 / 3):9.5f} "
              f"{2 * float(A_fold(ts)) / th:9.3f}")
    print("   -> t_* = t_b [1 + O(theta^(2/3))] and fold width Dx ~ v_T/(k v0)")


def check_answer():
    print("\n== the answer, in numbers (air-like gas) ==")
    g = gas(**AIR)
    print(f"   {'wavelength':>12} {'k lambda':>10} {'v0 > nu/k':>14} {'Ma req.':>9}")
    for L in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7):
        k = 2 * np.pi / L
        v0m = v0_threshold(k, **AIR)
        print(f"   {L * 1e3:9.4f} mm {k * g['lam']:10.4f} {v0m:11.3e} m/s "
              f"{v0m / g['vT']:9.1f}")
    k, v0 = 2 * np.pi / 1e-6, 1.0e4
    delta = g["vT"] / (k * v0)
    print(f"   alternative-2 cross-check at L=1um, v0=1e4 m/s: Gamma={k * v0 / g['nu']:.3f}, "
          f"delta={delta * 1e9:.3f} nm, lambda/delta={g['lam'] / delta:.3f} "
          f"(= Gamma * v_bar/v_T)")


def check_mc(theta=0.05):
    print("\n== Monte Carlo: collisionless histogram vs analytic ==")
    xi0, V0 = free_stream(theta)
    for tau in (1.0, 1.2, 1.5):
        xi = (xi0 + V0 * tau) % (2 * np.pi)
        m = np.abs((xi - np.pi + np.pi) % (2 * np.pi) - np.pi) < 0.01
        h, e = np.histogram(V0[m], bins=140, range=(-1.25, 1.25), density=True)
        c = 0.5 * (e[1:] + e[:-1])
        Vg = np.linspace(-1.25, 1.25, 6001)
        Fa = F(Vg, np.pi, tau, theta)
        Fa /= np.trapezoid(Fa, Vg)
        ref = np.interp(c, Vg, Fa)
        good = ref > 0.05 * ref.max()
        print(f"   tau={tau}: {int(m.sum()):7d} particles, max rel. deviation "
              f"{(np.abs(h[good] - ref[good]) / ref[good]).max():.3f}")


def check_bgk(theta=0.05):
    print("\n== Monte Carlo with BGK collisions: contrast vs Gamma = k v0 / nu ==")
    Vpk, Vdp = probe_points(1.5, theta)
    print(f"   theory peaks {np.round(Vpk, 4)}  dips {np.round(Vdp, 4)}")
    out = {}
    for G in (np.inf, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03):
        c = contrast(*bgk_run(G, theta=theta), Vpk, Vdp)[0]
        out[G] = c
        print(f"   Gamma={G:>6.3g}  nu t_end={1.5 / G:7.3f}  contrast={c:8.2f}  "
              f"{'MULTI-STREAM' if c >= 2 else 'single peak'}")
    print("   -> threshold at Gamma_* ~ 0.3, i.e. O(1): k v0 >~ nu")
    return out


def check_mott_smith():
    print("\n== caveat: bimodality inside a collisional strong shock ==")
    for Ma in (2, 3, 5, 10, 30, 100):
        n2, u2, vt2, c = mott_smith_contrast(Ma)
        print(f"   Ma={Ma:5g}: n2/n1={n2:.3f} u2/u1={u2:.3f} vT2/u1={vt2:.3f} "
              f"contrast={c:.2f}")


# --------------------------------------------------------------------- figures
def figures(outdir, show, theta=0.05, do_bgk=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                                "grid.alpha": 0.25, "figure.autolayout": True})

    def finish(name):
        if show:
            plt.show()
        else:
            path = os.path.join(outdir, name)
            plt.savefig(path)
            plt.close()
            print(f"   wrote {path}")

    # 1: steepening, Jacobian, density
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.5))
    eta = np.linspace(-np.pi, 3 * np.pi, 4001)
    xig = np.linspace(0, 2 * np.pi, 1201)
    for tau, c in zip([0.0, 0.5, 1.0, 1.5], ["#4477aa", "#66ccee", "#ee6677", "#aa3377"]):
        ax[0].plot(eta + tau * np.sin(eta), np.sin(eta), color=c, lw=1.8,
                   label=rf"$\tau={tau}$")
        ax[1].plot(eta, 1 + tau * np.cos(eta), color=c, lw=1.8, label=rf"$\tau={tau}$")
        ax[2].plot(xig, moments(xig, tau, theta, vmax=4.0, nV=80001)[0], color=c, lw=1.8,
                   label=rf"$\tau={tau}$")
    ax[0].set(xlim=(0, 2 * np.pi), ylim=(-1.15, 1.15), xlabel=r"$\xi=kx$",
              ylabel=r"$V=v_x/v_0$", title="velocity profile folds")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlim=(0, 2 * np.pi), xlabel=r"$\eta=kx_0$",
              ylabel=r"$\partial\xi/\partial\eta$", title=r"Jacobian zero at $\eta=\pi,\tau=1$")
    ax[2].set(xlim=(0, 2 * np.pi), yscale="log", xlabel=r"$\xi=kx$", ylabel=r"$\rho/n$",
              title="density (caustics)")
    for a in ax:
        a.legend(fontsize=8)
    finish("fig1_breaking.png")

    # 2: F(V) and contrast
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    Vg = np.linspace(-1.25, 1.25, 200001)
    for tau, c in zip([0.90, 1.00, 1.10, 1.20, 1.50],
                      ["#bbbbbb", "#4477aa", "#228833", "#ccbb44", "#ee6677"]):
        ax[0].plot(Vg, F(Vg, np.pi, tau, theta), color=c, lw=1.6, label=rf"$\tau={tau:.2f}$")
    ax[0].set(xlabel="$V$", ylabel="$F$", yscale="log", ylim=(1e-3, 20),
              title=rf"$F(V)$ at $\xi=\pi$, $\theta={theta}$")
    taus = np.linspace(1.0, 1.8, 400)
    for th, c in zip([0.02, 0.05, 0.10, 0.20], ["#4477aa", "#228833", "#ccbb44", "#ee6677"]):
        ax[1].plot(taus, np.exp((A_fold(taus) / taus) ** 2 / (2 * th ** 2)), color=c,
                   lw=1.7, label=rf"$\theta={th}$")
    ax[1].axhline(2, color="k", ls="--", lw=1.2)
    ax[1].set(xlabel=r"$\tau$", ylabel=r"$F_{max}/F_{min}$", yscale="log", ylim=(1, 1e6),
              title="peak-to-dip contrast")
    for a in ax:
        a.legend(fontsize=8)
    finish("fig2_distribution.png")

    # 3: phase space and H
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.0))
    tau = 1.5
    xig = np.linspace(np.pi - 1.0, np.pi + 1.0, 500)
    Vg = np.linspace(-1.2, 1.2, 500)
    Fm = F(Vg[None, :], xig[:, None], tau, theta)
    im = ax[0].imshow(np.log10(np.maximum(Fm.T, 1e-6)), origin="lower", aspect="auto",
                      extent=(xig[0], xig[-1], Vg[0], Vg[-1]), cmap="magma", vmin=-4,
                      vmax=np.log10(Fm.max()))
    A = float(A_fold(tau))
    for s in (-1, 1):
        ax[0].axvline(np.pi + s * A, color="#66ccee", ls="--", lw=1.2)
    ax[0].set(xlabel=r"$\xi=kx$", ylabel="$V$", title=rf"$\log_{{10}}F$, $\tau={tau}$")
    plt.colorbar(im, ax=ax[0])
    Vg = np.linspace(-1.3, 1.3, 4001)
    for frac, c in zip([0.0, 0.5, 0.95], ["#4477aa", "#228833", "#ee6677"]):
        X = frac * A
        ax[1].plot(Vg, H(Vg, np.pi + X, tau), color=c, lw=1.7, label=rf"$X={frac:.2f}A$")
        for lev in dips(X, tau):
            ax[1].axhline(lev, color=c, ls=":", lw=1.0)
    ax[1].axhline(0, color="k", lw=0.9)
    ax[1].set(xlabel="$V$", ylabel="$H(V)$", ylim=(-0.9, 0.9),
              title=r"zeros = streams, extrema $(X\mp A)/\tau$ = dips")
    ax[1].legend(fontsize=8)
    finish("fig3_phase_space.png")

    # 4: eps_* scaling
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ths = np.logspace(-4, np.log10(0.3), 60)
    e1 = np.array([tau_star(t) - 1 for t in ths])
    e2 = np.array([tau_star(t, 2.0) - 1 for t in ths])
    ax.loglog(ths, e1, color="#4477aa", lw=2, label=r"exact, $\xi=\pi$")
    ax.loglog(ths, (1.5 * np.sqrt(np.log(2))) ** (2 / 3) * ths ** (2 / 3), "--",
              color="#4477aa", lw=1.2, label=r"$1.160\,\theta^{2/3}$")
    ax.loglog(ths, e2, color="#ee6677", lw=2, label="exact, best point")
    ax.loglog(ths, (0.75 * np.sqrt(np.log(2))) ** (2 / 3) * ths ** (2 / 3), "--",
              color="#ee6677", lw=1.2, label=r"$0.731\,\theta^{2/3}$")
    ax.set(xlabel=r"$\theta=v_T/v_0$", ylabel=r"$\varepsilon_*=k v_0 t_*-1$",
           title=r"delay after breaking $\propto\theta^{2/3}$")
    ax.legend(fontsize=8)
    finish("fig4_threshold.png")

    # 5: regime diagram
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    kl = np.logspace(-4, 2, 400)
    ratio = np.sqrt(8 / np.pi)
    ax.loglog(kl, ratio / kl, color="#ee6677", lw=2, label=r"$\Gamma=k v_0/\nu=1$")
    ax.axhline(1, color="#4477aa", lw=2, label=r"$v_0=v_T$")
    ax.fill_between(kl, np.maximum(ratio / kl, 1.0), 1e6, color="#88ccaa", alpha=0.35,
                    label="multi-stream possible")
    g = gas(**AIR)
    for L, mk in [(1e-3, "o"), (1e-5, "s"), (1e-6, "^"), (1e-7, "D")]:
        k = 2 * np.pi / L
        ax.plot(k * g["lam"], max(g["nu"] / k / g["vT"], 1.0), mk, color="k", ms=6)
    ax.set(xlabel=r"$k\lambda$", ylabel=r"$v_0/v_T$", xlim=(1e-4, 1e2), ylim=(0.3, 1e6),
           title="regime diagram (markers: air thresholds)")
    ax.legend(fontsize=8)
    finish("fig5_regime.png")

    # 6: MC vs analytic
    xi0, V0 = free_stream(theta)
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=True)
    for a, tau in zip(ax, [1.0, 1.2, 1.5]):
        xi = (xi0 + V0 * tau) % (2 * np.pi)
        m = np.abs((xi - np.pi + np.pi) % (2 * np.pi) - np.pi) < 0.01
        h, e = np.histogram(V0[m], bins=140, range=(-1.25, 1.25), density=True)
        Vg = np.linspace(-1.25, 1.25, 3001)
        Fa = F(Vg, np.pi, tau, theta)
        Fa /= np.trapezoid(Fa, Vg)
        a.plot(0.5 * (e[1:] + e[:-1]), h, ".", ms=3, color="#ee6677", label="MC")
        a.plot(Vg, Fa, color="#4477aa", lw=1.5, label="analytic")
        a.set(xlabel="$V$", yscale="log", ylim=(2e-3, 30), title=rf"$\tau={tau}$")
        a.legend(fontsize=8)
    ax[0].set_ylabel("$f(V)$")
    finish("fig6_montecarlo.png")

    if not do_bgk:
        return
    # 7: BGK
    Vpk, Vdp = probe_points(1.5, theta)
    gammas = [np.inf, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03]
    hist, cs = {}, []
    for G in gammas:
        xi, V = bgk_run(G, theta=theta)
        c = contrast(xi, V, Vpk, Vdp)[0]
        m = np.abs((xi - np.pi + np.pi) % (2 * np.pi) - np.pi) < np.pi / 64
        h, e = np.histogram(V[m], bins=90, range=(-1.3, 1.3))
        hist[G] = (0.5 * (e[1:] + e[:-1]), h.astype(float), c)
        cs.append(c)
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.1))
    cols = ["#000000", "#332288", "#4477aa", "#117733", "#999933", "#ddaa33",
            "#ee6677", "#aa3377"]
    for (G, (ctr, h, c)), col in zip(hist.items(), cols):
        lab = "collisionless" if not np.isfinite(G) else rf"$\Gamma={G:g}$"
        ax[0].plot(ctr, h / h.max(), lw=1.5, color=col, label=lab + f" ({c:.1f})")
    ax[0].set(xlabel="$V$", ylabel="counts (norm.)", yscale="log", ylim=(2e-5, 3),
              title=r"$f(V)$ in the fold, $\tau=1.5$")
    ax[0].legend(fontsize=7.5, ncol=2, loc="lower center")
    fin = np.array([G for G in gammas if np.isfinite(G)])
    ax[1].loglog(fin, [hist[G][2] for G in fin], "o-", color="#4477aa", lw=1.8, ms=6)
    ax[1].axhline(2, color="k", ls="--", lw=1.2)
    ax[1].axvline(1, color="#ee6677", ls=":", lw=1.5)
    ax[1].set(xlabel=r"$\Gamma=k v_0/\nu$", ylabel="contrast",
              title=r"crossover at $\Gamma\approx0.3$")
    finish("fig7_bgk.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("ANSWER")[0])
    ap.add_argument("--no-bgk", action="store_true", help="skip the slow BGK section")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--show", action="store_true", help="show figures instead of saving")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    print(__doc__.split("This module")[0].strip())
    print("\n" + "=" * 78)
    check_gas()
    check_solution()
    check_fold()
    check_threshold()
    check_answer()
    check_mc()
    if not a.no_bgk:
        check_bgk()
    check_mott_smith()
    if not a.no_figures:
        print("\n== figures ==")
        figures(a.outdir, a.show, do_bgk=not a.no_bgk)
    g = gas(**AIR)
    print("\n" + "=" * 78)
    print("ANSWER:  v0 >~ nu/k ~ (n r^2/k) sqrt(kB T/m)   <=>   k v0 >~ nu   <=>   "
          "Ma * k lambda >~ 1")
    print(f"         (hard spheres: nu = sqrt2 n sigma v_bar = "
          f"{g['nu'] / (AIR['n'] * AIR['r'] ** 2 * g['vT']):.0f} n r^2 v_T)")
    print("         with v0 >> v_T assumed:  v0 >> v_T max(1, 1/(k lambda))")


if __name__ == "__main__":
    main()
