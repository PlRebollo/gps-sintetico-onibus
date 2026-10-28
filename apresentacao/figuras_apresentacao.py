"""
figuras_apresentacao.py
=======================
Figuras da apresentacao do modelo atual (25 rotas).
Rodar de qualquer pasta:
    python apresentacao/figuras_apresentacao.py
Saida: figuras/apresentacao/*.png
"""
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PIPE = AQUI.parent / "pipeline"
sys.path.insert(0, str(PIPE))
os.chdir(PIPE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

import config as cfg  # noqa: E402
from core.shapes import load_shape_models  # noqa: E402
from core.trips import build_configs  # noqa: E402
from faixas import particao_ritmo  # noqa: E402
from functional_synthesis import _sem_saltos_a_frente, build_leg_curve, dp_align, srvf  # noqa: E402
from functional_synthesis_dirichlet import build_neighbour_cache  # noqa: E402
from legs import build_route_legs, load_route  # noqa: E402

OUT = AQUI.parent / "figuras" / "apresentacao"
OUT.mkdir(parents=True, exist_ok=True)
DADOS = cfg.ROOT / "dados_sinteticos"
C_REAL, C_SYN = "#2a78d6", "#eb6834"
C_TXT, C_SUB = "#1d1d1d", "#555555"
M = cfg.M_PER_DEG_LAT
TAU = np.linspace(0, 1, 81)

# rotas mostradas nas figuras de resultado (uma de cada tipo/tamanho)
SUB = ["020_circular", "331_circular", "508_circular", "030_ida",
       "216_ida", "506_volta", "603_ida", "707_ida"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "legend.frameon": False, "font.size": 11,
})


def rot(tag):
    l, s = tag.split("_")
    return f"{l} {s}"


def salvar(fig, nome, titulo=None, sub=None):
    if titulo:
        fig.text(0.01, 0.985, titulo, fontsize=15, fontweight="bold", color=C_TXT, va="top")
    if sub:
        fig.text(0.01, 0.935, sub, fontsize=11, color=C_SUB, va="top")
    fig.savefig(OUT / nome, dpi=170, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ", nome)


def legenda_rs(fig, y=0.985, extra=()):
    h = [plt.Line2D([], [], color=C_REAL, lw=2.5), plt.Line2D([], [], color=C_SYN, lw=2.5)]
    l = ["real", "sintético"]
    for hh, ll in extra:
        h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, y), ncol=len(h), fontsize=11)


# ======================================================================
# dados
# ======================================================================
def carregar(tags):
    shape_models = load_shape_models(cfg.SHAPE_PATH)
    configs = build_configs(shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
    R = {}
    for (linha, sentido, shape_id, _s, _e) in configs:
        tag = f"{linha}_{sentido}"
        if tags is not None and tag not in tags:
            continue
        r = build_route_legs(linha, sentido, shape_models, configs)
        shape, _ = load_route(linha, shape_id)
        rota = pd.read_csv(cfg.BASE / f"discovered_route_{linha}_{shape_id}.csv") \
            if (cfg.BASE / f"discovered_route_{linha}_{shape_id}.csv").exists() \
            else pd.read_csv(cfg.BASE / f"discovered_route_{linha}.csv")
        R[tag] = dict(linha=linha, sentido=sentido, shape=shape, legs=r["legs"], db=r["db"],
                      meta=r["meta"], rota=rota)
    return R


def pontos_sinteticos(linhas):
    cols = ["linha", "sentido", "synth_leg_id", "t_s", "lat", "lon", "s_m"]
    return pd.concat([pd.read_csv(DADOS / f"synthetic_fulltrip_dirichlet_{l}.csv.gz", usecols=cols,
                                  dtype={"linha": str, "sentido": str}) for l in linhas],
                     ignore_index=True)


def gamma_sint(R, ms, grupos):
    L = R["meta"]["route_length_m"] if R["meta"]["is_loop"] else None
    out = {}
    for sid in ms["synth_leg_id"]:
        g = grupos[sid]
        s = g["s_m"].to_numpy(float)
        if L:
            d = np.diff(s)
            d = np.where(d < -L / 2, d + L, d)
            s = np.concatenate([[0.0], np.cumsum(d)])
        span = s[-1] - s[0]
        if span <= 0:
            continue
        tau = g["t_s"].to_numpy(float) / max(g["t_s"].iloc[-1], 1.0)
        out[sid] = np.interp(TAU, tau, (s - s[0]) / span)
    return out


def vel_grade(G, T_min, S_km):
    dG = np.diff(G, axis=1)
    v = S_km[:, None] * dG / ((T_min[:, None] / 60) * np.diff(TAU)[None, :])
    rel = 0.5 * (G[:, 1:] + G[:, :-1])
    return v.ravel(), rel.ravel()


# ======================================================================
# 1. visao geral do modelo
# ======================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.6); ax.axis("off")
    cores = {"pre": "#4470a8", "fda": "#118c8b", "sint": "#c47a1e", "fim": "#3e8c58", "cinza": "#6b7a8b"}

    def caixa(x, y, w, h, txt, cor):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=cor, ec="#33475b", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color="white",
                fontsize=11.5, fontweight="bold", linespacing=1.35)

    def seta(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#33475b", lw=2, mutation_scale=18))

    w, h, y1, y2 = 2.15, 1.35, 4.25, 1.35
    xs = [0.15 + i * 2.48 for i in range(6)]
    l1 = [("Amostras de GPS\n(irregulares,\ncom ruído)", "cinza"),
          ("Descoberta\nda rota", "pre"),
          ("Projeção na rota\ns(t) + resíduo\nlateral", "pre"),
          ("Monotonização\n(última obs.)\n+ PCHIP → γ(τ)", "fda"),
          ("Faixas horárias\npor ritmo\n(útil × sáb/dom)", "fda"),
          ("SRVF\nq(τ)", "fda")]
    l2 = [("Validação:\ndistribuições,\nmapa, horários", "cinza"),
          ("Viagem sintética\n1 Hz (CSV)", "fim"),
          ("Ruído lateral\n→ (lat, lon)", "fim"),
          ("SRVF inversa\n+ restrições\nfísicas", "sint"),
          ("Dirichlet\np ~ Dir(α)", "sint"),
          ("k-NN na faixa\n+ alinhamento\n(DP)", "fda")]
    for x, (t, c) in zip(xs, l1):
        caixa(x, y1, w, h, t, cores[c])
    for x, (t, c) in zip(xs, l2):
        caixa(x, y2, w, h, t, cores[c])
    for i in range(5):
        seta(xs[i] + w, y1 + h / 2, xs[i + 1], y1 + h / 2)
        seta(xs[i + 1], y2 + h / 2, xs[i] + w, y2 + h / 2)
    seta(xs[5] + w / 2, y1, xs[5] + w / 2, y2 + h)
    ax.text(xs[0] + 1.5, 6.15, "PRÉ-PROCESSAMENTO", color=cores["pre"], fontweight="bold", fontsize=12)
    ax.text(xs[3], 6.15, "ANÁLISE FUNCIONAL (FDA)", color=cores["fda"], fontweight="bold", fontsize=12)
    ax.text(xs[3] + 0.6, 3.15, "SÍNTESE", color=cores["sint"], fontweight="bold", fontsize=12)
    ax.text(xs[1] + 0.6, 3.15, "SAÍDA", color=cores["fim"], fontweight="bold", fontsize=12)
    salvar(fig, "f01_pipeline.png")


# ======================================================================
# 2. projecao na rota
# ======================================================================
def fig_projecao(R):
    r = R["020_circular"]
    legs = r["legs"]
    lid = legs.groupby("leg_id").size().sort_values().index[len(legs["leg_id"].unique()) // 2]
    g = legs[legs["leg_id"] == lid].sort_values("dt")
    t0 = g["dt"].iloc[0]
    g = g[(g["dt"] - t0).dt.total_seconds().between(30 * 60, 54 * 60)]
    t = (g["dt"] - g["dt"].iloc[0]).dt.total_seconds().to_numpy() / 60
    rota = r["rota"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.35))
    ax = axes[0]
    pad = 0.002
    m = rota["LON"].between(g["lon"].min() - pad, g["lon"].max() + pad) & \
        rota["LAT"].between(g["lat"].min() - pad, g["lat"].max() + pad)
    ax.plot(rota["LON"][m], rota["LAT"][m], ".", ms=1.5, c="k", label="rota descoberta")
    sc = ax.scatter(g["lon"], g["lat"], c=t, cmap="viridis", s=28, zorder=3, ec="k", lw=0.3)
    # setas de residuo em algumas amostras
    lon_r, lat_r = rota["LON"].to_numpy(), rota["LAT"].to_numpy()
    kx = np.cos(np.radians(lat_r.mean()))
    for i in range(0, len(g), max(1, len(g) // 9)):
        lo, la = g["lon"].iloc[i], g["lat"].iloc[i]
        j = np.argmin(((lon_r - lo) * kx) ** 2 + (lat_r - la) ** 2)
        ax.annotate("", xy=(lon_r[j], lat_r[j]), xytext=(lo, la),
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5))
        ax.plot(lon_r[j], lat_r[j], "o", c="#c0392b", ms=6, zorder=4)
    ax.plot([], [], "o", c="#c0392b", label="ponto projetado (posição s)")
    ax.set_aspect(1 / np.cos(np.radians(g["lat"].mean())))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.ticklabel_format(useOffset=False)
    ax.set_title("A) Cada amostra de GPS é projetada\nno ponto mais próximo da rota")
    ax.legend(loc="lower left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04)
    cb.set_label("tempo (min)")
    ax = axes[1]
    ax.scatter(t, g["s_est"] / 1000, c=t, cmap="viridis", s=26, ec="k", lw=0.3)
    ax.plot(t, g["s_est"] / 1000, c="#bbbbbb", lw=0.8, zorder=0)
    ax.set_xlabel("tempo (min)"); ax.set_ylabel("s (km)")
    ax.set_title("B) Resultado: cada amostra vira um par (t, s),\nposição 1D ao longo da rota")
    salvar(fig, "f02_projecao.png")


# ======================================================================
# 3. monotonizacao
# ======================================================================
def _inversoes(s):
    return np.flatnonzero(np.diff(s) < 0) + 1


def fig_monotonia(R):
    legs = R["030_ida"]["legs"]
    melhor = None
    for lid, g in legs.groupby("leg_id"):
        s = g.sort_values("dt")["s_est"].to_numpy()
        inv = np.diff(s)
        k = ((inv < -5) & (inv > -120)).sum()
        if 4 <= k <= 7 and (inv < -120).sum() == 0 and len(s) > 300:
            melhor = g.sort_values("dt"); break
    g = melhor
    t = (g["dt"] - g["dt"].iloc[0]).dt.total_seconds().to_numpy() / 60
    s = g["s_est"].to_numpy() / 1000
    sm = np.maximum.accumulate(s)
    inv = _inversoes(s)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    ax = axes[0]
    ax.plot(t, s, c="#999999", lw=1.2, label="s observado (amostras de GPS)")
    ax.plot(t, sm, c=C_SYN, lw=2, label="última observação (máximo acumulado)")
    ax.plot(t[inv], s[inv], "o", c="#c0392b", ms=6, label=f"recuos detectados (n={len(inv)})")
    ax.set_xlabel("tempo (min)"); ax.set_ylabel("s (km)")
    ax.set_title("A) Viagem real: pequenos recuos de posição\ncausados pelo ruído da projeção")
    ax.legend(loc="upper left")
    ax = axes[1]
    i = inv[len(inv) // 2]
    w = slice(max(0, i - 12), i + 14)
    ax.plot(t[w], s[w], "o-", c="#999999", ms=5, lw=1, label="amostras de GPS")
    ax.step(t[w], sm[w], where="post", c=C_SYN, lw=2.2, label="repete a última posição válida")
    ax.plot(t[w][s[w] < sm[w]], s[w][s[w] < sm[w]], "o", c="#c0392b", ms=8, label="amostra que recuou")
    ax.set_xlabel("tempo (min)"); ax.set_ylabel("s (km)")
    ax.set_title("B) Zoom: quando a posição recua, o ônibus é\nmantido parado na última posição válida")
    ax.legend(loc="upper left")
    fig.tight_layout()
    salvar(fig, "f03_monotonia_ultima_obs.png")

    # salto a frente (050 volta)
    legs = R["050_volta"]["legs"]
    for lid, g in legs.groupby("leg_id"):
        g = g.sort_values("dt")
        sv = g["s_est"].to_numpy()
        salto = np.max(sv - np.minimum.accumulate(sv[::-1])[::-1])
        if 600 < salto < 1200:
            break
    t = (g["dt"] - g["dt"].iloc[0]).dt.total_seconds().to_numpy()
    i = np.argmax(sv - np.minimum.accumulate(sv[::-1])[::-1])
    w = slice(max(0, i - 25), i + 40)
    tt, ss = t[w], sv[w]
    t2, s2 = _sem_saltos_a_frente(tt, ss)
    fora = ~np.isin(tt, t2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    for ax, filtro in zip(axes, [False, True]):
        ax.plot(tt / 60, ss / 1000, "o", ms=4.5, c="#777777", label="amostras de GPS")
        if filtro:
            ax.plot(tt[fora] / 60, ss[fora] / 1000, "x", ms=11, mew=2.5, c="#c0392b",
                    label="amostras descartadas (> 100 m à frente das seguintes)")
            ax.step(t2 / 60, np.maximum.accumulate(s2) / 1000, where="post", c=C_SYN, lw=2.2,
                    label="última observação")
            ax.set_title("B) Descartando a amostra que está > 100 m à\nfrente das 3 seguintes, a curva fica correta")
        else:
            ax.step(tt / 60, np.maximum.accumulate(ss) / 1000, where="post", c="#c0392b", lw=2.2,
                    label="última observação sem o descarte")
            ax.set_title("A) Uma amostra projetada ~800 m à frente: a curva\n\"congela\" nela e depois salta (velocidade falsa)")
        ax.set_xlabel("tempo (min)"); ax.set_ylabel("s (km)")
        ax.legend(loc="upper left")
    fig.tight_layout()
    salvar(fig, "f04_monotonia_salto.png")


# ======================================================================
# 4. selecao de horarios
# ======================================================================
def _faixas_de(meta, fds):
    fx = [(a, b) for (a, b, f) in meta["faixas"].values() if bool(f) == fds]
    return sorted(fx)


def fig_horarios(R):
    tag = "216_ida"
    db = R[tag]["db"]
    st = pd.to_datetime(db["start_dt"])
    h = (st.dt.hour + st.dt.minute / 60).to_numpy()
    fds = (st.dt.dayofweek >= 5).to_numpy()
    T = db["T"].to_numpy() / 60
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    ax = axes[0]
    m = ~fds
    ax.plot(h[m], T[m], ".", ms=2.5, c="#9a9a9a", label="viagem real (dia útil)")
    b = np.floor(h[m] * 4) / 4
    med = pd.Series(T[m]).groupby(b).median()
    ax.plot(med.index + 0.125, med.values, "o", ms=4.5, c=C_SYN, label="mediana a cada 15 min")
    ax.set_title("1) Duração × hora de partida\n(dias úteis): o ritmo muda ao longo do dia")
    ax.set_ylabel("duração da viagem (min)")
    ax.legend(loc="upper right", markerscale=2)
    for ax, tipo, cor, titulo in [(axes[1], False, C_REAL, "2) Faixas encontradas nos dias úteis:\nnova faixa quando o ritmo muda"),
                                  (axes[2], True, "#8e44ad", f"3) Sábado/domingo separado: mais\nrápido e quase sem pico ({len(_faixas_de(R[tag]['meta'], True))} faixas)")]:
        m = fds == tipo
        ax.plot(h[~m], T[~m], ".", ms=2, c="#e3e3e3")
        ax.plot(h[m], T[m], ".", ms=2.5, c="#9a9a9a")
        for k, (a, bb) in enumerate(_faixas_de(R[tag]["meta"], tipo)):
            sel = m & (h >= a) & (h < bb)
            if sel.sum() == 0:
                continue
            ax.axvspan(a, bb, color=cor, alpha=0.06 if k % 2 else 0.14, lw=0)
            ax.hlines(np.median(T[sel]), a, bb, color=cor, lw=3)
        ax.set_title(titulo)
    for ax in axes:
        ax.set_xlim(4, 24); ax.set_xticks(range(4, 25, 4)); ax.set_xlabel("hora de partida")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    salvar(fig, "f05_horarios_ideia.png", f"Seleção de horários na linha {rot(tag)}",
           "Cada faixa sombreada é um grupo de viagens com ritmo parecido; o traço é a mediana da duração da faixa.")


def fig_horarios_metodo(R, todas):
    tag = "216_ida"
    db = R[tag]["db"]
    st = pd.to_datetime(db["start_dt"])
    m = (st.dt.dayofweek < 5).to_numpy()
    h = (st.dt.hour + st.dt.minute / 60).to_numpy()[m]
    y = np.log(db["T"].to_numpy()[m])
    # custo x numero de faixas: particoes otimas com multa crescente
    slots = np.floor(h).astype(int)
    n = len(y)
    sig2 = np.var(y - pd.Series(y).groupby(slots).transform("mean").to_numpy())
    res = []
    for fp in [0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 40, 1e3]:
        fr = particao_ritmo(h, y, min_n=cfg.RITMO_MIN_PERNAS, fator_pen=fp)
        idx = np.searchsorted(np.asarray(fr[1:-1]), h, side="right")
        sse = sum(((y[idx == k] - y[idx == k].mean()) ** 2).sum() for k in np.unique(idx))
        res.append((len(fr) - 1, sse))
    d = pd.DataFrame(res, columns=["nf", "sse"]).groupby("nf")["sse"].min()
    pen = cfg.RITMO_FATOR_PEN * sig2 * np.log(n)
    custo = d + pen * d.index
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw=dict(width_ratios=[1, 1.45]))
    ax = axes[0]
    ax.plot(d.index, d.values, "o-", c=C_REAL, lw=2, label="erro dentro das faixas")
    ax.plot(d.index, pen * d.index, "--", c="#888888", lw=1.6, label="multa: 2·σ²·ln(n) por faixa")
    ax.plot(custo.index, custo.values, "o-", c=C_SYN, lw=2.2, label="custo total")
    k = custo.idxmin()
    ax.plot(k, custo[k], "*", ms=20, c="#c0392b", zorder=5, label=f"escolhido: {k} faixas")
    ax.set_xlabel("número de faixas no dia útil"); ax.set_ylabel("soma dos erros de log(duração)")
    ax.set_title(f"A) {rot(tag)}: mais faixas reduzem o erro,\na multa evita dividir demais")
    ax.legend()
    ax = axes[1]
    tags = [t for t in todas if t in ("020_circular", "022_circular", "331_circular", "508_circular",
                                      "030_ida", "216_ida", "303_volta", "506_ida", "603_ida", "707_volta")]
    y0 = 0
    yt, yl = [], []
    for t in tags:
        meta = todas[t]
        for tipo, cor in [(False, C_REAL), (True, "#8e44ad")]:
            fx = _faixas_de(meta, tipo)
            if t == "331_circular" and tipo:
                fx = []
            for j, (a, b) in enumerate(fx):
                ax.barh(y0, b - a, left=a, height=0.75, color=cor, alpha=0.9 if j % 2 else 0.55,
                        ec="white", lw=0.8)
            yt.append(y0); yl.append(f"{rot(t)} {'sáb/dom' if tipo else 'útil'}")
            y0 -= 1
        y0 -= 0.5
    ax.set_yticks(yt, yl, fontsize=8.5)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("hora de partida")
    ax.grid(axis="y", visible=False)
    ax.set_title("B) Faixas de cada rota (cada bloco = uma faixa;\nsem linha no sáb/dom = rota sem serviço)")
    ax.text(24, yt[5], "", fontsize=8)
    fig.tight_layout()
    salvar(fig, "f06_horarios_metodo.png")


# ======================================================================
# 5. k-NN e alinhamento
# ======================================================================
def fig_knn(R):
    r = R["030_ida"]
    db, legs = r["db"], r["legs"]
    rng = np.random.default_rng(5)
    fx = db["hour_bucket"].value_counts().index[0]
    cand = db[db["hour_bucket"] == fx]
    anc = cand.iloc[rng.integers(len(cand))]
    g = legs[legs["leg_id"] == anc["leg_id"]].sort_values("dt")
    t = (g["dt"] - g["dt"].iloc[0]).dt.total_seconds().to_numpy()
    s = g["s_est"].to_numpy()
    cache = build_neighbour_cache(cand.reset_index(drop=True), anc, cfg.SPAN_RATIO)
    K = cfg.SYNTH_K
    nb = db.set_index("leg_id").loc[cache.ids[:K]]
    tau = anc["tau_grid"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    ax = axes[0]
    ax.plot(t / 60, s / 1000, ".", c="#888888", ms=4, label=f"{len(s)} amostras de GPS")
    ax.set_xlabel("tempo desde o início (min)"); ax.set_ylabel("distância percorrida (km)")
    ax.set_title("A) Entrada: amostras de GPS da viagem âncora\n(com ruído, intervalos irregulares)")
    ax.legend(loc="upper left")
    ax = axes[1]
    T = t[-1] - t[0]
    ax.plot((t - t[0]) / T, (s - s.min()) / (s.max() - s.min()), ".", c="#bbbbbb", ms=4,
            label="amostras normalizadas")
    ax.plot(tau, anc["gamma"], c="k", lw=2, label="γ(τ): última observação + PCHIP")
    ax.set_xlabel("tempo normalizado τ"); ax.set_ylabel("fração do percurso γ")
    ax.set_title("B) Curva contínua γ(τ)")
    ax.legend(loc="upper left")
    ax = axes[2]
    for (_, row), c in zip(nb.iterrows(), ["#e67e22", "#27ae60", "#c0392b", "#8e44ad"]):
        ax.plot(tau, row["q"], lw=1.1, c=c, alpha=0.85)
    ax.plot(tau, anc["q"], c="k", lw=2.2, label="âncora q(τ)")
    ax.plot([], [], c="#e67e22", label=f"{K} vizinhos (sem alinhar)")
    ax.set_xlabel("tempo normalizado τ"); ax.set_ylabel("q(τ)")
    ax.set_title("C) SRVF q(τ) = sign(γ′)·√|γ′|\nâncora × vizinhos, antes do alinhamento")
    ax.legend(loc="lower right")
    fig.tight_layout()
    salvar(fig, "f07_knn_entrada.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9))
    ax = axes[0]
    ax.plot(tau, anc["gamma"], c="k", lw=2.4, label="âncora (referência)")
    for (_, row), c in zip(nb.iterrows(), ["#e67e22", "#27ae60", "#c0392b", "#8e44ad"]):
        _, _, gw = dp_align(anc["q"], row["q"], row["gamma"], tau)
        g2 = np.interp(gw, tau, row["gamma"])
        ax.plot(tau, row["gamma"], "--", c=c, lw=1, alpha=0.6)
        ax.plot(tau, g2, c=c, lw=1.5)
    ax.plot([], [], "--", c="#777777", label="vizinho antes"); ax.plot([], [], c="#777777", label="vizinho alinhado (DP)")
    ax.set_xlabel("tempo normalizado τ"); ax.set_ylabel("γ(τ)")
    ax.set_title("D) Alinhamento elástico (programação dinâmica)")
    ax.legend(loc="upper left")
    ax = axes[1]
    D = np.sort(cache.D)
    ax.bar(np.arange(len(D)), D, color=["#c0392b" if i < K else "#cfcfcf" for i in range(len(D))])
    ax.set_xlabel(f"os {len(D)} candidatos mais próximos (mesma faixa), por distância")
    ax.set_ylabel("distância elástica D")
    ax.set_title(f"E) Distância de cada candidato à âncora\n(vermelho = {K} vizinhos escolhidos)")
    fig.tight_layout()
    salvar(fig, "f08_knn_alinhamento.png")


# ======================================================================
# 6. linhas e volume gerado
# ======================================================================
def fig_linhas(todas_R):
    meta = pd.read_csv(DADOS / "metadados_sinteticos.csv", dtype={"linha": str})
    nsint = meta.groupby(["linha", "sentido"]).size()
    linhas = list(dict.fromkeys(R["linha"] for R in todas_R.values()))
    cmap = plt.get_cmap("tab20")
    cor = {l: cmap(i % 20) for i, l in enumerate(linhas)}
    fig = plt.figure(figsize=(16, 7.6))
    ax = fig.add_axes([0.0, 0.03, 0.47, 0.88])
    lat0 = np.mean([R["rota"]["LAT"].mean() for R in todas_R.values()])
    kx = np.cos(np.radians(lat0))
    for tag, R in todas_R.items():
        ax.plot(R["rota"]["LON"] * kx, R["rota"]["LAT"], c=cor[R["linha"]], lw=2.0, alpha=0.9)
    for l in linhas:
        R = [r for r in todas_R.values() if r["linha"] == l][0]
        ro = R["rota"]
        i = len(ro) // 3
        ax.text(ro["LON"].iloc[i] * kx, ro["LAT"].iloc[i], l, fontsize=10, fontweight="bold",
                color=cor[l], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cor[l], lw=1))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("15 linhas de Curitiba (25 rotas)", loc="center", fontsize=13)
    ax = fig.add_axes([0.6, 0.08, 0.39, 0.83])
    tags = list(todas_R)
    nreal = [len(todas_R[t]["db"]) for t in tags]
    ns = [nsint.get((todas_R[t]["linha"], todas_R[t]["sentido"]), 0) for t in tags]
    y = np.arange(len(tags))[::-1]
    ax.barh(y + 0.2, nreal, 0.4, color=C_REAL, label="viagens reais (março/2026)")
    ax.barh(y - 0.2, ns, 0.4, color=C_SYN, label="viagens sintéticas geradas")
    ax.set_yticks(y, [rot(t) for t in tags], fontsize=9.5)
    for c in ["020", "022", "331", "023", "508"]:
        pass
    ax.set_xlabel("número de viagens")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Viagens reais usadas e sintéticas geradas por rota", loc="left", fontsize=13)
    fig.savefig(OUT / "f09_linhas.png", dpi=170, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("   f09_linhas.png")
    return dict(n_real=int(sum(len(R["db"]) for R in todas_R.values())), n_sint=int(len(meta)))


# ======================================================================
# 7. resultados (modelo completo, rotas de exemplo)
# ======================================================================
def grade(n=8, ncols=4, w=3.9, h=3.0, **kw):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.9), **kw)
    return fig, np.atleast_1d(axes).ravel()


def figs_resultado(R, pts, meta, todas):
    grupos = {k: g for k, g in pts.groupby("synth_leg_id")}
    P = {}
    for tag in SUB:
        r = R[tag]
        ms = meta[(meta["linha"] == r["linha"]) & (meta["sentido"] == r["sentido"])]
        gs = gamma_sint(r, ms, grupos)
        ms = ms.set_index("synth_leg_id").loc[list(gs)]
        d = dict(ms=ms, G_real=np.vstack(r["db"]["gamma"].to_numpy()), G_sint=np.vstack(list(gs.values())),
                 T_real=r["db"]["T"].to_numpy(float) / 60, S_real=r["db"]["S"].to_numpy(float) / 1000,
                 T_sint=ms["duracao_s"].to_numpy(float) / 60, S_sint=ms["extensao_m"].to_numpy(float) / 1000)
        d["vr"], d["rr"] = vel_grade(d["G_real"], d["T_real"], d["S_real"])
        d["vs"], d["rs"] = vel_grade(d["G_sint"], d["T_sint"], d["S_sint"])
        P[tag] = d

    # mapa de viagens sinteticas
    fig, axes = grade(h=3.2)
    for ax, tag in zip(axes, SUB):
        r = R[tag]
        kx = np.cos(np.radians(r["rota"]["LAT"].mean()))
        ax.plot(r["rota"]["LON"] * kx, r["rota"]["LAT"], c="#c8c8c8", lw=4, zorder=0)
        ids = P[tag]["ms"].sample(10, random_state=2).index
        for sid in ids:
            g = grupos[sid]
            ax.plot(g["lon"] * kx, g["lat"], c=C_SYN, lw=0.6, alpha=0.7)
        ax.set_title(rot(tag), fontsize=11)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    salvar(fig, "r01_mapa_sinteticas.png", "Viagens sintéticas sobre a rota",
           "10 viagens sintéticas por rota (laranja) sobre a rota descoberta (cinza), já com o ruído lateral.")

    # curvas gamma
    tags = SUB[:4] + SUB[4:6]
    fig, axes = plt.subplots(2, 6, figsize=(17, 6.6), sharex=True, sharey=True)
    for j, tag in enumerate(tags):
        d = P[tag]
        rng = np.random.default_rng(0)
        for i in rng.choice(len(d["G_real"]), 60, replace=False):
            axes[0, j].plot(TAU, d["G_real"][i], c=C_REAL, lw=0.5, alpha=0.35)
        for i in rng.choice(len(d["G_sint"]), 60, replace=False):
            axes[1, j].plot(TAU, d["G_sint"][i], c=C_SYN, lw=0.5, alpha=0.4)
        axes[0, j].set_title(rot(tag), fontsize=11)
        axes[1, j].set_xlabel("τ = t / T")
    axes[0, 0].set_ylabel("real\nγ(τ): fração do percurso")
    axes[1, 0].set_ylabel("sintético\nγ(τ): fração do percurso")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    salvar(fig, "r02_curvas_gamma.png", "Curvas γ(τ): reais × sintéticas",
           "60 viagens de cada; as sintéticas reproduzem os mesmos trechos de parada (degraus) e de ritmo.")

    # velocidade (histograma)
    fig, axes = grade(sharey=False)
    bins = np.arange(0, 80, 2.5)
    for ax, tag in zip(axes, SUB):
        d = P[tag]
        for v, c in [(d["vr"], C_REAL), (d["vs"], C_SYN)]:
            ax.hist(v, bins=bins, density=True, histtype="step", color=c, lw=2)
            ax.axvline(np.percentile(v, 99), color=c, ls=":", lw=1.4)
        ax.set_title(rot(tag), fontsize=11); ax.set_yticks([]); ax.set_xlabel("velocidade (km/h)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    legenda_rs(fig, extra=[(plt.Line2D([], [], color="#666666", ls=":"), "percentil 99")])
    salvar(fig, "r03_velocidade.png", "Distribuição da velocidade",
           "Velocidade em 80 trechos de tempo iguais de cada viagem. O sintético tem menos paradas e menos picos (média de 4 viagens).")

    # velocidade ao longo do percurso
    fig, axes = grade(sharey=False)
    ed = np.linspace(0, 1, 21)
    for ax, tag in zip(axes, SUB):
        d = P[tag]
        for v, rel, c in [(d["vr"], d["rr"], C_REAL), (d["vs"], d["rs"], C_SYN)]:
            k = np.clip(np.digitize(rel, ed) - 1, 0, 19)
            q = pd.Series(v).groupby(k).quantile([0.25, 0.5, 0.75]).unstack()
            x = 0.5 * (ed[:-1] + ed[1:])[q.index]
            ax.fill_between(x, q[0.25], q[0.75], color=c, alpha=0.15, lw=0)
            ax.plot(x, q[0.5], c=c, lw=2)
        ax.set_title(rot(tag), fontsize=11); ax.set_xlabel("fração do percurso")
    axes[0].set_ylabel("velocidade (km/h)"); axes[4].set_ylabel("velocidade (km/h)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    legenda_rs(fig)
    salvar(fig, "r04_velocidade_percurso.png", "Onde o ônibus anda rápido e onde para",
           "Mediana (linha) e faixa interquartil (sombra) da velocidade em 20 trechos da rota.")

    # duracao por hora, util e fds
    fig, axes = grade(sharey=False)
    for ax, tag in zip(axes, SUB):
        r, d = R[tag], P[tag]
        st = pd.to_datetime(r["db"]["start_dt"])
        real = pd.DataFrame(dict(h=st.dt.hour, f=st.dt.dayofweek >= 5, T=d["T_real"]))
        si = pd.to_datetime(d["ms"]["inicio"])
        sint = pd.DataFrame(dict(h=si.dt.hour.to_numpy(), f=(si.dt.dayofweek >= 5).to_numpy(), T=d["T_sint"]))
        for f, ls in [(False, "-"), (True, "--")]:
            for df, c, mn in [(real, C_REAL, 10), (sint, C_SYN, 5)]:
                g = df[df["f"] == f].groupby("h")["T"].agg(["median", "size"])
                g = g[g["size"] >= mn]
                ax.plot(g.index + 0.5, g["median"], ls, c=c, lw=1.8)
        ax.set_title(rot(tag), fontsize=11); ax.set_xlim(4, 24); ax.set_xticks(range(6, 25, 6))
        ax.set_xlabel("hora de partida")
    axes[0].set_ylabel("duração mediana (min)"); axes[4].set_ylabel("duração mediana (min)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    legenda_rs(fig, extra=[(plt.Line2D([], [], color="#666666", ls="-"), "dia útil"),
                           (plt.Line2D([], [], color="#666666", ls="--"), "sáb/dom")])
    salvar(fig, "r05_duracao_por_hora.png", "Duração ao longo do dia",
           "Mediana da duração por hora de partida: o sintético acompanha os picos e o fim de semana.")

    # QQ
    fig, axes = grade(sharey=False)
    for ax, tag in zip(axes, SUB):
        d = P[tag]
        q = np.linspace(0.01, 0.99, 60)
        a, b = np.quantile(d["T_real"], q), np.quantile(d["T_sint"], q)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], "k:", lw=1)
        ax.plot(a, b, "o", ms=3.5, c=C_SYN)
        ax.set_title(rot(tag), fontsize=11); ax.set_xlabel("quantis reais (min)")
    axes[0].set_ylabel("quantis sintéticos (min)"); axes[4].set_ylabel("quantis sintéticos (min)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    salvar(fig, "r06_qq_duracao.png", "QQ-plot da duração (sintético × real)",
           "Pontos sobre a diagonal = mesma distribuição de duração.")

    # zoom nas amostras de 1 Hz
    tag = "030_ida"
    r = R[tag]
    sid = P[tag]["ms"].index[5] if tag in P else meta[meta["synth_leg_id"].str.startswith(tag)].iloc[5]["synth_leg_id"]
    g = grupos[sid]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1, 1.2]))
    kx = np.cos(np.radians(g["lat"].mean()))
    ax = axes[0]
    ax.plot(r["rota"]["LON"] * kx, r["rota"]["LAT"], c="#c8c8c8", lw=4)
    ax.plot(g["lon"] * kx, g["lat"], c=C_SYN, lw=0.8)
    i0 = len(g) // 2
    z = g.iloc[i0:i0 + 120]
    cx, cy = z["lon"].mean(), z["lat"].mean()
    half = 0.0022
    ax.add_patch(plt.Rectangle(((cx - half / kx) * kx, cy - half), 2 * half, 2 * half, fill=False, ec="k", lw=1.8))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"A) Uma viagem sintética da {rot(tag)}\n({len(g):,} amostras a 1 Hz; quadrado = zoom)".replace(",", "."))
    ax = axes[1]
    lg = r["legs"]
    box = (cx - half / kx, cx + half / kx, cy - half, cy + half)
    z = g[g["lon"].between(box[0], box[1]) & g["lat"].between(box[2], box[3])]
    real = lg[lg["lon"].between(box[0], box[1]) & lg["lat"].between(box[2], box[3])]
    ax.plot(real["lon"].iloc[::3] * kx, real["lat"].iloc[::3], ".", c=C_REAL, ms=3, alpha=0.5,
            label="amostras de GPS reais (todas as viagens)")
    ax.plot(z["lon"] * kx, z["lat"], ".-", c=C_SYN, ms=5, lw=0.8, label="amostras sintéticas (1 por segundo)")
    ax.set_xlim(box[0] * kx, box[1] * kx); ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("B) Zoom (~500 m): amostras sintéticas a cada segundo,\ncom o mesmo espalhamento lateral do GPS real")
    ax.legend(loc="lower left", markerscale=3)
    fig.tight_layout()
    salvar(fig, "r07_amostras_1hz.png")

    # resumo das 25 rotas
    res = pd.read_csv(cfg.BASE / "resultados" / "desempenho_resumo.csv")
    res = res.set_index("rota").loc[list(todas)]
    cols = [("erro_T_pct", "erro da mediana\nda duração (%)", lambda v: abs(v) / 5, "{:+.1f}"),
            ("KS_T", "KS\nduração", lambda v: v / 0.2, "{:.2f}"),
            ("KS_v", "KS\nvelocidade média", lambda v: v / 0.2, "{:.2f}"),
            ("razao_CV_T", "variabilidade da\nduração (÷ real)", lambda v: abs(1 - v) / 0.4, "{:.2f}"),
            ("razao_p99", "p99 velocidade\n(÷ real)", lambda v: abs(1 - v) / 0.4, "{:.2f}")]
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(res), 0); ax.axis("off")
    cm = plt.get_cmap("Oranges")
    for i, (tag, row) in enumerate(res.iterrows()):
        ax.text(-0.05, i + 0.5, rot(tag), ha="right", va="center", fontsize=10.5)
        for j, (c, _, f, fmt) in enumerate(cols):
            v = row[c]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cm(0.08 + 0.75 * min(f(v), 1)), ec="white"))
            ax.text(j + 0.5, i + 0.5, fmt.format(v).replace(".", ","), ha="center", va="center", fontsize=10.5)
    for j, (_, lab, _, _) in enumerate(cols):
        ax.text(j + 0.5, -0.4, lab, ha="center", va="bottom", fontsize=11, color=C_SUB)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.83, bottom=0.01)
    salvar(fig, "r08_resumo_25_rotas.png", "Resumo do desempenho nas 25 rotas",
           "Cor mais forte = mais longe do ideal (erro 0%, KS 0, razões 1). 150 sintéticas por rota.")


def main():
    fig_pipeline()
    print("carregando rotas...")
    R_all = carregar(None)
    fig_projecao(R_all)
    fig_monotonia(R_all)
    fig_horarios(R_all)
    fig_horarios_metodo(R_all, {t: r["meta"] for t, r in R_all.items()})
    fig_knn(R_all)
    info = fig_linhas(R_all)
    print(info)
    meta = pd.read_csv(DADOS / "metadados_sinteticos.csv", dtype={"linha": str}, parse_dates=["inicio"])
    pts = pontos_sinteticos(sorted({R_all[t]["linha"] for t in SUB}))
    figs_resultado(R_all, pts, meta, R_all)


if __name__ == "__main__":
    main()
