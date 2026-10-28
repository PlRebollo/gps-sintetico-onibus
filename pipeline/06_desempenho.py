"""
06_desempenho.py
================
PASSO 6: graficos de desempenho do gerador (real x sintetico), no formato usual
de avaliacao de dados sinteticos: fidelidade (medianas, KS, ECDF, QQ),
variabilidade, estrutura temporal (perfil de velocidade, faixa horaria),
indistinguibilidade (classificador real x sintetico, PCA das curvas) e
nao copia (distancia ao registro real mais proximo, DCR).

Reaproveita a carga de 05_analise.py. Saidas:
  figuras/06_desempenho/*.png
  resultados/desempenho_resumo.csv     uma linha por rota com as metricas do painel

Tamanhos e titulos se ajustam ao numero de rotas (25) e a
duracao e mostrada por hora de partida e tipo de dia (faixas variam por rota).

Uso:
    cd pipeline && python 06_desempenho.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
import locale
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

import config as cfg  # noqa: E402

analise = importlib.import_module("05_analise")
warnings.filterwarnings("ignore")

FIG = cfg.ROOT / "figuras" / "06_desempenho"
RES = cfg.BASE / "resultados"

# real = azul, sintetico = laranja (mesma convencao da apresentacao)
C_REAL, C_SYN = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e4df", "#fcfcfb"
TAU = np.linspace(0, 1, 81)

try:
    locale.setlocale(locale.LC_NUMERIC, "pt_BR.UTF-8")
except locale.Error:
    pass
plt.rcParams.update({
    "axes.formatter.use_locale": True,
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelcolor": INK2,
    "axes.edgecolor": "#b9b7b0", "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "lines.linewidth": 2,
})


def br(v, fmt="{:.2f}"):
    """Numero formatado com virgula decimal."""
    return fmt.format(v).replace(".", ",")


def rotulo(tag):
    linha, sentido = tag.split("_")
    return f"{linha} {sentido}"


def savefig(fig, name, titulo, subtitulo):
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.0)
    fig.text(0.01, 0.955, subtitulo, ha="left", va="top", fontsize=10.5, color=INK2,
             transform=fig.transFigure)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: 06_desempenho/{name}")


def legenda_real_sint(fig, extra=None):
    h = [plt.Line2D([], [], c=C_REAL, lw=2.5, label="real"),
         plt.Line2D([], [], c=C_SYN, lw=2.5, label="sintético")]
    fig.legend(handles=h + (extra or []), loc="upper right", ncol=len(h) + len(extra or []),
               bbox_to_anchor=(0.995, 1.005), fontsize=10.5)


def grade(n, ncols=5, w=3.4, h=2.9, **kw):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.6), **kw)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


# ============================================================
# PREPARACAO: curvas e velocidades por viagem
# ============================================================
def gamma_sinteticas(R, ms, grupos):
    """gamma(tau) de cada viagem sintetica na grade de 81 pontos (s desenrolado nos lacos)."""
    L = R["meta"]["route_length_m"] if R["meta"]["is_loop"] else None
    out = {}
    for _, m in ms.iterrows():
        g = grupos[m["synth_leg_id"]]
        s = g["s_m"].to_numpy(float)
        if L:
            d = np.diff(s)
            d = np.where(d < -L / 2, d + L, d)
            s = np.concatenate([[0.0], np.cumsum(d)])
        span = s[-1] - s[0]
        if span <= 0:
            continue
        tau = g["t_s"].to_numpy(float) / max(g["t_s"].iloc[-1], 1.0)
        out[m["synth_leg_id"]] = np.interp(TAU, tau, (s - s[0]) / span)
    return out


def vel_grade(G, T_min, S_km):
    """Velocidade (km/h) em cada um dos 80 trechos de tempo iguais da curva gamma,
    e a fracao do percurso no meio do trecho. Mesma conta para real e sintetico."""
    dG = np.diff(G, axis=1)
    v = S_km[:, None] * dG / ((T_min[:, None] / 60) * np.diff(TAU)[None, :])
    rel = 0.5 * (G[:, 1:] + G[:, :-1])
    return v.ravel(), rel.ravel()


def preparar(routes, meta_syn, pts):
    grupos = {k: g for k, g in pts.groupby("synth_leg_id", observed=True)}
    P = {}
    for tag, R in routes.items():
        ms = meta_syn[(meta_syn["linha"] == R["linha"]) & (meta_syn["sentido"] == R["sentido"])]
        gs = gamma_sinteticas(R, ms, grupos)
        P[tag] = dict(
            R=R, ms=ms.set_index("synth_leg_id").loc[list(gs)],
            G_real=np.vstack(R["db"]["gamma"].to_numpy()), G_sint=np.vstack(list(gs.values())),
            T_real=R["db"]["T"].to_numpy(float) / 60, S_real=R["db"]["S"].to_numpy(float) / 1000,
        )
        d = P[tag]
        d["T_sint"] = d["ms"]["duracao_s"].to_numpy(float) / 60
        d["S_sint"] = d["ms"]["extensao_m"].to_numpy(float) / 1000
        d["v_real"] = 60 * d["S_real"] / d["T_real"]
        d["v_sint"] = 60 * d["S_sint"] / d["T_sint"]
        d["v_real_ping"], d["rel_real"] = vel_grade(d["G_real"], d["T_real"], d["S_real"])
        d["v_sint_ping"], d["rel_sint"] = vel_grade(d["G_sint"], d["T_sint"], d["S_sint"])
        print(f"  {tag}: {len(d['T_real'])} reais, {len(d['T_sint'])} sinteticas")
    return P


# ============================================================
# METRICAS EXTRAS: discriminador e DCR
# ============================================================
def feats(G, T, S):
    idx = np.linspace(0, 80, 21).astype(int)[1:-1]
    return np.column_stack([G[:, idx] - TAU[idx], T, S])


def auc_discriminador(d, rng, repeticoes=10):
    """AUC (validacao cruzada 5x) de uma floresta aleatoria separando real de sintetico.
    0,5 = indistinguivel. Controle: duas amostras reais disjuntas (real x real)."""
    Xr = feats(d["G_real"], d["T_real"], d["S_real"])
    Xs = feats(d["G_sint"], d["T_sint"], d["S_sint"])
    n = len(Xs)
    out = {"forma": [], "forma_T_S": [], "controle": []}
    for _ in range(repeticoes):
        perm = rng.permutation(len(Xr))
        a, b = Xr[perm[:n]], Xr[perm[n:2 * n]]
        y = np.r_[np.zeros(n), np.ones(n)]
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                     random_state=int(rng.integers(1e9)))
        out["forma_T_S"].append(cross_val_score(clf, np.vstack([a, Xs]), y, cv=5, scoring="roc_auc").mean())
        out["forma"].append(cross_val_score(clf, np.vstack([a, Xs])[:, :-2], y, cv=5, scoring="roc_auc").mean())
        if len(b) == n:
            out["controle"].append(cross_val_score(clf, np.vstack([a, b]), y, cv=5, scoring="roc_auc").mean())
    return {k: (np.mean(v), np.std(v)) if v else (np.nan, np.nan) for k, v in out.items()}


def dcr(d, rng, n=150):
    """Distancia (L2 na curva gamma) ao registro real mais proximo: sintetico -> real e,
    como controle, real (fora do banco) -> restante do banco real."""
    G = d["G_real"]
    hold = rng.choice(len(G), size=min(n, len(G) // 2), replace=False)
    resto = np.setdiff1d(np.arange(len(G)), hold)
    ctrl = np.array([np.sqrt(((G[resto] - G[i]) ** 2).sum(1)).min() for i in hold])
    Gs = d["G_sint"][rng.choice(len(d["G_sint"]), size=min(n, len(d["G_sint"])), replace=False)]
    sint = np.array([np.sqrt(((G[resto] - g) ** 2).sum(1)).min() for g in Gs])
    return ctrl, sint


# ============================================================
# FIGURAS
# ============================================================
def fig_painel(P, resumo):
    """1) erro relativo das medianas e 2) KS, por rota e variavel."""
    tags = list(P)
    y = np.arange(len(tags))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.34 * len(tags) + 2.2),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    a1.axvspan(-5, 5, color="#ecebe6", zorder=0)
    a1.axvline(0, c=INK2, lw=1)
    marcas = [("erro_T_pct", "duração", "o"), ("erro_S_pct", "extensão", "s"), ("erro_v_pct", "velocidade", "D")]
    cores = [C_REAL, C_SYN, "#1baf7a"]
    for k, ((col, lab, mk), c) in enumerate(zip(marcas, cores)):
        a1.scatter(resumo[col], y + (k - 1) * 0.22, marker=mk, s=36, c=c, label=lab,
                   edgecolor=SURF, linewidth=1.5, zorder=3)
    a1.set_yticks(y, [rotulo(t) for t in tags])
    a1.set_xlabel("diferença da mediana sintética em relação à real (%)")
    pior = np.nanmax(np.abs(resumo[["erro_T_pct", "erro_S_pct", "erro_v_pct"]].to_numpy()))
    dentro = int((resumo[["erro_T_pct", "erro_S_pct", "erro_v_pct"]].abs() <= 5).all(axis=1).sum())
    a1.set_title(f"Medianas: {dentro} de {len(tags)} rotas dentro de ±5% (pior {br(pior, '{:.1f}')}%)")
    a1.legend(loc="lower left", fontsize=9.5)
    a1.text(5.2, y[0] + 0.45, "faixa ±5%", fontsize=9, color=INK2)
    lim = max(6.5, np.nanmax(np.abs(resumo[[m[0] for m in marcas]].to_numpy())) + 1)
    a1.set_xlim(-lim, lim)

    ks = resumo[["KS_T", "KS_S", "KS_v"]].to_numpy()
    im = a2.imshow(ks, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    for i in range(ks.shape[0]):
        for j in range(ks.shape[1]):
            a2.text(j, i, br(ks[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if ks[i, j] > 0.45 else INK)
    a2.set_xticks(range(3), ["duração", "extensão", "velocidade"])
    a2.set_yticks(range(len(tags)), [rotulo(t) for t in tags])
    a2.grid(False)
    a2.set_title("Distância KS entre as distribuições (0 = iguais)")
    for s in a2.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=a2, fraction=0.04, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, "01_painel_fidelidade.png", "Fidelidade por rota",
            "Esquerda: quanto a mediana sintética se afasta da real. Direita: estatística de Kolmogorov–Smirnov. "
            "KS alto na extensão vem de distribuições muito concentradas (diferenças de poucos metros).")


def fig_ecdf(P, var, nome, xlabel, titulo):
    fig, axes = grade(len(P))
    for ax, (tag, d) in zip(axes, P.items()):
        for x, c in [(d[f"{var}_real"], C_REAL), (d[f"{var}_sint"], C_SYN)]:
            xs = np.sort(x)
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", c=c)
        ks = ks_2samp(d[f"{var}_real"], d[f"{var}_sint"]).statistic
        ax.set_title(rotulo(tag))
        ax.text(0.97, 0.06, f"KS = {br(ks)}", transform=ax.transAxes, ha="right", fontsize=10, color=INK)
        lo, hi = np.percentile(np.r_[d[f"{var}_real"], d[f"{var}_sint"]], [0.5, 99.5])
        ax.set_xlim(lo, hi)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("fração acumulada")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    legenda_real_sint(fig)
    savefig(fig, nome, titulo, "Função de distribuição acumulada empírica (ECDF), uma viagem por ponto. "
            "Curvas sobrepostas = distribuições iguais.")


def fig_qq(P):
    q = np.linspace(0.02, 0.98, 49)
    lim = [0.6, 1.5]
    fig, axes = grade(len(P))
    for ax, (tag, d) in zip(axes, P.items()):
        ax.plot(lim, lim, c=INK2, lw=1, ls="--")
        ax.fill_between(lim, [l * 0.95 for l in lim], [l * 1.05 for l in lim], color="#ecebe6", zorder=0)
        m = np.median(d["T_real"])
        ax.plot(np.quantile(d["T_real"], q) / m, np.quantile(d["T_sint"], q) / m, "o-", ms=3, lw=1.2,
                c=C_SYN)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_title(rotulo(tag))
    axes[0].set_ylabel("sintético ÷ mediana real")
    for ax in axes[-5:]:
        ax.set_xlabel("real ÷ mediana real")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    savefig(fig, "04_qq_duracao_normalizado.png", "QQ da duração por rota",
            "Quantis 2% a 98% divididos pela mediana real. Faixa cinza = ±5% da diagonal. "
            "Curva mais plana que a diagonal = sintético varia menos.")


def fig_variabilidade(resumo):
    tags = resumo["rota"].tolist()
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(max(11, 0.62 * len(tags)), 4.8))
    ax.axhline(1, c=INK2, lw=1, ls="--")
    ax.bar(x - 0.2, resumo["razao_CV_T"], 0.38, color="#6250d6", label="duração (CV sint. ÷ CV real)")
    ax.bar(x + 0.2, resumo["razao_IQR_v"], 0.38, color="#1baf7a", label="velocidade média (IQR sint. ÷ IQR real)")
    for i, (a, b) in enumerate(zip(resumo["razao_CV_T"], resumo["razao_IQR_v"])):
        ax.text(i - 0.2, a + 0.02, br(a, "{:.1f}"), ha="center", fontsize=7.5, color=INK)
        ax.text(i + 0.2, b + 0.02, br(b, "{:.1f}"), ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [rotulo(t) for t in tags], rotation=45, ha="right")
    ax.set_ylabel("razão sintético ÷ real")
    ax.set_ylim(0, max(1.25, resumo[["razao_CV_T", "razao_IQR_v"]].to_numpy().max() + 0.15))
    ax.legend(loc="upper center", fontsize=9.5, ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    n_menor = int((resumo["razao_CV_T"] < 1).sum())
    savefig(fig, "05_variabilidade.png",
            f"A variação da duração fica abaixo da real em {n_menor} de {len(tags)} rotas",
            "Razão entre a dispersão sintética e a real (linha tracejada = 1, mesma variação). "
            "Abaixo de 1: a média das vizinhas encolhe a variação entre viagens.")


def fig_vel_pontual(P, resumo):
    fig, axes = grade(len(P), sharey=False)
    bins = np.arange(0, 80, 2.5)
    for ax, (tag, d) in zip(axes, P.items()):
        for v, c in [(d["v_real_ping"], C_REAL), (d["v_sint_ping"], C_SYN)]:
            h, e = np.histogram(np.clip(v, 0, 79.9), bins=bins, density=True)
            ax.step(e[:-1], h, where="post", c=c)
            ax.axvline(np.percentile(v, 99), c=c, lw=1.2, ls=":")
        ax.set_title(rotulo(tag))
        ax.set_xlabel("velocidade (km/h)")
        ax.set_yticks([])
        r = resumo.set_index("rota").loc[tag]
        ax.text(0.97, 0.9, f"p99 real {r['v_p99_real']:.0f} km/h\np99 sint. {r['v_p99_sint']:.0f} km/h",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK)
    axes[0].set_ylabel("densidade")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    legenda_real_sint(fig, [plt.Line2D([], [], c=INK2, lw=1.2, ls=":", label="percentil 99")])
    savefig(fig, "06_velocidade_instantanea.png", "Velocidade: o sintético se concentra no meio, com menos paradas e menos picos",
            "Velocidade em 80 trechos de tempo iguais de cada viagem (≈ 1/80 da duração), calculada do mesmo jeito "
            "nas curvas γ reais e sintéticas.")


def fig_perfil(P):
    fig, axes = grade(len(P), sharey=True)
    bins = np.linspace(0, 1, 21)
    mid = 0.5 * (bins[1:] + bins[:-1])
    for ax, (tag, d) in zip(axes, P.items()):
        for v, rel, c in [(d["v_real_ping"], d["rel_real"], C_REAL), (d["v_sint_ping"], d["rel_sint"], C_SYN)]:
            idx = np.clip(np.digitize(rel, bins) - 1, 0, 19)
            qs = np.array([np.nanpercentile(v[idx == i], [25, 50, 75]) if np.any(idx == i) else [np.nan] * 3
                           for i in range(20)])
            ax.fill_between(mid, qs[:, 0], qs[:, 2], color=c, alpha=0.15, lw=0)
            ax.plot(mid, qs[:, 1], c=c)
        ax.set_title(rotulo(tag))
        ax.set_xlabel("fração do percurso")
    axes[0].set_ylabel("velocidade (km/h)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    legenda_real_sint(fig)
    savefig(fig, "07_perfil_velocidade.png", "Onde o ônibus anda rápido e onde para",
            "Mediana (linha) e faixa interquartil (sombra) da velocidade em 20 trechos da rota, "
            "a partir das curvas γ. As medianas coincidem; a faixa sintética, mais estreita, mostra menos variação entre viagens.")


def fig_horario(P):
    """Mediana e faixa interquartil da duracao por bloco de 2 h de partida, dia util e fim de semana."""
    fig, axes = grade(len(P), sharey=False)
    blocos = np.arange(4, 24, 2)
    for ax, (tag, d) in zip(axes, P.items()):
        st = pd.to_datetime(d["R"]["db"]["start_dt"])
        ini = pd.to_datetime(d["ms"]["inicio"])
        series = [(d["T_real"], st, C_REAL, 10), (d["T_sint"], ini, C_SYN, 3)]
        for fds, ls in ((False, "-"), (True, ":")):
            for T, t, c, nmin in series:
                h = t.dt.hour.to_numpy()
                m = (t.dt.dayofweek.to_numpy() >= 5) == fds
                xs, q = [], []
                for b in blocos:
                    v = T[m & (h >= b) & (h < b + 2)]
                    if len(v) >= nmin:
                        xs.append(b + 1)
                        q.append(np.percentile(v, [25, 50, 75]))
                if not q:
                    continue
                q = np.array(q)
                if not fds:
                    ax.fill_between(xs, q[:, 0], q[:, 2], color=c, alpha=0.13, lw=0)
                ax.plot(xs, q[:, 1], ls, c=c, lw=1.8 if not fds else 1.4)
        ax.set_xlim(4, 24); ax.set_xticks(range(6, 25, 6))
        ax.set_title(rotulo(tag))
    axes[0].set_ylabel("duração (min)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    legenda_real_sint(fig, [plt.Line2D([], [], c=INK2, lw=1.8, label="dia útil (sombra = p25–p75)"),
                            plt.Line2D([], [], c=INK2, lw=1.4, ls=":", label="sáb/dom")])
    savefig(fig, "08_duracao_por_horario.png", "Duração ao longo do dia: dia útil e fim de semana",
            "Mediana por bloco de 2 h de partida (real com ≥ 10 viagens no bloco). "
            "O sintético deve acompanhar o pico, o vale e a diferença do fim de semana.")


def fig_discriminador(resumo):
    tags = resumo["rota"].tolist()
    y = np.arange(len(tags))[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(tags) + 2.2))
    ax.axvline(0.5, c=INK2, lw=1, ls="--")
    for col, c, lab, dy in [("AUC_controle", "#9a988f", "controle: real × real", 0.22),
                            ("AUC_forma", C_REAL, "real × sintético, só a forma γ", 0.0),
                            ("AUC_forma_T_S", C_SYN, "real × sintético, forma + T + S", -0.22)]:
        ax.errorbar(resumo[col], y + dy, xerr=resumo[col + "_dp"], fmt="o", c=c, ms=8, mec=SURF,
                    mew=1.5, elinewidth=1.5, capsize=0, label=lab)
    ax.set_yticks(y, [rotulo(t) for t in tags])
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("AUC do classificador (validação cruzada 5×, média de 10 sorteios)")
    ax.text(0.505, y[0] + 0.55, "0,5 = não distingue", fontsize=9.5, color=INK2)
    ax.text(0.99, y[0] + 0.55, "1 = separa tudo", fontsize=9.5, color=INK2, ha="right")
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig(fig, "02_teste_discriminador.png", "Um classificador consegue separar real de sintético?",
            "Floresta aleatória treinada com a curva γ (19 pontos) e, na versão laranja, também T e S. "
            "Quanto mais perto do controle, mais realista.")


def fig_pca(P):
    fig, axes = grade(len(P), sharey=False)
    for ax, (tag, d) in zip(axes, P.items()):
        pca = PCA(n_components=2).fit(d["G_real"] - TAU)
        zr = pca.transform(d["G_real"] - TAU)
        zs = pca.transform(d["G_sint"] - TAU)
        ax.scatter(zr[:, 0], zr[:, 1], s=6, c=C_REAL, alpha=0.35, lw=0)
        ax.scatter(zs[:, 0], zs[:, 1], s=14, c=C_SYN, alpha=0.85, lw=0)
        ev = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"CP1 ({ev[0]:.0f}% da variância)")
        ax.set_ylabel(f"CP2 ({ev[1]:.0f}%)")
        ax.set_title(rotulo(tag))
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    legenda_real_sint(fig)
    savefig(fig, "09_pca_curvas.png", "Cobertura do espaço de formas (PCA das curvas γ)",
            "Componentes principais ajustadas nas viagens reais. Sintéticas devem cair dentro da nuvem real "
            "e espalhadas, não amontoadas num ponto.")


def fig_dcr(P, D):
    fig, axes = grade(len(P), sharey=True)
    for ax, (tag, d) in zip(axes, P.items()):
        ctrl, sint = D[tag]
        for x, c in [(ctrl, C_REAL), (sint, C_SYN)]:
            xs = np.sort(x)
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", c=c)
            ax.axvline(np.median(x), c=c, lw=1, ls=":")
        ax.set_title(rotulo(tag))
        ax.set_xlabel("distância à real mais próxima")
        ax.text(0.97, 0.06, f"razão das medianas {br(np.median(sint) / np.median(ctrl))}",
                transform=ax.transAxes, ha="right", fontsize=9.5, color=INK)
    axes[0].set_ylabel("fração acumulada")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    h = [plt.Line2D([], [], c=C_REAL, lw=2.5, label="real fora do banco → banco (controle)"),
         plt.Line2D([], [], c=C_SYN, lw=2.5, label="sintético → banco")]
    fig.legend(handles=h, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.005), fontsize=10.5)
    savefig(fig, "10_nao_copia_dcr.png", "Não cópia: distância ao registro real mais próximo (DCR)",
            "Laranja à esquerda do azul = sintéticas mais perto das reais do que reais entre si. "
            "Valores perto de zero seriam cópias.")


def fig_scorecard(resumo):
    cols = [("erro_T_pct", "erro mediana\nduração (%)", lambda v: abs(v) / 6),
            ("KS_T", "KS\nduração", lambda v: v / 0.3),
            ("KS_v", "KS\nvelocidade", lambda v: v / 0.3),
            ("razao_CV_T", "variabilidade\nduração (÷ real)", lambda v: abs(1 - v) / 0.3),
            ("razao_p99", "p99 velocidade\n(÷ real)", lambda v: abs(1 - v) / 0.5),
            ("AUC_forma_T_S", "AUC\ndiscriminador", lambda v: (v - 0.5) / 0.4),
            ("razao_DCR", "DCR\n(÷ controle)", lambda v: abs(1 - v) / 0.5)]
    M = np.array([[f(r[c]) for c, _, f in cols] for _, r in resumo.iterrows()]).clip(0, 1)
    fig, ax = plt.subplots(figsize=(13, 0.34 * len(resumo) + 2.2))
    ax.imshow(M, cmap="Oranges", vmin=0, vmax=1.4, aspect="auto")
    for i, (_, r) in enumerate(resumo.iterrows()):
        for j, (c, _, _) in enumerate(cols):
            v = r[c]
            ax.text(j, i, br(v, "{:+.1f}") if c == "erro_T_pct" else br(v), ha="center", va="center",
                    fontsize=9.5, color=INK)
    ax.set_xticks(range(len(cols)), [c[1] for c in cols], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(resumo)), [rotulo(t) for t in resumo["rota"]])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    savefig(fig, "00_resumo_desempenho.png", "Resumo do desempenho por rota",
            "Cor mais forte = mais longe do ideal (0% de erro, KS 0, razões 1, AUC 0,5). "
            "Os números são os valores medidos.")


# ============================================================
def main():
    print("Carregando dados (05_analise.load_all)...")
    routes, meta_syn, pts = analise.load_all()
    print("Preparando curvas e velocidades...")
    P = preparar(routes, meta_syn, pts)
    rng = np.random.default_rng(2026)

    print("Discriminador e DCR...")
    rows, D = [], {}
    for tag, d in P.items():
        auc = auc_discriminador(d, rng)
        D[tag] = dcr(d, rng)
        med = lambda a: np.median(a)  # noqa: E731
        iqr = lambda a: np.subtract(*np.percentile(a, [75, 25]))  # noqa: E731
        rows.append(dict(
            rota=tag, n_real=len(d["T_real"]), n_sint=len(d["T_sint"]),
            erro_T_pct=100 * (med(d["T_sint"]) / med(d["T_real"]) - 1),
            erro_S_pct=100 * (med(d["S_sint"]) / med(d["S_real"]) - 1),
            erro_v_pct=100 * (med(d["v_sint"]) / med(d["v_real"]) - 1),
            KS_T=ks_2samp(d["T_real"], d["T_sint"]).statistic,
            KS_S=ks_2samp(d["S_real"], d["S_sint"]).statistic,
            KS_v=ks_2samp(d["v_real"], d["v_sint"]).statistic,
            razao_CV_T=(d["T_sint"].std() / d["T_sint"].mean()) / (d["T_real"].std() / d["T_real"].mean()),
            razao_IQR_v=iqr(d["v_sint"]) / iqr(d["v_real"]),
            v_p99_real=np.percentile(d["v_real_ping"], 99), v_p99_sint=np.percentile(d["v_sint_ping"], 99),
            AUC_forma=auc["forma"][0], AUC_forma_dp=auc["forma"][1],
            AUC_forma_T_S=auc["forma_T_S"][0], AUC_forma_T_S_dp=auc["forma_T_S"][1],
            AUC_controle=auc["controle"][0], AUC_controle_dp=auc["controle"][1],
            razao_DCR=np.median(D[tag][1]) / np.median(D[tag][0]),
        ))
        print(f"  {tag}: AUC forma+T+S {auc['forma_T_S'][0]:.2f} | controle {auc['controle'][0]:.2f}")
    resumo = pd.DataFrame(rows)
    resumo["razao_p99"] = resumo["v_p99_sint"] / resumo["v_p99_real"]
    resumo.to_csv(RES / "desempenho_resumo.csv", index=False, float_format="%.4f")
    print(resumo.round(3).to_string())

    print("\nFiguras:")
    fig_scorecard(resumo)
    fig_painel(P, resumo)
    fig_discriminador(resumo)
    fig_ecdf(P, "T", "03_ecdf_duracao.png", "duração (min)", "Distribuição da duração das viagens")
    fig_ecdf(P, "v", "03b_ecdf_velocidade_media.png", "velocidade média (km/h)",
             "Distribuição da velocidade média por viagem")
    fig_qq(P)
    fig_variabilidade(resumo)
    fig_vel_pontual(P, resumo)
    fig_perfil(P)
    fig_horario(P)
    fig_pca(P)
    fig_dcr(P, D)
    print("\nConcluido.")


if __name__ == "__main__":
    main()
