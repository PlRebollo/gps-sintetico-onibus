"""
05_analise.py
=============
PASSO 5: analise dos resultados. Gera metricas (resultados/metricas_qualidade.csv)
e figuras em figuras/ comparando dados reais e sinteticos, a qualidade das rotas
descobertas e o comportamento da sintese Dirichlet.

Figuras por rota em grade (25 sentidos), faixas horarias da rota
(meta["faixas"]) e duracao por hora e tipo de dia no lugar das 5 faixas fixas.

Uso:
    cd pipeline && python 05_analise.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

import config as cfg  # noqa: E402
from core.shapes import load_shape_models  # noqa: E402
from core.trips import build_configs  # noqa: E402
from functional_synthesis_dirichlet import build_neighbour_cache, sample_anchor  # noqa: E402
from functional_synthesis import dp_align  # noqa: E402
from legs import build_route_legs, load_route  # noqa: E402

warnings.filterwarnings("ignore")
DADOS = cfg.ROOT / "dados_sinteticos"
FIG = cfg.ROOT / "figuras"
RES = cfg.BASE / "resultados"
ANTERIOR = DADOS / "versao_anterior"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_REAL, C_SYN, C_OUT = "#4d4d4d", "#1f77b4", "#d62728"


def savefig(fig, sub, name):
    path = FIG / sub / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {sub}/{name}")


def grade(n, ncols=5, w=3.2, h=3.0, **kw):
    """Grade de paineis (um por rota); esconde os que sobram."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.5), **kw)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def cores_rotas(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) if i < 20 else plt.get_cmap("Dark2")(i - 20) for i in range(n)]


def ecdf(x):
    x = np.sort(x)
    return x, np.arange(1, len(x) + 1) / len(x)


# ============================================================
# CARGA
# ============================================================
def load_all():
    shape_models = load_shape_models(cfg.SHAPE_PATH)
    configs = build_configs(shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
    routes = {}
    for (linha, sentido, shape_id, _s, _e) in configs:
        tag = f"{linha}_{sentido}"
        r = build_route_legs(linha, sentido, shape_models, configs)
        shape, _ = load_route(linha, shape_id)
        routes[tag] = dict(linha=linha, sentido=sentido, shape_id=shape_id, shape=shape,
                           legs=r["legs"], db=r["db"], meta=r["meta"],
                           shape_model=shape_models.get((linha, shape_id)))
    meta_syn = pd.read_csv(DADOS / "metadados_sinteticos.csv", dtype={"linha": str},
                           parse_dates=["inicio"])
    cols = ["linha", "sentido", "synth_leg_id", "t_s", "lat", "lon", "s_m", "v_ms"]
    pts = pd.concat([pd.read_csv(DADOS / f"synthetic_fulltrip_dirichlet_{l}.csv.gz",
                                 usecols=cols, dtype={"linha": str, "sentido": str})
                     for l in cfg.LINHAS], ignore_index=True)
    for c in ["linha", "sentido", "synth_leg_id"]:
        pts[c] = pts[c].astype("category")
    pts["tag"] = (pts["linha"].astype(str) + "_" + pts["sentido"].astype(str)).astype("category")
    return routes, meta_syn, pts


def route_quality():
    comp = pd.read_csv(RES / "comparacao_rotas.csv", dtype={"linha": str, "shape_id": str})
    grade_v11 = pd.read_csv(RES / "estudo_grade_v11.csv", dtype={"linha": str})
    return comp, grade_v11


def ping_resid(route, legs):
    _, resid = route.match(legs["lat"].to_numpy(), legs["lon"].to_numpy())
    return resid


# ============================================================
# METRICAS
# ============================================================
def metricas(routes, meta_syn, pts, comp):
    rows = []
    for tag, R in routes.items():
        db, legs, meta = R["db"], R["legs"], R["meta"]
        ms = meta_syn[(meta_syn["linha"] == R["linha"]) & (meta_syn["sentido"] == R["sentido"])]
        p = pts[pts["tag"] == tag]
        Tr, Sr = db["T"].to_numpy(float), db["S"].to_numpy(float)
        vr = 3.6 * Sr / Tr
        Ts, Ss = ms["duracao_s"].to_numpy(float), ms["extensao_m"].to_numpy(float)
        vs = ms["velocidade_media_kmh"].to_numpy(float)
        c = comp[(comp["linha"] == R["linha"]) & (comp["sentido"] == R["sentido"])]
        lg = legs.sort_values(["leg_id", "dt"])
        dts = lg.groupby("leg_id")["dt"].diff().dt.total_seconds()
        dss = lg.groupby("leg_id")["s_est"].diff()
        okv = (dts >= 5) & (dts <= 120) & (dss >= 0)
        v99_real = float(np.percentile(3.6 * dss[okv] / dts[okv], 99))
        rows.append(dict(
            rota=tag,
            viagens_reais_selecionadas=meta["n_trips_selecionadas"],
            pernas_reais_validas=len(db),
            pings_reais_na_rota=len(legs),
            viagens_sinteticas=len(ms),
            pings_sinteticos=len(p),
            T_real_med_min=np.median(Tr) / 60, T_sint_med_min=np.median(Ts) / 60,
            S_real_med_km=np.median(Sr) / 1000, S_sint_med_km=np.median(Ss) / 1000,
            v_real_med_kmh=np.median(vr), v_sint_med_kmh=np.median(vs),
            KS_T=ks_2samp(Ts, Tr).statistic, KS_S=ks_2samp(Ss, Sr).statistic,
            KS_v=ks_2samp(vs, vr).statistic,
            CV_T_real=Tr.std() / Tr.mean(), CV_T_sint=Ts.std() / Ts.mean(),
            CV_S_real=Sr.std() / Sr.mean(), CV_S_sint=Ss.std() / Ss.mean(),
            v_max_sint_kmh=p["v_ms"].max() * 3.6,
            v_p99_real_kmh=v99_real,
            v_p99_sint_kmh=float(np.percentile(p["v_ms"] * 3.6, 99)),
            viagens_com_corte_teto=float((ms["frac_tempo_teto_velocidade"] > 0).mean()),
            frac_pernas_parciais_real=float((Sr / meta["route_length_m"] < 0.97).mean()),
            frac_pernas_parciais_sint=float((Ss / meta["route_length_m"] < 0.97).mean()),
            N_eff_med=ms["N_eff"].median(), peso_max_med=ms["peso_max"].median(),
            cobertura_15m_rota=float(c["v11_cov15"].iloc[0]) if len(c) else np.nan,
            razao_comprimento_rota=float(c["v11_razao_oficial"].iloc[0]) if len(c) else np.nan,
            sigma_lateral_m=meta["sigma_lateral_m"],
            n_faixas_util=sum(1 for b, f in meta["faixas"].items()
                              if f[2] is not True and (db["hour_bucket"] == b).any()),
            n_faixas_fds=sum(1 for b, f in meta["faixas"].items()
                             if f[2] is True and (db["hour_bucket"] == b).any()),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "metricas_qualidade.csv", index=False, float_format="%.4f")
    return df


# ============================================================
# FIGURAS
# ============================================================
def fig_mapa_rotas(routes, comp):
    fig, axes = grade(len(routes), h=3.4)
    for ax, (tag, R) in zip(axes, routes.items()):
        legs = R["legs"]
        M = cfg.M_PER_DEG_LAT
        mlon = M * np.cos(np.radians(legs["lat"].mean()))
        ax.hexbin(legs["lon"] * mlon, legs["lat"] * M, gridsize=60, bins="log",
                  cmap="Greys", mincnt=1, linewidths=0)
        sm = R["shape_model"]
        ax.plot(np.asarray(sm["lons"]) * mlon, np.asarray(sm["lats"]) * M, "--",
                c=C_OUT, lw=0.9, label="oficial")
        rota = pd.read_csv(cfg.BASE / "resultados" / f"rota_v11_{R['linha']}_{R['shape_id']}.csv")
        ax.plot(rota["LON"] * mlon, rota["LAT"] * M, c=C_SYN, lw=1.3, label="v11")
        c = comp[(comp["linha"] == R["linha"]) & (comp["sentido"] == R["sentido"])]
        if R["linha"] == "331":
            ver = pd.read_csv(cfg.BASE / "verified_routes" / "route_331_verified.csv")
            ax.plot(ver["LON"] * mlon, ver["LAT"] * M, c="#2ca02c", lw=1.0, label="verificada")
        cov = f" | cob. {c['v11_cov15'].iloc[0]:.0f}%" if len(c) else ""
        ax.set_title(f"{R['linha']} {R['sentido']}{cov}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        if ax is axes[0]:
            ax.legend(fontsize=6, loc="lower left")
    fig.suptitle("Pings reais (cinza, escala log) x shape oficial (tracejado) x rota descoberta "
                 "(cob. = pings a ate 15 m da rota)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "01_rotas", "mapa_rotas_descobertas.png")


def fig_aderencia(routes):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (tag, R), cor in zip(routes.items(), cores_rotas(len(routes))):
        res = ping_resid(R["shape"], R["legs"])
        x, y = ecdf(res)
        ax.plot(x, 100 * y, lw=1.2, label=tag, color=cor)
    ax.axvline(15, c="k", ls=":", lw=0.9)
    ax.text(16, 5, "15 m", fontsize=7)
    ax.set_xlim(0, 50)
    ax.set_xlabel("distancia do ping à rota descoberta (m)")
    ax.set_ylabel("% dos pings (acumulado)")
    ax.set_title("Aderência dos pings à rota descoberta")
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    savefig(fig, "01_rotas", "aderencia_pings_rota.png")


def fig_grade(grade):
    g = grade.copy()
    g["ok"] = (g["completed"] == 1) & g["razao_comprimento"].between(0.9, 1.1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    sc = ax.scatter(g["razao_comprimento"], g["coverage_15m"], c=g["angular_samples"],
                    s=10, cmap="viridis", alpha=0.6)
    ax.axvspan(0.9, 1.1, color="#2ca02c", alpha=0.08)
    ax.set_xlabel("comprimento / oficial"); ax.set_ylabel("cobertura GPS a 15 m (%)")
    ax.set_title("Estudo de grade do v11: razão x cobertura")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("ângulos (N)")
    ax = axes[1]
    piv = g.groupby(["meters", "increase_meters"])["ok"].mean().unstack()
    im = ax.imshow(piv.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set_xlabel("fator de crescimento G"); ax.set_ylabel("raio mínimo r (m)")
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_title("% de execuções válidas (concluiu e razão 0,9–1,1)")
    fig.colorbar(im, ax=ax)
    ax = axes[2]
    cov = g.groupby(["angular_samples", "meters"])["coverage_15m"].median().unstack()
    im = ax.imshow(cov.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(cov.shape[1])); ax.set_xticklabels(cov.columns)
    ax.set_yticks(range(cov.shape[0])); ax.set_yticklabels(cov.index)
    ax.set_xlabel("raio mínimo r (m)"); ax.set_ylabel("ângulos N")
    for (i, j), v in np.ndenumerate(cov.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cobertura GPS mediana (%)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    savefig(fig, "01_rotas", "estudo_grade_v11.png")


def fig_dados(routes, metricas_df):
    d = metricas_df.set_index("rota")
    fig, axes = plt.subplots(3, 1, figsize=(max(12, 0.55 * len(d)), 11))
    x = np.arange(len(d))
    ax = axes[0]
    ax.bar(x - 0.2, d["pernas_reais_validas"], 0.4, color=C_REAL, label="pernas reais válidas")
    ax.bar(x + 0.2, d["viagens_sinteticas"], 0.4, color=C_SYN, label="viagens sintéticas")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=45, ha="right")
    ax.set_title("Pernas reais válidas x viagens sintéticas (escala log)"); ax.legend(fontsize=7)
    ax = axes[1]
    ax.bar(x - 0.2, d["pings_reais_na_rota"], 0.4, color=C_REAL, label="pings reais na rota")
    ax.bar(x + 0.2, d["pings_sinteticos"], 0.4, color=C_SYN, label="pings sintéticos (1 Hz)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=45, ha="right")
    ax.set_title("Pontos GPS (escala log)"); ax.legend(fontsize=7)
    ax = axes[2]
    ax.bar(x - 0.2, d["T_real_med_min"], 0.4, color=C_REAL, label="real")
    ax.bar(x + 0.2, d["T_sint_med_min"], 0.4, color=C_SYN, label="sintético")
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=45, ha="right")
    ax.set_title("Duração mediana da viagem (min)"); ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "02_dados", "quantidade_dados.png")


def fig_mapa_real_sint(routes, pts):
    fig, axes = grade(len(routes), h=3.4)
    for ax, (tag, R) in zip(axes, routes.items()):
        M = cfg.M_PER_DEG_LAT
        mlon = M * np.cos(np.radians(R["legs"]["lat"].mean()))
        p = pts[pts["tag"] == tag]
        ax.hexbin(R["legs"]["lon"] * mlon, R["legs"]["lat"] * M, gridsize=55, bins="log",
                  cmap="Greys", mincnt=1, linewidths=0, alpha=0.9)
        ax.hexbin(p["lon"] * mlon, p["lat"] * M, gridsize=55, bins="log",
                  cmap="Blues", mincnt=1, linewidths=0, alpha=0.55)
        ax.set_title(f"{R['linha']} {R['sentido']}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Densidade de pontos: reais (cinza) x sintéticos (azul)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "02_dados", "mapa_real_x_sintetico.png")


def fig_gamma(routes, meta_syn, pts):
    fig, axes = grade(len(routes), sharey=True)
    for ax, (tag, R) in zip(axes, routes.items()):
        db = R["db"]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(db), size=min(60, len(db)), replace=False)
        for i in idx:
            ax.plot(db["tau_grid"].iloc[i], db["gamma"].iloc[i] - db["tau_grid"].iloc[i],
                    c=C_REAL, lw=0.4, alpha=0.4)
        ax.set_title(f"{R['linha']} {R['sentido']}", fontsize=8)
        ms = meta_syn[meta_syn["synth_leg_id"].str.startswith(tag)]
        sample = ms.sample(min(60, len(ms)), random_state=0)
        p = pts[pts["synth_leg_id"].isin(sample["synth_leg_id"].tolist())]
        L = R["meta"]["route_length_m"] if R["meta"]["is_loop"] else None
        for sid, g in p.groupby("synth_leg_id"):
            m = ms[ms["synth_leg_id"] == sid].iloc[0]
            tau = g["t_s"].to_numpy() / m["duracao_s"]
            s = g["s_m"].to_numpy()
            rel = np.mod(s - m["s0_m"], L) if L else s - m["s0_m"]
            ax.plot(tau, rel / m["extensao_m"] - tau, c=C_SYN, lw=0.4, alpha=0.4)
        ax.axhline(0, c="k", lw=0.5)
        ax.set_xlabel("τ = t / T")
    axes[0].set_ylabel("γ(τ) − τ")
    fig.suptitle("Curvas funcionais γ(τ) − τ (adiantado > 0, atrasado < 0): "
                 "60 pernas reais (cinza) x 60 sintéticas (azul)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03_qualidade", "curvas_gamma_real_x_sintetico.png")


def fig_distribuicoes(routes, meta_syn, metricas_df):
    variaveis = [("duracao", "duração (min)", "KS_T"), ("extensao", "extensão (km)", "KS_S"),
                 ("velocidade", "velocidade média (km/h)", "KS_v")]
    for nome, lab, col in variaveis:
        fig, axes = grade(len(routes), sharey=True)
        for ax, (tag, R) in zip(axes, routes.items()):
            db = R["db"]
            ms = meta_syn[(meta_syn["linha"] == R["linha"]) & (meta_syn["sentido"] == R["sentido"])]
            xr = {"duracao": db["T"].to_numpy(float) / 60, "extensao": db["S"].to_numpy(float) / 1000,
                  "velocidade": 3.6 * db["S"].to_numpy(float) / db["T"].to_numpy(float)}[nome]
            xs = {"duracao": ms["duracao_s"].to_numpy(float) / 60,
                  "extensao": ms["extensao_m"].to_numpy(float) / 1000,
                  "velocidade": ms["velocidade_media_kmh"].to_numpy(float)}[nome]
            a, b = ecdf(xr); ax.plot(a, b, c=C_REAL, lw=1.4, label="real")
            a, b = ecdf(xs); ax.plot(a, b, c=C_SYN, lw=1.4, label="sintético")
            ks = metricas_df.set_index("rota").loc[tag, col]
            ax.set_title(f"{tag}  KS={ks:.2f}", fontsize=8)
            ax.set_xlabel(lab, fontsize=7)
        axes[0].set_ylabel("ECDF")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Distribuição real x sintética: {lab} (ECDF) e estatística KS", fontweight="bold")
        fig.tight_layout()
        savefig(fig, "03_qualidade", f"distribuicoes_ecdf_{nome}.png")


def fig_qq(routes, meta_syn):
    fig, axes = grade(len(routes))
    for ax, (tag, R) in zip(axes, routes.items()):
        ms = meta_syn[(meta_syn["linha"] == R["linha"]) & (meta_syn["sentido"] == R["sentido"])]
        xr = np.sort(R["db"]["T"].to_numpy(float) / 60)
        xs = np.sort(ms["duracao_s"].to_numpy(float) / 60)
        q = np.linspace(0.01, 0.99, 99)
        ax.plot(np.quantile(xr, q), np.quantile(xs, q), "o", ms=2.5, c=C_SYN)
        lim = [min(xr.min(), xs.min()), max(xr.max(), xs.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(tag, fontsize=8)
        ax.set_xlabel("quantis reais (min)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("quantis sintéticos (min)", fontsize=7)
    fig.suptitle("QQ-plot da duração (sintético x real)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03_qualidade", "qq_duracao.png")


def fig_velocidade_percurso(routes, pts, meta_syn):
    fig, axes = grade(len(routes))
    for ax, (tag, R) in zip(axes, routes.items()):
        legs = R["legs"].sort_values(["leg_id", "dt"])
        dt = legs.groupby("leg_id")["dt"].diff().dt.total_seconds()
        ds = legs.groupby("leg_id")["s_est"].diff()
        ok = (dt >= 5) & (dt <= 120) & (ds >= 0)
        v = (ds[ok] / dt[ok]) * 3.6
        first = legs.groupby("leg_id")["s_est"].transform("first")
        last = legs.groupby("leg_id")["s_est"].transform("last")
        rel = ((legs["s_est"] - first) / (last - first).replace(0, np.nan))[ok]
        bins = np.linspace(0, 1, 11)
        mid = 0.5 * (bins[1:] + bins[:-1])
        vr = [np.nanmedian(v[(rel >= bins[i]) & (rel < bins[i + 1])]) for i in range(10)]
        ax.plot(mid, vr, "o-", c=C_REAL, label="real")
        p = pts[pts["tag"] == tag]
        ms = meta_syn[meta_syn["synth_leg_id"].str.startswith(tag)].set_index("synth_leg_id")
        L = R["meta"]["route_length_m"] if R["meta"]["is_loop"] else None
        sub = p.copy()
        sub["ext"] = sub["synth_leg_id"].map(ms["extensao_m"])
        sub["s0"] = sub["synth_leg_id"].map(ms["s0_m"])
        relS = (np.mod(sub["s_m"] - sub["s0"], L) if L else (sub["s_m"] - sub["s0"])) / sub["ext"]
        bidx = np.clip((relS * 10).astype(int), 0, 9)
        vs = sub.groupby(bidx)["v_ms"].median() * 3.6
        ax.plot(mid, vs.reindex(range(10)).to_numpy(), "o-", c=C_SYN, label="sintético")
        ax.set_title(tag, fontsize=8)
        ax.set_xlabel("posição relativa na viagem (γ)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("velocidade mediana (km/h)")
            ax.legend(fontsize=7)
    fig.suptitle("Perfil de velocidade ao longo da viagem", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03_qualidade", "velocidade_ao_longo_da_viagem.png")


def fig_horario(routes, meta_syn):
    """Duracao mediana por hora de partida, dia util e fim de semana, real x sintetico."""
    fig, axes = grade(len(routes), w=3.6)
    for ax, (tag, R) in zip(axes, routes.items()):
        db = R["db"]
        st = pd.to_datetime(db["start_dt"])
        ms = meta_syn[(meta_syn["linha"] == R["linha"]) & (meta_syn["sentido"] == R["sentido"])]
        ini = pd.to_datetime(ms["inicio"])
        for fds, ls in ((False, "-"), (True, "--")):
            r = db[(st.dt.dayofweek >= 5) == fds].assign(h=st.dt.hour)
            g = r.groupby("h")["T"].agg(["median", "size"])
            g = g[g["size"] >= 10]
            ax.plot(g.index + 0.5, g["median"] / 60, ls, c=C_REAL, lw=1.6)
            s = ms[(ini.dt.dayofweek >= 5) == fds].assign(h=ini.dt.hour)
            g = s.groupby("h")["duracao_s"].agg(["median", "size"])
            g = g[g["size"] >= 3]
            ax.plot(g.index + 0.5, g["median"] / 60, ls, c=C_SYN, lw=1.2)
        ax.set_xlim(4, 24); ax.set_xticks(range(6, 25, 6))
        ax.set_title(tag, fontsize=8)
        ax.set_xlabel("hora de partida", fontsize=7)
    axes[0].set_ylabel("duração mediana (min)")
    h = [plt.Line2D([], [], c=C_REAL, label="real"), plt.Line2D([], [], c=C_SYN, label="sintético"),
         plt.Line2D([], [], c="k", ls="-", label="dia útil"),
         plt.Line2D([], [], c="k", ls="--", label="sáb/dom")]
    fig.legend(handles=h, loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("Duração mediana por hora de partida (real com ≥ 10 viagens na hora)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, "03_qualidade", "duracao_por_hora_tipo_dia.png")


def fig_faixas(routes):
    """Faixas horarias encontradas pela particao por ritmo, por rota e tipo de dia."""
    tags = list(routes)
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(tags) * 2 + 1.5))
    y = 0
    yt, yl = [], []
    for tag in tags[::-1]:
        R = routes[tag]
        db = R["db"]
        for fds, cmap in ((True, plt.get_cmap("Reds")), (False, plt.get_cmap("Blues"))):
            fx = [(b, f) for b, f in R["meta"]["faixas"].items() if (f[2] is True) == fds]
            for j, (b, (a, e, _)) in enumerate(fx):
                n = int((db["hour_bucket"] == b).sum())
                if n == 0:
                    continue
                med = db.loc[db["hour_bucket"] == b, "T"].median()
                ax.barh(y, e - a, left=a, color=cmap(0.35 + 0.5 * (j % 2)), edgecolor="white", lw=0.8)
                if e - a >= 1.2:
                    ax.text((a + e) / 2, y, f"{med / 60:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white")
            yt.append(y); yl.append(f"{tag} {'sáb/dom' if fds else 'útil'}")
            y += 1
        y += 0.4
    ax.set_yticks(yt, yl, fontsize=7)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("hora de partida")
    ax.set_title("Faixas horárias encontradas pela partição por ritmo (número = duração mediana da faixa, min). "
                 "Só faixas com viagens reais.", fontsize=10, loc="left")
    fig.tight_layout()
    savefig(fig, "03_qualidade", "faixas_por_rota.png")


def groups_by_leg(pts):
    return {k: g for k, g in pts.groupby("synth_leg_id", observed=True)}


def fig_copias(routes, pts, meta_syn):
    grupos = groups_by_leg(pts)
    fig, axes = grade(len(routes))
    rng = np.random.default_rng(0)
    for ax, (tag, R) in zip(axes, routes.items()):
        db = R["db"]
        G = np.vstack(db["gamma"].to_numpy())
        ctrl = []
        for i in rng.choice(len(G), size=min(150, len(G)), replace=False):
            d = np.sqrt(((G - G[i]) ** 2).sum(1)); d[i] = np.inf
            ctrl.append(d.min())
        ms_all = meta_syn[meta_syn["synth_leg_id"].str.startswith(tag)]
        ms = ms_all.sample(min(150, len(ms_all)), random_state=1)
        L = R["meta"]["route_length_m"] if R["meta"]["is_loop"] else None
        dmin = []
        for _, m in ms.iterrows():
            g = grupos[m["synth_leg_id"]]
            tau = g["t_s"].to_numpy() / m["duracao_s"]
            rel = (np.mod(g["s_m"].to_numpy() - m["s0_m"], L) if L else g["s_m"].to_numpy() - m["s0_m"]) / m["extensao_m"]
            gi = np.interp(np.linspace(0, 1, 81), tau, rel)
            dmin.append(np.sqrt(((G - gi[None, :]) ** 2).sum(1)).min())
        ax.hist(ctrl, bins=25, density=True, alpha=0.6, color=C_REAL, label="real → real (controle)")
        ax.hist(dmin, bins=25, density=True, alpha=0.6, color=C_SYN, label="sintético → real")
        ax.set_title(tag, fontsize=8)
        ax.set_xlabel("distância γ à perna real mais próxima", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("densidade")
            ax.legend(fontsize=6)
    fig.suptitle("Não cópia: distância à perna real mais próxima", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "04_diversidade", "distancia_a_real_mais_proxima.png")


def fig_trajetorias_exemplo(routes, pts, meta_syn):
    grupos = groups_by_leg(pts)
    fig, axes = grade(len(routes), h=3.4)
    for ax, (tag, R) in zip(axes, routes.items()):
        M = cfg.M_PER_DEG_LAT
        mlon = M * np.cos(np.radians(R["legs"]["lat"].mean()))
        sm = R["shape_model"]
        ax.plot(np.asarray(sm["lons"]) * mlon, np.asarray(sm["lats"]) * M, c="#bbbbbb", lw=3, zorder=0)
        ms = meta_syn[meta_syn["synth_leg_id"].str.startswith(tag)]
        ids = ms["synth_leg_id"].sample(min(12, len(ms)), random_state=2)
        for sid in ids:
            g = grupos[sid]
            ax.plot(g["lon"] * mlon, g["lat"] * M, c=C_SYN, lw=0.5, alpha=0.6)
        ax.set_title(tag, fontsize=8)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Exemplos de viagens sintéticas (12 por rota) sobre a rota", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03_qualidade", "exemplos_viagens_sinteticas.png")


def fig_sweep(sweep):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    s2 = sweep[sweep["span"] == 2.0]
    for ax, col, lab, cmap in [
        (axes[0], "KS_medio", "KS médio (menor = mais fiel)", "viridis_r"),
        (axes[1], "div_T_med", "variabilidade de T (div_T)", "RdYlGn"),
        (axes[2], "N_eff", "vizinhos efetivos N_eff", "Blues"),
        (axes[3], "razao_privacidade", "razão de privacidade", "PuOr"),
    ]:
        piv = s2.pivot(index="K", columns="alpha0", values=col)
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
        ax.set_xlabel("α0"); ax.set_ylabel("K")
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(lab, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Estudo de K e α0 (span = 2,0)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "05_dirichlet", "estudo_K_alpha0_mapas.png")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ok = sweep["ok_criterio"].astype(bool)
    sc = ax.scatter(sweep.loc[~ok, "div_T_med"], sweep.loc[~ok, "KS_medio"], s=14,
                    c="#bbbbbb", label="fora do critério")
    sc2 = ax.scatter(sweep.loc[ok, "div_T_med"], sweep.loc[ok, "KS_medio"], s=26,
                     c=sweep.loc[ok, "N_eff"], cmap="Blues", edgecolor="k", lw=0.4,
                     label="critério atendido (cor = N_eff)")
    ax.axvline(0.75, c="k", ls=":", lw=0.9)
    ax.axhline(sweep.loc[ok, "KS_medio"].min(), c="#2ca02c", ls="--", lw=0.8)
    ax.set_xlabel("div_T (variabilidade da duração / real)")
    ax.set_ylabel("KS médio (fidelidade)")
    ax.set_title("Trade-off: fidelidade x variabilidade x mistura de vizinhos")
    fig.colorbar(sc2, ax=ax, label="N_eff"); ax.legend(fontsize=7)
    savefig(fig, "05_dirichlet", "tradeoff_fidelidade_variabilidade.png")


def fig_pesos(meta_syn):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    ax = axes[0]
    circ = meta_syn["sentido"] == "circular"
    for m, lab in ((circ, "circulares"), (~circ, "ida/volta")):
        ax.hist(meta_syn.loc[m, "peso_max"], bins=30, histtype="step", lw=1.4, density=True, label=lab)
    ax.set_xlabel("peso máximo p_max por viagem"); ax.set_title("Pesos Dirichlet (p_max)")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.hist(meta_syn["N_eff"], bins=40, color=C_SYN, alpha=0.8)
    ax.axvline(cfg.SYNTH_K, c="k", ls="--", lw=0.8, label=f"K = {cfg.SYNTH_K}")
    ax.set_xlabel("vizinhos efetivos N_eff = 1/Σp²"); ax.set_title("Mistura efetiva dos vizinhos")
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.scatter(meta_syn["D_medio_vizinhos"], meta_syn["peso_max"], s=5, alpha=0.4, c=C_SYN)
    ax.set_xlabel("distância média D aos vizinhos (norma L2)")
    ax.set_ylabel("peso máximo"); ax.set_title("Distância aos vizinhos x peso máximo")
    fig.tight_layout()
    savefig(fig, "05_dirichlet", "pesos_dirichlet.png")


def fig_exemplo_alinhamento(routes):
    R = routes["030_ida"]
    db = R["db"]
    rng = np.random.default_rng(11)
    anc = sample_anchor(db, db["hour_bucket"].value_counts().index[0], rng)
    cache = build_neighbour_cache(db, anc, cfg.SPAN_RATIO)
    tau = anc["tau_grid"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9))
    ax = axes[0]
    ax.plot(tau, anc["gamma"], c="k", lw=2, label="âncora")
    nb = db.set_index("leg_id").loc[cache.ids[:cfg.SYNTH_K]]
    for lid, row in nb.iterrows():
        ax.plot(tau, row["gamma"], lw=0.9, alpha=0.7)
    ax.set_title(f"γ da âncora e dos {cfg.SYNTH_K} vizinhos mais próximos"); ax.set_xlabel("τ")
    ax = axes[1]
    ax.plot(tau, anc["q"], c="k", lw=2, label="q âncora")
    q2 = nb.iloc[0]["q"]
    ax.plot(tau, q2, c="#ff7f0e", lw=1, label="vizinho (sem alinhar)")
    _, q2r, _ = dp_align(anc["q"], q2, nb.iloc[0]["gamma"], tau)
    ax.plot(tau, q2r, c=C_SYN, lw=1, ls="--", label="vizinho alinhado (DP)")
    ax.set_title("SRVF: alinhamento elástico"); ax.legend(fontsize=7); ax.set_xlabel("τ")
    ax = axes[2]
    r = np.random.default_rng(3)
    from functional_synthesis_dirichlet import synthesize_from_cache
    res = synthesize_from_cache(cache, cfg.SYNTH_K, cfg.SYNTH_ALPHA0, r)
    x = np.arange(cfg.SYNTH_K)
    ax.bar(x - 0.2, res.alpha, 0.4, color="#bbbbbb", label="α_k (prior)")
    ax.bar(x + 0.2, res.weights, 0.4, color=C_SYN, label="p_k (sorteado)")
    ax.set_xticks(x); ax.set_xticklabels([f"v{k+1}\nD={d:.2f}" for k, d in enumerate(res.D)], fontsize=6)
    ax.set_title(f"Pesos Dirichlet (α0={cfg.SYNTH_ALPHA0}, K={cfg.SYNTH_K})"); ax.legend(fontsize=7)
    ax = axes[3]
    ax.plot(tau, res.gamma, c=C_SYN, lw=2, label="γ sintética")
    ax.plot(tau, anc["gamma"], c="k", lw=1, ls=":", label="γ âncora (não entra)")
    ax.set_title("Curva sintética γ(τ)"); ax.legend(fontsize=7); ax.set_xlabel("τ")
    fig.tight_layout()
    savefig(fig, "05_dirichlet", "exemplo_alinhamento_e_sintese.png")


def fig_antes_depois(routes, meta_syn, metricas_df):
    old = {}
    for l in ["020", "030", "331"]:
        p = ANTERIOR / f"synthetic_fulltrip_dirichlet_{l}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["synth_leg_id", "t_s", "s_m"])
        g = df.groupby("synth_leg_id")
        old[l] = dict(T=g["t_s"].max().to_numpy(float), S=(g["s_m"].max() - g["s_m"].min()).to_numpy(float))
    if not old:
        print("  (sem dados da versao anterior; antes_depois_variabilidade.png nao gerada)")
        return
    labels, cvT_old, cvT_new, cvT_real, cvS_old, cvS_new, cvS_real = [], [], [], [], [], [], []
    m = metricas_df.set_index("rota")
    for l, tag in [("020", "020_circular"), ("030", "030_ida"), ("331", "331_circular")]:
        if l not in old:
            continue
        labels.append(l)
        cvT_old.append(old[l]["T"].std() / old[l]["T"].mean())
        cvS_old.append(old[l]["S"].std() / old[l]["S"].mean())
        cvT_new.append(m.loc[tag, "CV_T_sint"]); cvS_new.append(m.loc[tag, "CV_S_sint"])
        cvT_real.append(m.loc[tag, "CV_T_real"]); cvS_real.append(m.loc[tag, "CV_S_real"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(labels))
    for ax, (lab, a, b, c) in zip(axes, [("CV da duração (T)", cvT_old, cvT_new, cvT_real),
                                         ("CV da extensão (S)", cvS_old, cvS_new, cvS_real)]):
        ax.bar(x - 0.27, a, 0.27, color="#bbbbbb", label="versão anterior")
        ax.bar(x, b, 0.27, color=C_SYN, label="versão corrigida")
        ax.bar(x + 0.27, c, 0.27, color=C_REAL, label="real")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(lab); ax.legend(fontsize=7)
    fig.suptitle("Antes x depois das correções: variabilidade (coeficiente de variação)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "05_dirichlet", "antes_depois_variabilidade.png")


def main():
    for d in ["01_rotas", "02_dados", "03_qualidade", "04_diversidade", "05_dirichlet"]:
        (FIG / d).mkdir(parents=True, exist_ok=True)
    print("Carregando dados...")
    routes, meta_syn, pts = load_all()
    comp, grade = route_quality()
    print("Metricas...")
    met = metricas(routes, meta_syn, pts, comp)
    print(met.round(3).T.to_string())
    print("\nFiguras:")
    fig_mapa_rotas(routes, comp)
    fig_aderencia(routes)
    fig_grade(grade)
    fig_dados(routes, met)
    fig_mapa_real_sint(routes, pts)
    fig_gamma(routes, meta_syn, pts)
    fig_distribuicoes(routes, meta_syn, met)
    fig_qq(routes, meta_syn)
    fig_velocidade_percurso(routes, pts, meta_syn)
    fig_horario(routes, meta_syn)
    fig_faixas(routes)
    fig_copias(routes, pts, meta_syn)
    fig_trajetorias_exemplo(routes, pts, meta_syn)
    sweep = pd.read_csv(RES / "sweep_resumo.csv")
    fig_sweep(sweep)
    fig_pesos(meta_syn)
    fig_exemplo_alinhamento(routes)
    fig_antes_depois(routes, meta_syn, met)
    print("\nConcluido.")


if __name__ == "__main__":
    main()
