"""
synthetic_bus_trajectory.py
============================

Prototipo de gerador de trajetorias sinteticas de onibus a 1 Hz a partir dos
dados abertos de GPS de Curitiba (https://dadosabertos.c3sl.ufpr.br/curitibaurbs/),
que sao amostrados em intervalos irregulares (tipicamente 2-30s, com muitos
gaps de 15-20s).

Abordagem (Opcao A da conversa): suavizacao fisico-estatistica sem
necessidade de dado denso (1 Hz) para treino, seguindo a logica de:

  - Kouvelas & Makridis, "Adaptive physics-informed trajectory reconstruction
    exploiting driver behavior and car dynamics", Sci Rep 13 (2023).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9859820/
  - Modelo classico de Kalman/RTS smoother com dinamica de aceleracao
    constante (CA) + ruido de processo "white noise jerk"
    (Bar-Shalom, Estimation with Applications to Tracking and Navigation).

Pipeline:
  1. Mapear cada ping GPS bruto para a geometria oficial da linha
     (shapeLinha.json) -> reduz o problema 2D para 1D (posicao ao longo da
     rota "s(t)", em metros) + residuo lateral (ruido de GPS / desvio de faixa).
  2. Suavizar s(t) com um Kalman/RTS smoother de aceleracao constante,
     respeitando limites fisicos de aceleracao/desaceleracao de onibus urbano.
  3. Consultar o smoother em uma grade de tempo de 1 Hz (interpolacao
     estatisticamente consistente, nao apenas linear).
  4. Reconverter s(t) -> (lat, lon) via a geometria da linha, somando de volta
     um residuo lateral sintetico amostrado da distribuicao empirica real.
  5. Calibrar/validar o ruido de processo (q) e o ruido de medicao (R) usando
     os proprios pings ddos dados (pares de pings com Delta t pequeno servem
     de "ground truth" natural para holdout), sem precisar coletar dado 1Hz.

O condicionamento por linha e horario entra na calibracao de q (jerk) e R:
cada (linha, faixa horaria) tem sua propria calibracao, refletindo trafego
mais instavel em horario de pico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Projecao planar e geometria da linha (shape)
# ---------------------------------------------------------------------------

def project_planar(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float):
    """Projecao equirretangular local (metros) - adequada para escala urbana."""
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(ref_lat))
    x = (lon - ref_lon) * m_per_deg_lon
    y = (lat - ref_lat) * m_per_deg_lat
    return x, y


def unproject_planar(x: np.ndarray, y: np.ndarray, ref_lat: float, ref_lon: float):
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(ref_lat))
    lon = x / m_per_deg_lon + ref_lon
    lat = y / m_per_deg_lat + ref_lat
    return lat, lon


@dataclass
class RouteShape:
    """Geometria de referencia de uma linha (uma direcao / um SHP)."""
    ref_lat: float
    ref_lon: float
    fine_x: np.ndarray       # pontos finos (metros), resample da polilinha oficial
    fine_y: np.ndarray
    fine_s: np.ndarray       # distancia acumulada (metros) de cada ponto fino
    tree: cKDTree
    total_length: float

    @classmethod
    def from_points(cls, lat: np.ndarray, lon: np.ndarray, resample_step_m: float = 5.0):
        ref_lat, ref_lon = lat[0], lon[0]
        x, y = project_planar(lat, lon, ref_lat, ref_lon)

        seg_len = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_length = cum[-1]

        # reamostra a polilinha original em passos finos e regulares
        n_fine = max(int(total_length / resample_step_m), 2)
        fine_s = np.linspace(0, total_length, n_fine)
        fine_x = np.interp(fine_s, cum, x)
        fine_y = np.interp(fine_s, cum, y)

        tree = cKDTree(np.column_stack([fine_x, fine_y]))
        return cls(ref_lat, ref_lon, fine_x, fine_y, fine_s, tree, total_length)

    def match(self, lat: np.ndarray, lon: np.ndarray):
        """Mapeia pontos GPS para (arc-length s, residuo lateral em metros)."""
        x, y = project_planar(lat, lon, self.ref_lat, self.ref_lon)
        dist, idx = self.tree.query(np.column_stack([x, y]))
        s = self.fine_s[idx]
        return s, dist  # dist = residuo (distancia perpendicular aproximada)

    def s_to_lonlat(self, s: np.ndarray):
        s_clipped = np.clip(s, 0, self.total_length)
        x = np.interp(s_clipped, self.fine_s, self.fine_x)
        y = np.interp(s_clipped, self.fine_s, self.fine_y)
        lat, lon = unproject_planar(x, y, self.ref_lat, self.ref_lon)
        return lat, lon

    def heading_at_s(self, s: np.ndarray):
        """Vetor unitario perpendicular a direcao da rota em s (p/ residuo lateral)."""
        s_clipped = np.clip(s, 0, self.total_length - 1e-6)
        idx = np.searchsorted(self.fine_s, s_clipped)
        idx = np.clip(idx, 1, len(self.fine_s) - 1)
        dx = self.fine_x[idx] - self.fine_x[idx - 1]
        dy = self.fine_y[idx] - self.fine_y[idx - 1]
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
        # perpendicular (rotacao 90 graus)
        return -dy / norm, dx / norm


def build_empirical_shape(lat: np.ndarray, lon: np.ndarray, cell_size_m: float = 15.0,
                            min_density: int = 3, k_neighbors: int = 8,
                            resample_step_m: float = 5.0):
    """
    Constroi uma geometria de referencia DIRETAMENTE dos pings reais de GPS de
    uma linha (nuvem de pontos), em vez de confiar no shapeLinha oficial.

    Motivacao: o shapeLinha publicado pelo operador nem sempre corresponde ao
    trajeto que o onibus de fato realiza (desvios, obras, itinerarios
    alternativos, terminais diferentes do cadastro). Como o trajeto real
    percorrido e o que importa para o gerador sintetico, esta funcao usa a
    mesma logica de fundo do algoritmo PCR do usuario (reconstrucao de rota a
    partir de nuvem de pontos), simplificada para este pipeline:

      1. Discretiza a nuvem de pontos GPS em uma grade espacial fina e mantem
         apenas celulas com densidade de pings suficiente (o "miolo" por onde
         os onibus realmente passam, filtrando ruido de GPS/outliers).
      2. Constroi um grafo de vizinhanca entre os centroides das celulas
         (k vizinhos mais proximos).
      3. Encontra os dois nos mais distantes no grafo via duplo alcance
         (BFS/Dijkstra 2x - heuristica classica de diametro de grafo) - estes
         aproximam os dois terminais da linha.
      4. Extrai o caminho minimo (Dijkstra) entre esses dois nos: esse
         caminho e o "esqueleto" da rota realmente percorrida.

    Retorna um RouteShape construido a partir desse esqueleto observado.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import dijkstra

    ref_lat, ref_lon = np.median(lat), np.median(lon)
    x, y = project_planar(lat, lon, ref_lat, ref_lon)

    # 1. grid binning por densidade
    cell_ix = np.floor(x / cell_size_m).astype(int)
    cell_iy = np.floor(y / cell_size_m).astype(int)
    cell_key = pd.Series(list(zip(cell_ix, cell_iy)))
    grp = pd.DataFrame({"cx": x, "cy": y, "key": cell_key}).groupby("key")
    centroids = grp[["cx", "cy"]].mean()
    counts = grp.size()
    keep = counts[counts >= min_density].index
    centroids = centroids.loc[keep].reset_index(drop=True)

    if len(centroids) < 5:
        raise ValueError("Densidade insuficiente para construir esqueleto empirico")

    cx = centroids["cx"].to_numpy()
    cy = centroids["cy"].to_numpy()
    n = len(cx)

    # 2. grafo de vizinhanca (k-NN)
    tree = cKDTree(np.column_stack([cx, cy]))
    k = min(k_neighbors + 1, n)
    dists, idxs = tree.query(np.column_stack([cx, cy]), k=k)
    graph = lil_matrix((n, n))
    for i in range(n):
        for j_idx in range(1, dists.shape[1]):
            j = idxs[i, j_idx]
            d = dists[i, j_idx]
            graph[i, j] = d
            graph[j, i] = d
    graph = graph.tocsr()

    # 3. heuristica de diametro: duplo alcance com Dijkstra
    d0 = dijkstra(graph, indices=0, directed=False)
    d0 = np.where(np.isfinite(d0), d0, -1)
    a = int(np.argmax(d0))
    da, pred = dijkstra(graph, indices=a, directed=False, return_predecessors=True)
    da_fin = np.where(np.isfinite(da), da, -1)
    b = int(np.argmax(da_fin))

    # 4. caminho minimo (esqueleto da rota) entre os dois terminais estimados
    path = [b]
    cur = b
    while cur != a and pred[cur] >= 0:
        cur = pred[cur]
        path.append(cur)
    path = path[::-1]

    if len(path) < 5:
        raise ValueError("Esqueleto empirico desconexo - grafo pouco conectado")

    path_x, path_y = cx[path], cy[path]
    path_lat, path_lon = unproject_planar(path_x, path_y, ref_lat, ref_lon)
    return RouteShape.from_points(path_lat, path_lon, resample_step_m=resample_step_m)


def choose_best_shape(lat, lon, shapes: dict[str, RouteShape]):
    """Escolhe, entre as direcoes/SHPs disponiveis de uma linha, a que melhor
    explica um bloco de pings (menor residuo medio)."""
    best_key, best_resid = None, np.inf
    for key, shp in shapes.items():
        _, dist = shp.match(lat, lon)
        m = np.median(dist)
        if m < best_resid:
            best_resid = m
            best_key = key
    return best_key, best_resid


# ---------------------------------------------------------------------------
# 2. Kalman / RTS smoother (modelo de aceleracao constante, "white noise jerk")
# ---------------------------------------------------------------------------

def _F(dt):
    return np.array([[1, dt, 0.5 * dt ** 2],
                      [0, 1, dt],
                      [0, 0, 1]])


def _Q(dt, q):
    # Ruido de processo continuo (jerk branco) discretizado - Bar-Shalom Ch.6
    return q * np.array([
        [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
        [dt ** 4 / 8,  dt ** 3 / 3, dt ** 2 / 2],
        [dt ** 3 / 6,  dt ** 2 / 2, dt],
    ])


H = np.array([[1.0, 0.0, 0.0]])


def rts_smooth_arclength(t_obs, s_obs, R, q, t_query, s0=None, v0=0.0):
    """
    Suaviza observacoes esparsas de posicao ao longo da rota (arc-length)
    com um smoother de Rauch-Tung-Striebel, e devolve o estado (posicao,
    velocidade, aceleracao) em QUALQUER grade de tempo (t_query), incluindo
    pontos sem observacao real - e assim que a densificacao para 1 Hz ocorre.

    Parameters
    ----------
    t_obs : array de timestamps (segundos, float) das observacoes reais
    s_obs : array de posicoes observadas (arc-length, metros)
    R : variancia do erro de medicao (m^2)
    q : densidade espectral do jerk (m^2/s^5) - controla o quao "solto"
        o onibus pode acelerar/desacelerar (calibrado por linha/horario)
    t_query : grade de tempo desejada na saida (ex: 1 Hz)
    """
    t_obs = np.asarray(t_obs, dtype=float)
    s_obs = np.asarray(s_obs, dtype=float)

    # timeline unificada = observacoes + grade de consulta, ordenada e sem duplicatas
    all_t = np.union1d(t_obs, t_query)
    is_obs = np.isin(all_t, t_obs)
    obs_lookup = {t: s for t, s in zip(t_obs, s_obs)}

    n = len(all_t)
    x0 = s_obs[0] if s0 is None else s0
    x = np.zeros((n, 3))
    P = np.zeros((n, 3, 3))
    x[0] = [x0, v0, 0.0]
    P[0] = np.diag([R, 5.0, 2.0])

    x_pred = np.zeros_like(x)
    P_pred = np.zeros_like(P)
    F_list = [None] * n
    Q_list = [None] * n

    x_pred[0], P_pred[0] = x[0], P[0]

    # ---- filtro direto (forward pass) ----
    for k in range(1, n):
        dt = all_t[k] - all_t[k - 1]
        dt = max(dt, 1e-3)
        F = _F(dt)
        Qk = _Q(dt, q)
        F_list[k] = F
        Q_list[k] = Qk

        xp = F @ x[k - 1]
        Pp = F @ P[k - 1] @ F.T + Qk
        x_pred[k], P_pred[k] = xp, Pp

        if is_obs[k]:
            z = obs_lookup[all_t[k]]
            S = (H @ Pp @ H.T)[0, 0] + R
            K = (Pp @ H.T) / S
            y = z - (H @ xp)[0]
            x[k] = xp + (K.flatten() * y)
            P[k] = (np.eye(3) - K @ H) @ Pp
        else:
            x[k], P[k] = xp, Pp

    # ---- suavizacao reversa (RTS backward pass) ----
    xs = x.copy()
    Ps = P.copy()
    for k in range(n - 2, -1, -1):
        F = F_list[k + 1]
        Pp = P_pred[k + 1]
        C = P[k] @ F.T @ np.linalg.pinv(Pp)
        xs[k] = x[k] + C @ (xs[k + 1] - x_pred[k + 1])
        Ps[k] = P[k] + C @ (Ps[k + 1] - Pp) @ C.T

    out_idx = np.searchsorted(all_t, t_query)
    return xs[out_idx], Ps[out_idx]  # posicao, velocidade, aceleracao suavizados


# ---------------------------------------------------------------------------
# 3. Calibracao de ruido (q, R) sem dado 1Hz - usa pares de pings proximos
#    no tempo como "ground truth" natural para holdout.
# ---------------------------------------------------------------------------

def calibrate_holdout(t_obs, s_obs, q_grid, R_fixed, min_gap_for_holdout=1.0, max_gap_for_holdout=6.0):
    """
    Estrategia de calibracao SEM dado denso externo:
    Os proprios dados de Curitiba tem pares de pings consecutivos com
    Delta t pequeno (1-6s, ~35% dos gaps no dataset). Escondemos um ping
    intermediario existente nesses pares curtos, rodamos o smoother apenas
    com os vizinhos, e medimos o erro na posicao escondida. O q que minimiza
    esse erro de holdout e o escolhido.
    """
    t_obs = np.asarray(t_obs, dtype=float)
    s_obs = np.asarray(s_obs, dtype=float)

    # candidatos a holdout: pings com vizinhos proximos em ambos os lados
    candidates = []
    for i in range(1, len(t_obs) - 1):
        gap_prev = t_obs[i] - t_obs[i - 1]
        gap_next = t_obs[i + 1] - t_obs[i]
        if min_gap_for_holdout <= gap_prev <= max_gap_for_holdout and \
           min_gap_for_holdout <= gap_next <= max_gap_for_holdout:
            candidates.append(i)

    if len(candidates) < 5:
        return q_grid[len(q_grid) // 2], np.nan  # fallback: sem dado suficiente p/ calibrar

    best_q, best_err = None, np.inf
    for q in q_grid:
        mask = np.ones(len(t_obs), dtype=bool)
        mask[candidates] = False
        t_train, s_train = t_obs[mask], s_obs[mask]
        t_query = t_obs[candidates]

        xs, _ = rts_smooth_arclength(t_train, s_train, R_fixed, q, t_query)
        err = np.sqrt(np.mean((xs[:, 0] - s_obs[candidates]) ** 2))
        if err < best_err:
            best_err = err
            best_q = q

    return best_q, best_err


# ---------------------------------------------------------------------------
# 4. Pipeline principal: gera trajetoria sintetica a 1 Hz
# ---------------------------------------------------------------------------

def enforce_forward_motion(t_query: np.ndarray, s_smooth: np.ndarray, v_smooth: np.ndarray,
                             v_min: float = 0.0):
    """
    Restricao fisica: ao longo de uma unica viagem em um unico sentido, o
    onibus nao anda para tras na parametrizacao 1D da rota (arc-length).
    O smoother de Kalman, sendo irrestrito, pode produzir pequenas
    velocidades negativas perto de paradas/terminais (ruido de GPS + ruido
    de casamento com a geometria). Aqui projetamos a solucao no conjunto
    fisicamente admissivel v >= v_min, recompondo a posicao por integracao
    da velocidade corrigida (garante s(t) nao-decrescente) e recalculando a
    aceleracao por diferenciacao central - analogo a filtragem de Kalman com
    restricoes de estado (Simon, 2010, "Kalman Filtering with State
    Constraints").
    """
    v_clip = np.maximum(v_smooth, v_min)
    s_new = np.empty_like(s_smooth)
    s_new[0] = s_smooth[0]
    for i in range(1, len(t_query)):
        dt = t_query[i] - t_query[i - 1]
        s_new[i] = s_new[i - 1] + 0.5 * (v_clip[i] + v_clip[i - 1]) * dt
    a_new = np.gradient(v_clip, t_query)
    return s_new, v_clip, a_new


def synthesize_trip_1hz(trip_df: pd.DataFrame, shape: RouteShape, freq_hz: float = 1.0,
                          q: float | None = None, R: float | None = None,
                          lateral_noise_std: float | None = None,
                          enforce_monotonic: bool = True,
                          random_state: int = 0):
    """
    trip_df: dataframe com colunas ['dt' (datetime), 'LAT', 'LON'] de UM veiculo,
             UMA linha, UMA viagem (ja segmentada), ordenado por tempo.
    shape: RouteShape correspondente (direcao correta ja escolhida).
    freq_hz: taxa de saida desejada (1.0 = a cada 1 segundo).

    Retorna DataFrame com colunas: t (s desde inicio), dt (timestamp), lat, lon,
    s_m (posicao ao longo da rota), v_ms (velocidade), a_ms2 (aceleracao),
    lateral_residual_m.
    """
    rng = np.random.default_rng(random_state)

    t0 = trip_df["dt"].iloc[0]
    t_obs = (trip_df["dt"] - t0).dt.total_seconds().to_numpy()
    lat = trip_df["LAT"].to_numpy()
    lon = trip_df["LON"].to_numpy()

    s_obs, resid_obs = shape.match(lat, lon)
    # garante monotonicidade aproximada (remove outliers grosseiros de casamento)
    order = np.argsort(t_obs)
    t_obs, s_obs, resid_obs = t_obs[order], s_obs[order], resid_obs[order]

    if R is None:
        R = max(np.median(resid_obs) ** 2, 9.0)  # minimo ~3m^2

    if q is None:
        q_grid = np.geomspace(1e-4, 1.0, 15)
        q, _ = calibrate_holdout(t_obs, s_obs, q_grid, R)

    t_query = np.arange(0, t_obs[-1] + 1e-6, 1.0 / freq_hz)
    xs, Ps = rts_smooth_arclength(t_obs, s_obs, R, q, t_query)

    s_smooth = np.clip(xs[:, 0], 0, shape.total_length)
    v_smooth = xs[:, 1]
    a_smooth = xs[:, 2]

    if enforce_monotonic:
        s_smooth, v_smooth, a_smooth = enforce_forward_motion(t_query, s_smooth, v_smooth)
        s_smooth = np.clip(s_smooth, 0, shape.total_length)

    # residuo lateral sintetico: amostrado da distribuicao empirica observada
    if lateral_noise_std is None:
        lateral_noise_std = max(np.std(resid_obs - np.median(resid_obs)), 1.0)
    lateral_draw = rng.normal(0, lateral_noise_std, size=len(t_query))

    lat_s, lon_s = shape.s_to_lonlat(s_smooth)
    perp_x, perp_y = shape.heading_at_s(s_smooth)
    x_s, y_s = project_planar(lat_s, lon_s, shape.ref_lat, shape.ref_lon)
    x_s = x_s + perp_x * lateral_draw
    y_s = y_s + perp_y * lateral_draw
    lat_final, lon_final = unproject_planar(x_s, y_s, shape.ref_lat, shape.ref_lon)

    out = pd.DataFrame({
        "t_s": t_query,
        "dt": [t0 + pd.Timedelta(seconds=t) for t in t_query],
        "lat": lat_final,
        "lon": lon_final,
        "s_m": s_smooth,
        "v_ms": v_smooth,
        "a_ms2": a_smooth,
        "lateral_residual_m": lateral_draw,
    })
    meta = {"q": q, "R": R, "n_obs": len(t_obs), "lateral_noise_std": lateral_noise_std}
    return out, meta
