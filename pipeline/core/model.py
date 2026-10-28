"""
Modelo de descoberta de rota (caminhada gulosa guiada por destino).

Ideia: caminhar por uma nuvem de pontos GPS escolhendo, a cada passo, o
candidato mais apoiado pelos dados. A pontuação de cada candidato combina
densidade (pontos próximos) e uma guia proporcional ao progresso em direção
ao destino. O cosseno do deslocamento do ponto NÃO pondera o score: serve
apenas de filtro direcional.

Mudanças em relação ao modelo original:
1. Filtro direcional: cada ponto GPS só pontua candidatos alinhados ao seu
   próprio vetor de deslocamento (cos > 0); o valor do cosseno NÃO é usado
   como peso (aplica-se 1/d puro). Evita recuo e não zera candidatos curvos.
2. Guia ao destino g_i = 1 + progresso * max(0, cos(alinhamento)),
   sem constantes arbitrárias; progresso = fração do eixo A->B já percorrida.
3. Direção inicial: vetor resultante dos deslocamentos de TODOS os pontos
   ativos dentro do raio mínimo (não apenas os N mais próximos).
4. Captura apenas em torno do candidato escolhido.
5. Raio adaptativo: crescimento serve de ponte sobre buracos de dados.
6. Otimização exata: pré-filtro por desigualdade triangular (só pontos a
   distância <= 2r podem pontuar algum candidato).
"""

import math

import numpy as np
import pandas as pd

from config import M_PER_DEG_LAT
from core.geometry import route_length_km


# ============================================================
# SUAVIZAÇÃO (Chaikin corner-cutting)
# ============================================================
def chaikin_smooth(lons, lats, iterations=2, cut=0.25):
    """
    Cada segmento (Pk, Pk+1) é substituído por dois pontos a 25% e 75%
    do comprimento. Extremidades preservadas. Em N iterações converge
    para uma B-spline quadrática dos pontos de controle.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    for _ in range(iterations):
        if len(lons) < 3:
            break
        q_lon = lons[:-1] + cut * (lons[1:] - lons[:-1])
        q_lat = lats[:-1] + cut * (lats[1:] - lats[:-1])
        r_lon = lons[:-1] + (1 - cut) * (lons[1:] - lons[:-1])
        r_lat = lats[:-1] + (1 - cut) * (lats[1:] - lats[:-1])
        pairs_lon = np.stack([q_lon, r_lon], axis=1).ravel()
        pairs_lat = np.stack([q_lat, r_lat], axis=1).ravel()
        lons = np.concatenate([[lons[0]], pairs_lon, [lons[-1]]])
        lats = np.concatenate([[lats[0]], pairs_lat, [lats[-1]]])
    return lons, lats


# ============================================================
# NÚCLEO DA CAMINHADA
# ============================================================
def _discover_core(
    df: pd.DataFrame,
    initial_point,
    end_point,
    meters: float,
    angular_samples: int,
    iterations: int,
    increase_meters: float,
    decrease_meters: float,
    max_meters: float,
    loop_close_radius: float,
    loop_min_steps: int,
    arrival_radius_m: float,
    consecutive_fail_stop: int,
    verbose: bool,
    dist_floor: float = 0.0,
    no_revisit: bool = False,
    revisit_penalty: float = 0.0,
    revisit_lookback: int = 4,
    max_turn_deg: float = 180.0,
    turn_penalty: float = 0.0,
    turn_lookback: int = 3,
    turn_max_steps: int = None,
    min_step_factor: float = 1.0,
    guide_scale: float = 1.0,
    init_dir_radius_m: float = None,
    initial_dir=None,
    heading_filter: bool = False,
    heading_lookback: int = 3,
    heading_seed_init: bool = False,
    heading_min_cos_deg: float = 90.0,
    init_cone_deg: float = 90.0,
    init_cone_steps: int = 2,
):
    """
    Executa a caminhada. Retorna (lons, lats, dist_meters, info).

    Componentes do score de cada candidato i:
      v_p    = deslocamento do ponto p
      û_i    = direção do candidato i
      S_i    = [ Σ_{p : cos(v_p,û_i) > 0} 1/d(p,C_i) ] · g_i
      g_i    = 1 + progresso(P) · max(0, cos(ângulo(C_i, P->B)))

    O cosseno cos(v_p,û_i) atua apenas como filtro (descarta pontos que se
    movem no sentido oposto); NÃO multiplica o termo 1/d.

    Melhorias opcionais (default desligado = comportamento baseline):
      no_revisit     : penaliza candidatos que voltam a uma região já
                       percorrida há mais de `revisit_lookback` passos
                       (evita oscilação/travamento no terminal).
      max_turn_deg   : penaliza candidatos que invertem o rumo em relação
                       à média dos últimos `turn_lookback` passos
                       (suaviza o caminho, evita zigzag).
      turn_max_steps : se definido, a penalidade de rumo só vale para os
                       primeiros `turn_max_steps` passos (foco no terminal,
                       sem alterar o meio do trajeto).
    """
    # vetores de deslocamento por ponto (dentro de cada trip)
    if "dx" in df.columns and "dy" in df.columns:
        d_lon = df["dx"].to_numpy(dtype=float)
        d_lat = df["dy"].to_numpy(dtype=float)
    else:
        d_lon = df["lon"].diff().to_numpy()
        d_lat = df["lat"].diff().to_numpy()
    lons_all = df["lon"].to_numpy()
    lats_all = df["lat"].to_numpy()

    meters_rad = meters / M_PER_DEG_LAT
    meters_min_rad = meters_rad
    max_meters_rad = max_meters / M_PER_DEG_LAT
    loop_close_rad = loop_close_radius / M_PER_DEG_LAT
    arrival_rad = arrival_radius_m / M_PER_DEG_LAT

    start = np.array(initial_point, dtype=float)
    pos = start.copy()

    end = None
    axis = None
    axis_norm2 = None
    if end_point is not None:
        end = np.array(end_point, dtype=float)
        axis = end - start
        axis_norm2 = float(np.dot(axis, axis))

    route_lons, route_lats, dist_meters = [], [], []
    cumulative = 0.0

    # memória anti-revisita e de rumo (melhorias opcionais)
    cell_size = max(meters_min_rad * min_step_factor, 1e-9)
    cell_last = {}
    step_hist = []
    heading = None
    h_hist = []
    heading_h = None

    angles = np.linspace(0, 2 * np.pi, angular_samples, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    info = {
        "reached_end": False,
        "closed_loop": False,
        "stop_reason": "max_iterations",
        "n_steps": 0,
    }

    fails = 0
    n = 0

    # arrays de trabalho: pontos ativos compactados
    alive = np.ones(len(df), dtype=bool)
    alive[0] = False
    w_lon = lons_all[alive]
    w_lat = lats_all[alive]
    w_dlon = d_lon[alive]
    w_dlat = d_lat[alive]
    w_norm = np.sqrt(w_dlon ** 2 + w_dlat ** 2)
    w_norm_safe = np.where(w_norm > 0, w_norm, 1e-9)

    # ------------------------------------------------------------
    # DIREÇÃO INICIAL: vetor resultante dos deslocamentos de todos os
    # pontos ativos dentro do raio mínimo.
    #   d0 = (Σ v_j) / |Σ v_j|,  para j com d(x_j, P0) <= r_min
    # Direções opostas se cancelam; a resultante indica o sentido
    # dominante de movimento no terminal.
    # ------------------------------------------------------------
    init_dir = None
    if len(w_lon) >= 3:
        d_start = np.sqrt((w_lon - start[0]) ** 2 + (w_lat - start[1]) ** 2)
        init_rad = (meters_min_rad if init_dir_radius_m is None
                    else init_dir_radius_m / M_PER_DEG_LAT)
        idx_near = np.where(d_start <= init_rad)[0]
        if len(idx_near) >= 3:
            vlon = w_dlon[idx_near]
            vlat = w_dlat[idx_near]
            valid = np.isfinite(vlon) & np.isfinite(vlat)
            if valid.sum() >= 3:
                sx = float(np.sum(vlon[valid]))
                sy = float(np.sum(vlat[valid]))
                norm = np.sqrt(sx ** 2 + sy ** 2)
                if norm > 0:
                    init_dir = np.array([sx / norm, sy / norm])

    # semente do filtro dinâmico: a direção inicial serve de rumo no
    # primeiro passo, antes de qualquer passo aceito existir
    if initial_dir is not None:
        iv = np.asarray(initial_dir, dtype=float)
        ivn = float(np.hypot(iv[0], iv[1]))
        if ivn > 0:
            init_dir = iv / ivn
    if heading_filter and heading_seed_init and init_dir is not None:
        heading_h = init_dir.copy()

    while (n < iterations) and (meters_rad <= max_meters_rad):
        n += 1
        info["n_steps"] = n

        cand_lon = pos[0] + meters_rad * cos_a
        cand_lat = pos[1] + meters_rad * sin_a

        # ---------------- fator de guia ----------------
        if end is not None:
            to_target = end - pos
            t_norm = np.linalg.norm(to_target)
            align = (
                cos_a * to_target[0] / t_norm + sin_a * to_target[1] / t_norm
                if t_norm > 0 else np.ones(angular_samples)
            )
            progress = (
                float(np.clip(np.dot(pos - start, axis) / axis_norm2, 0.0, 1.0))
                if axis_norm2 > 0 else 0.0
            )
            guide = 1.0 + guide_scale * progress * np.maximum(align, 0.0)
        else:
            guide = np.ones(angular_samples)

        # primeiros passos: restringe à direção inicial estimada (cone)
        if len(route_lons) < init_cone_steps and init_dir is not None:
            align_init = cos_a * init_dir[0] + sin_a * init_dir[1]
            lim = math.cos(math.radians(init_cone_deg))
            guide = guide * np.where(align_init > lim, 1.0, 0.0)

        if len(w_lon) == 0:
            info["stop_reason"] = "sem_pontos_ativos"
            break

        # pré-filtro exato: só pontos a <= 2r podem pontuar algum candidato
        d_now = np.sqrt((w_lon - pos[0]) ** 2 + (w_lat - pos[1]) ** 2)
        near = np.where(d_now <= 2.0 * meters_rad)[0]

        best_score = -1.0
        best_point = None
        best_captured = np.array([], dtype=int)

        if len(near) > 0:
            nl, nlt = w_lon[near], w_lat[near]
            ndl, ndt = w_dlon[near], w_dlat[near]
            nns = w_norm_safe[near]

            # compatibilidade do deslocamento do ponto com o rumo atual da
            # caminhada (filtro dinâmico: pontos contra o rumo NÃO contam
            # neste passo, mas permanecem na nuvem — não são apagados)
            ch_vec = None
            if heading_filter and heading_h is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ch_vec = (ndl * heading_h[0] + ndt * heading_h[1]) / nns
                np.nan_to_num(ch_vec, copy=False, nan=0.0, posinf=0.0,
                              neginf=0.0)

            # Avaliação vetorizada: distância e cosseno de cada ponto para
            # cada candidato. Processa os candidatos em blocos para limitar
            # o pico de memória (matrizes n_pontos x bloco).
            n_pts = len(near)
            block = max(1, int(2_000_000 // max(n_pts, 1)))
            for c0 in range(0, angular_samples, block):
                c1 = min(c0 + block, angular_samples)
                sl = slice(c0, c1)
                ca_cos, ca_sin = cos_a[sl], sin_a[sl]
                clon, clat = cand_lon[sl], cand_lat[sl]

                dlon = nl[:, None] - clon[None, :]
                dlat = nlt[:, None] - clat[None, :]
                with np.errstate(invalid="ignore"):
                    dist = np.sqrt(dlon * dlon + dlat * dlat)
                mask = dist <= meters_rad

                with np.errstate(invalid="ignore", divide="ignore"):
                    cos_dir = (ndl[:, None] * ca_cos[None, :]
                               + ndt[:, None] * ca_sin[None, :]) / nns[:, None]
                np.nan_to_num(cos_dir, copy=False, nan=0.0, posinf=0.0,
                              neginf=0.0)

                # filtro direcional: descarta pontos que se movem contra o
                # sentido do caminho (cos < 0). Não pondera pelo cosseno.
                w = mask & (cos_dir > 0.0)
                if heading_filter and ch_vec is not None:
                    lim_h = math.cos(math.radians(heading_min_cos_deg))
                    w = w & (ch_vec[:, None] > lim_h)

                if dist_floor > 0.0:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        inv = np.where(w, 1.0 / (dist + dist_floor), 0.0)
                else:
                    inv = np.zeros_like(dist)
                    np.divide(1.0, dist, out=inv, where=w)
                density = inv.sum(axis=0)
                scores = density * guide[sl]
                np.nan_to_num(scores, copy=False, nan=0.0, posinf=0.0,
                              neginf=0.0)

                # --- melhoria 1: penaliza revisita a regiao ja percorrida ---
                if no_revisit and cell_last:
                    inv_cell = 1.0 / cell_size
                    for j in range(len(scores)):
                        cj = (int(round(float(clon[j]) * inv_cell)),
                              int(round(float(clat[j]) * inv_cell)))
                        last = cell_last.get(cj)
                        if last is not None and (n - last) > revisit_lookback:
                            scores[j] *= revisit_penalty

                # --- melhoria 2: penaliza inversao brusca de rumo ---
                if (heading is not None and turn_penalty < 1.0
                        and (turn_max_steps is None or n <= turn_max_steps)):
                    ht = math.cos(math.radians(max_turn_deg))
                    dots = ca_cos * heading[0] + ca_sin * heading[1]
                    scores[dots < ht] *= turn_penalty

                local_i = int(np.argmax(scores))
                local_score = float(scores[local_i])
                if local_score > best_score:
                    best_score = local_score
                    best_point = (cand_lon[c0 + local_i],
                                  cand_lat[c0 + local_i])
                    best_captured = near[w[:, local_i]]

        if best_score <= 0:
            meters_rad *= increase_meters
            if meters_rad >= max_meters_rad:
                meters_rad = max_meters_rad
                fails += 1
                if fails >= consecutive_fail_stop:
                    info["stop_reason"] = "passos_sem_dados"
                    break
            continue

        fails = 0
        route_lons.append(best_point[0])
        route_lats.append(best_point[1])
        dist_meters.append(meters_rad * M_PER_DEG_LAT)
        cumulative += meters_rad

        keep = np.ones(len(w_lon), dtype=bool)
        keep[best_captured] = False
        w_lon, w_lat = w_lon[keep], w_lat[keep]
        w_dlon, w_dlat = w_dlon[keep], w_dlat[keep]
        w_norm_safe = w_norm_safe[keep]

        prev_pos = pos
        pos = np.array(best_point)

        # atualiza memória anti-revisita (célula = tamanho do raio mínimo)
        cj = (int(round(float(pos[0]) / cell_size)),
              int(round(float(pos[1]) / cell_size)))
        cell_last[cj] = n

        # atualiza rumo recente (média dos últimos passos)
        step_hist.append(pos - prev_pos)
        if len(step_hist) > turn_lookback:
            step_hist.pop(0)
        svec = np.sum(step_hist, axis=0)
        nv = float(np.hypot(svec[0], svec[1]))
        if nv > 0:
            heading = svec / nv

        # rumo usado pelo filtro dinâmico (independente do turn_lookback)
        h_hist.append(pos - prev_pos)
        if len(h_hist) > heading_lookback:
            h_hist.pop(0)
        hsv = np.sum(h_hist, axis=0)
        nhv = float(np.hypot(hsv[0], hsv[1]))
        if nhv > 0:
            heading_h = hsv / nhv

        meters_rad = max(meters_min_rad, meters_rad * decrease_meters)

        # ---------------- critérios de parada ----------------
        if end is not None:
            if np.sqrt((pos[0] - end[0]) ** 2 + (pos[1] - end[1]) ** 2) <= arrival_rad:
                info["reached_end"] = True
                info["stop_reason"] = "chegada_destino"
                break

        if end is None and loop_close_rad > 0 and len(route_lons) >= loop_min_steps:
            if np.sqrt((route_lons[-1] - start[0]) ** 2 +
                       (route_lats[-1] - start[1]) ** 2) <= loop_close_rad:
                info["closed_loop"] = True
                info["stop_reason"] = "loop_fechado"
                break

    if verbose:
        print(f"\n  parada: {info['stop_reason']} | passos={info['n_steps']}")

    return route_lons, route_lats, dist_meters, info


def discover_route(
    df: pd.DataFrame,
    initial_point: list,
    end_point: list = None,
    meters: float = 10.0,
    angular_samples: int = 180,
    iterations: int = 200000,
    increase_meters: float = 1.25,
    decrease_meters: float = None,
    max_meters: float = 250.0,
    loop_close_radius: float = 25.0,
    loop_min_steps: int = 80,
    arrival_radius_m: float = 80.0,
    consecutive_fail_stop: int = 10,
    dist_floor: float = 0.0,
    smooth_iterations: int = 2,
    retry_relaxed: bool = True,
    verbose: bool = False,
    no_revisit: bool = False,
    revisit_penalty: float = 0.0,
    revisit_lookback: int = 4,
    max_turn_deg: float = 180.0,
    turn_penalty: float = 0.0,
    turn_lookback: int = 3,
    turn_max_steps: int = None,
    min_step_factor: float = 1.0,
    guide_scale: float = 1.0,
    init_dir_radius_m: float = None,
    initial_dir=None,
    heading_filter: bool = False,
    heading_lookback: int = 3,
    heading_seed_init: bool = False,
    heading_min_cos_deg: float = 90.0,
    init_cone_deg: float = 90.0,
    init_cone_steps: int = 2,
):
    """
    Descobre a rota entre initial_point e end_point (ou loop circular se
    end_point=None). Retorna (route_df, info).

    Re-tenta com parâmetros relaxados se a rota ficar muito curta, e
    aplica suavização Chaikin no final.

    Melhorias opcionais (ver _discover_core): no_revisit / max_turn_deg.
    """
    if decrease_meters is None:
        decrease_meters = 1.0 / increase_meters

    def run(m, ams, inc, dec, mmax, fail_stop):
        return _discover_core(
            df=df, initial_point=initial_point, end_point=end_point,
            meters=m, angular_samples=ams, iterations=iterations,
            increase_meters=inc, decrease_meters=dec, max_meters=mmax,
            loop_close_radius=loop_close_radius, loop_min_steps=loop_min_steps,
            arrival_radius_m=arrival_radius_m,
            consecutive_fail_stop=fail_stop, dist_floor=dist_floor,
            verbose=verbose,
            no_revisit=no_revisit, revisit_penalty=revisit_penalty,
            revisit_lookback=revisit_lookback, max_turn_deg=max_turn_deg,
            turn_penalty=turn_penalty, turn_lookback=turn_lookback,
            turn_max_steps=turn_max_steps, min_step_factor=min_step_factor,
            guide_scale=guide_scale,
            init_dir_radius_m=init_dir_radius_m,
            heading_filter=heading_filter, heading_lookback=heading_lookback,
            heading_seed_init=heading_seed_init,
            heading_min_cos_deg=heading_min_cos_deg,
            initial_dir=initial_dir,
            init_cone_deg=init_cone_deg, init_cone_steps=init_cone_steps,
        )

    lons, lats, dists, info = run(
        meters, angular_samples, increase_meters, decrease_meters,
        max_meters, consecutive_fail_stop,
    )
    info["retry_used"] = False

    if retry_relaxed and not (info["reached_end"] or info["closed_loop"]) \
            and len(lons) < 30:
        lons2, lats2, dists2, info2 = run(
            meters * 2.0, max(angular_samples, 180), increase_meters * 1.5,
            decrease_meters * 0.85, max_meters * 1.5,
            consecutive_fail_stop * 2,
        )
        info2["retry_used"] = True
        if len(lons2) > len(lons):
            lons, lats, dists, info = lons2, lats2, dists2, info2

    if len(lons) >= 3:
        sl, sa = chaikin_smooth(lons, lats, iterations=smooth_iterations)
        if len(dists) > 0:
            sd = np.interp(
                np.linspace(0, 1, len(sl)),
                np.linspace(0, 1, len(dists)),
                np.asarray(dists, dtype=float),
            )
        else:
            sd = np.zeros(len(sl))
        route = pd.DataFrame({"lon": sl, "lat": sa, "_meters_": sd})
    else:
        route = pd.DataFrame({"lon": lons, "lat": lats, "_meters_": dists})

    return route, info


def _two_points(a, b) -> pd.DataFrame:
    return pd.DataFrame({"lon": [a[0], b[0]], "lat": [a[1], b[1]]})
