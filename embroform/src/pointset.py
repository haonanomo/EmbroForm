from collections import defaultdict, Counter, deque
import numpy as np


def _to2(xy):
    return np.asarray(xy, float).reshape(-1, 2)


def _apply(R, t, xy):
    xy = _to2(xy)
    return (R @ xy.T).T + t


def _k2(p):
    p = np.asarray(p, float).reshape(-1)
    return (float(p[0]), float(p[1]))


def _edge_key(p0, p1):
    a, b = _k2(p0), _k2(p1)
    return (a, b) if a <= b else (b, a)


def point_in_triangle(p, a, b, c, eps=1e-12):
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return (u >= -eps) and (v >= -eps) and (u + v <= 1 + eps)


def inside_patch(pt, faces, verts):
    pt = np.asarray(pt, float)
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        if point_in_triangle(pt, a, b, c):
            return True
    return False


def shift_stitch_points(stitch_points_world, layout, patches, patch_faces, contours, shift_eps=0.002, ward="inward"):
    shifted = []

    for pid, pt_world in stitch_points_world:
        if pid not in layout or pid not in contours:
            continue

        R = layout[pid]["R"]
        t = layout[pid]["t"]
        R_inv = R.T

        verts = patches[pid]
        faces = patch_faces[pid]
        poly = np.asarray(contours[pid], float)

        pt_local = R_inv @ (np.asarray(pt_world, float) - t)

        d = np.linalg.norm(poly - pt_local, axis=1)
        idx = int(np.argmin(d))
        n = len(poly)
        if n < 3:
            continue

        a = poly[(idx - 1) % n]
        b = pt_local
        c = poly[(idx + 1) % n]

        v1 = b - a
        v2 = b - c
        v1n = np.linalg.norm(v1)
        v2n = np.linalg.norm(v2)

        if v1n < 1e-8 and v2n < 1e-8:
            continue
        if v1n < 1e-8:
            dir_vec = v2 / max(v2n, 1e-12)
        elif v2n < 1e-8:
            dir_vec = v1 / max(v1n, 1e-12)
        else:
            dir_vec = v1 / v1n + v2 / v2n

        dn = np.linalg.norm(dir_vec)
        if dn < 1e-8:
            continue
        dir_unit = dir_vec / dn

        p_try = pt_local + shift_eps * dir_unit
        if ward == "inward":
            if not inside_patch(p_try, faces, verts):
                p_try = pt_local - shift_eps * dir_unit
        else:
            if inside_patch(p_try, faces, verts):
                p_try = pt_local - shift_eps * dir_unit

        pt_shift_world = R @ p_try + t
        shifted.append((pid, pt_shift_world))

    return shifted


def turning_angle(p_prev, p_curr, p_next, degrees=True):
    p_prev = np.asarray(p_prev, float)
    p_curr = np.asarray(p_curr, float)
    p_next = np.asarray(p_next, float)

    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    ang = np.arccos(np.clip(c, -1.0, 1.0))
    return float(np.degrees(ang) if degrees else ang)


def discrete_curvature(polyline, offset=3):
    P = _to2(polyline)
    n = len(P)
    out = np.full(n, np.nan, float)
    if n < 2 * offset + 1:
        return out

    for i in range(offset, n - offset):
        p_prev = P[i - offset]
        p_curr = P[i]
        p_next = P[i + offset]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue

        c = float(np.dot(v1, v2) / (n1 * n2))
        ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
        chord = float(np.linalg.norm(p_next - p_prev))
        if chord > 1e-12:
            out[i] = 2.0 * np.sin(ang) / chord

    return out


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent.get(x, x)
        if p != x:
            p = self.find(p)
            self.parent[x] = p
        else:
            self.parent.setdefault(x, x)
        return p

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _build_correspondence_map_from_adj(adj_graph_2d):
    pts = set()
    uf = _UnionFind()

    for (i, j), edge_pairs in adj_graph_2d.items():
        for (e0, e1) in edge_pairs:
            for k in range(2):
                pi = (i, _k2(e0[k]))
                pj = (j, _k2(e1[k]))
                pts.add(pi)
                pts.add(pj)
                uf.union(pi, pj)

    groups = defaultdict(list)
    for p in pts:
        groups[uf.find(p)].append(p)

    corr = defaultdict(set)
    for _, g in groups.items():
        if len(g) <= 1:
            continue
        gs = set(g)
        for p in g:
            corr[p] = gs - {p}

    return corr


def collect_stitch_points(contours_raw, layout, patches, adj_graph_2d, q=90, distance_threshold=None, debug_plot=False):
    curv_all = []
    curv_pid = {}

    for pid, poly in contours_raw.items():
        if poly is None:
            continue
        poly = np.asarray(poly, float)
        if len(poly) < 3:
            continue
        curv = discrete_curvature(poly, offset=3)
        curv_pid[pid] = curv
        curv_all.extend([c for c in curv if not np.isnan(c)])

    if not curv_all:
        return [], {}

    curv_all = np.asarray(curv_all, float)
    curv_min = float(np.percentile(curv_all, q))

    if debug_plot:
        import matplotlib.pyplot as plt
        plt.hist(curv_all, bins=100)
        plt.title("Curvature Distribution")
        plt.show()

    cand = {}
    for pid, poly in contours_raw.items():
        poly = np.asarray(poly, float)
        if pid not in curv_pid:
            continue
        curv = curv_pid[pid]
        for k in range(len(poly)):
            v = curv[k]
            if np.isnan(v):
                continue
            if v >= curv_min:
                cand[(pid, _k2(poly[k]))] = float(v)

    corr_local = _build_correspondence_map_from_adj(adj_graph_2d)

    ext = dict(cand)
    for k in list(cand.keys()):
        for c in corr_local.get(k, set()):
            if c not in ext:
                ext[c] = cand[k]

    final_local = set()
    used = set()
    disabled = set()
    MAX_GAP = 3

    for pid, poly in sorted(contours_raw.items(), key=lambda x: x[0]):
        poly = np.asarray(poly, float)
        n = len(poly)
        if n < 3:
            continue

        k = 0
        while k < n:
            group = []
            t = k
            gaps = MAX_GAP
            saw = False

            while t < n:
                key = (pid, _k2(poly[t]))
                ok = (key in ext) and (key not in used)
                if ok:
                    group.append(key)
                    saw = True
                    gaps = MAX_GAP
                    t += 1
                else:
                    if gaps > 0:
                        gaps -= 1
                        t += 1
                    else:
                        break

            if not saw:
                k += 1
                continue

            multi = []
            for gk in group:
                if len(set(corr_local.get(gk, set()))) >= 2:
                    multi.append(gk)

            if multi:
                keep = set()
                for mk in multi:
                    keep.add(mk)
                    keep |= set(corr_local.get(mk, set()))
                final_local |= keep
                used |= set(group)
                dis = set()
                for gk in group:
                    dis |= set(corr_local.get(gk, set()))
                disabled |= dis
                k = t
                continue

            corr_set = set(group)
            for gk in group:
                corr_set |= set(corr_local.get(gk, set()))
            corr_set -= disabled
            if not corr_set:
                k = t
                continue

            best = max(corr_set, key=lambda x: ext.get(x, 0.0))
            keep = {best} | set(corr_local.get(best, set()))
            final_local |= keep
            used |= corr_set
            k = t

    if distance_threshold is not None:
        loc2world = {}
        for pid, pt in final_local:
            if pid not in layout:
                continue
            R, t = layout[pid]["R"], layout[pid]["t"]
            w = R @ np.asarray(pt, float) + t
            loc2world[(pid, pt)] = np.asarray(w, float)
        uf2 = _UnionFind()
        for p in final_local:
            for q2 in corr_local.get(p, set()):
                if q2 in final_local:
                    uf2.union(p, q2)
        comp = defaultdict(list)
        for p in final_local:
            comp[uf2.find(p)].append(p)

        merged = []
        for _, g in comp.items():
            if len(g) < 2:
                continue
            merged.append(g)

        kept = set()
        for g in merged:
            chosen = []
            for p in g:
                wp = loc2world.get(p, None)
                if wp is None:
                    continue
                ok = True
                for q2 in chosen:
                    wq = loc2world.get(q2, None)
                    if wq is None:
                        continue
                    if np.linalg.norm(wp - wq) < distance_threshold:
                        ok = False
                        break
                if ok:
                    chosen.append(p)

            if len(chosen) >= 2:
                kept |= set(chosen)

        final_local = kept
        corr2 = defaultdict(set)
        for p in final_local:
            for q2 in corr_local.get(p, set()):
                if q2 in final_local:
                    corr2[p].add(q2)
        corr_local = corr2


    if distance_threshold is not None:
        final_local, corr_local = global_dedupe_across_components(
            final_local, corr_local, layout,
            dist_th=distance_threshold,
            score=ext,      
            min_keep_multi=2
        )


    
    stitch_world = []
    for pid, pt in final_local:
        if pid not in layout:
            continue
        R, t = layout[pid]["R"], layout[pid]["t"]
        w = R @ np.asarray(pt, float) + t
        stitch_world.append((pid, (float(w[0]), float(w[1]))))

    corr_world = {}
    for p, qs in corr_local.items():
        pid, pt = p
        if pid not in layout:
            continue
        R, t = layout[pid]["R"], layout[pid]["t"]
        w = R @ np.asarray(pt, float) + t
        wk = (pid, (float(w[0]), float(w[1])))

        out = []
        for q2 in qs:
            qpid, qpt = q2
            if qpid not in layout:
                continue
            R2, t2 = layout[qpid]["R"], layout[qpid]["t"]
            w2 = R2 @ np.asarray(qpt, float) + t2
            out.append((qpid, (float(w2[0]), float(w2[1]))))
        corr_world[wk] = out

    return stitch_world, corr_world


def segments_to_polylines(segments):
    if not segments:
        return []

    adj = defaultdict(list)
    undirected = set()

    for p0, p1 in segments:
        a, b = _k2(p0), _k2(p1)
        if a == b:
            continue
        key = (a, b) if a <= b else (b, a)
        if key in undirected:
            continue
        undirected.add(key)
        adj[a].append(b)
        adj[b].append(a)

    nodes = list(adj.keys())
    seen = set()
    polylines = []

    def trace(comp_nodes):
        deg = {v: len(adj[v]) for v in comp_nodes}
        ends = [v for v, d in deg.items() if d == 1]
        start = ends[0] if ends else next(iter(comp_nodes))

        prev = None
        cur = start
        used = set([start])
        path = [start]

        while True:
            nxts = [n for n in adj[cur] if n != prev]
            nxt = None
            for n in nxts:
                if n not in used:
                    nxt = n
                    break
            if nxt is None:
                if all(d == 2 for d in deg.values()):
                    if path[0] != path[-1]:
                        path.append(path[0])
                break
            path.append(nxt)
            used.add(nxt)
            prev, cur = cur, nxt

        return np.asarray(path, float)

    for v in nodes:
        if v in seen:
            continue
        q = deque([v])
        comp = {v}
        seen.add(v)
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    comp.add(w)
                    q.append(w)
        polylines.append(trace(comp))

    return polylines


def drop_edges_from_contour(poly, drop_keys):
    P = _to2(poly)
    if len(P) < 2:
        return []
    segs = []
    for i in range(len(P) - 1):
        p0, p1 = P[i], P[i + 1]
        if _edge_key(p0, p1) not in drop_keys:
            segs.append((p0, p1))
    return segments_to_polylines(segs)


def build_drop_keys(best_choice, adj_graph_2d):
    drop = set()
    for (i, j), idx in best_choice.items():
        pair = (i, j) if (i, j) in adj_graph_2d else (j, i)
        reverse = pair != (i, j)
        edge_pairs = adj_graph_2d[pair]
        if idx < 0 or idx >= len(edge_pairs):
            raise IndexError(f"best_choice[{(i, j)}]={idx} out of range for adj_graph_2d[{pair}]")
        (Ai, Bi), (Cj, Dj) = edge_pairs[idx]
        if reverse:
            (Ai, Bi), (Cj, Dj) = (Cj, Dj), (Ai, Bi)
        drop.add(_edge_key(Ai, Bi))
        drop.add(_edge_key(Cj, Dj))
    return drop


def build_contours_pieces(contours_raw, best_choice, adj_graph_2d):
    drop_keys = build_drop_keys(best_choice, adj_graph_2d)
    out = {}

    for pid, raw in contours_raw.items():
        pieces = []
        if isinstance(raw, (list, tuple)):
            for poly in raw:
                pieces.extend(drop_edges_from_contour(np.asarray(poly, float), drop_keys))
        else:
            pieces.extend(drop_edges_from_contour(np.asarray(raw, float), drop_keys))
        out[pid] = pieces

    return out


def merge_all_contours(contours_world):
    all_polys = []
    all_meta = []

    for pid, poly_list in contours_world.items():
        for poly in poly_list:
            poly = _to2(poly)
            if len(poly) < 2:
                continue
            all_polys.append(deque([p for p in poly]))
            all_meta.append(deque([(pid, (float(p[0]), float(p[1]))) for p in poly]))

    if not all_polys:
        raise ValueError("merge_all_contours: empty contours_world")

    cur = all_polys.pop(0)
    curm = all_meta.pop(0)

    while all_polys:
        best_i = -1
        best_case = None
        best_d = float("inf")

        for i, p in enumerate(all_polys):
            p0 = p[0]
            p1 = p[-1]
            d = {
                "end_to_start": np.linalg.norm(np.asarray(cur[-1]) - np.asarray(p0)),
                "end_to_end": np.linalg.norm(np.asarray(cur[-1]) - np.asarray(p1)),
                "start_to_start": np.linalg.norm(np.asarray(cur[0]) - np.asarray(p0)),
                "start_to_end": np.linalg.norm(np.asarray(cur[0]) - np.asarray(p1)),
            }
            case, dist = min(d.items(), key=lambda x: x[1])
            if dist < best_d:
                best_d = dist
                best_i = i
                best_case = case

        p = all_polys.pop(best_i)
        m = all_meta.pop(best_i)
        pl = list(p)
        ml = list(m)

        if best_case == "end_to_start":
            cur.extend(pl)
            curm.extend(ml)
        elif best_case == "end_to_end":
            cur.extend(reversed(pl))
            curm.extend(reversed(ml))
        elif best_case == "start_to_start":
            cur = deque(list(reversed(pl)) + list(cur))
            curm = deque(list(reversed(ml)) + list(curm))
        else:
            cur = deque(pl + list(cur))
            curm = deque(ml + list(curm))

    merged_poly = np.asarray(list(cur), float)
    merged_meta = list(curm)
    return merged_poly, merged_meta


def outline_euler(best_layout, patches, patch_faces, adj_graph_2d, best_choice, min_seg=0.0):
    world_pt_map = {}
    edge_owner_pid = {}
    vertex_owner_pid = defaultdict(set)

    boundary_edges = set()
    stitched_edges = set()
    bridge_edges = set()
    bridge_vertex_pid = defaultdict(dict)

    for pid, F in patch_faces.items():
        if pid not in patches or pid not in best_layout:
            continue
        V = _to2(patches[pid])
        F = np.asarray(F, int)
        if V.size == 0 or F.size == 0:
            continue

        cnt = Counter()
        for a, b, c in F:
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v:
                    u, v = v, u
                cnt[(u, v)] += 1
        outer = [(u, v) for (u, v), c in cnt.items() if c == 1]

        R, t = best_layout[pid]["R"], best_layout[pid]["t"]
        for u, v in outer:
            p0, p1 = _apply(R, t, [V[u], V[v]])
            if np.linalg.norm(p1 - p0) <= min_seg:
                continue
            a = (float(p0[0]), float(p0[1]))
            b = (float(p1[0]), float(p1[1]))
            boundary_edges.add(frozenset([a, b]))
            world_pt_map[a] = p0
            world_pt_map[b] = p1
            vertex_owner_pid[a].add(pid)
            vertex_owner_pid[b].add(pid)
            edge_owner_pid[frozenset([a, b])] = pid

    def add_bridge(ptA, pidA, ptB, pidB):
        if np.linalg.norm(ptA - ptB) <= min_seg:
            return
        a = (float(ptA[0]), float(ptA[1]))
        b = (float(ptB[0]), float(ptB[1]))
        world_pt_map[a] = ptA
        world_pt_map[b] = ptB
        vertex_owner_pid[a].add(pidA)
        vertex_owner_pid[b].add(pidB)
        key = frozenset([a, b])
        bridge_edges.add(key)
        bridge_vertex_pid[key][a] = pidA
        bridge_vertex_pid[key][b] = pidB

    for key, idx in best_choice.items():
        pairs = adj_graph_2d.get(key)
        if pairs is None:
            rkey = (key[1], key[0])
            pairs = adj_graph_2d.get(rkey)
            if pairs is None:
                continue
            i, j = rkey
        else:
            i, j = key

        if i not in best_layout or j not in best_layout:
            continue
        if not pairs or idx >= len(pairs):
            continue

        eA, eB = pairs[idx]
        Ai = _apply(best_layout[i]["R"], best_layout[i]["t"], eA)
        Bj = _apply(best_layout[j]["R"], best_layout[j]["t"], eB)

        a0 = (float(Ai[0][0]), float(Ai[0][1]))
        a1 = (float(Ai[-1][0]), float(Ai[-1][1]))
        b0 = (float(Bj[0][0]), float(Bj[0][1]))
        b1 = (float(Bj[-1][0]), float(Bj[-1][1]))

        stitched_edges.add(frozenset([a0, a1]))
        stitched_edges.add(frozenset([b0, b1]))

        add_bridge(Ai[0], i, Bj[0], j)
        add_bridge(Ai[-1], i, Bj[-1], j)

    remain = set(boundary_edges) | set(bridge_edges)
    remain -= set(stitched_edges)

    adj = defaultdict(list)
    for e in remain:
        a, b = list(e)
        adj[a].append(b)
        adj[b].append(a)

    bad = [v for v, ns in adj.items() if len(ns) != 2]
    if bad:
        raise ValueError(f"outline_euler: Euler failed (deg!=2) nodes={len(bad)}")

    start = next(iter(adj))
    cur = start
    prev = None
    used = set()
    chain = []
    last_pid = None

    while True:
        pid_here = None
        probe = frozenset([cur, adj[cur][0]])
        if probe in edge_owner_pid:
            pid_here = edge_owner_pid[probe]
        elif probe in bridge_vertex_pid:
            pid_here = bridge_vertex_pid[probe].get(cur, None)

        if pid_here is None and vertex_owner_pid[cur]:
            if len(vertex_owner_pid[cur]) == 1:
                pid_here = next(iter(vertex_owner_pid[cur]))
            else:
                pid_here = last_pid if last_pid in vertex_owner_pid[cur] else next(iter(vertex_owner_pid[cur]))

        chain.append((pid_here if pid_here is not None else -1, cur))
        last_pid = pid_here

        nxt = None
        for n in adj[cur]:
            e = frozenset([cur, n])
            if e not in used:
                used.add(e)
                nxt = n
                break

        if nxt is None:
            break
        prev, cur = cur, nxt
        if cur == start:
            break

    return chain


def adj_to_world(adj_graph_2d, layout):
    out = {}
    for (i, j), edge_list in adj_graph_2d.items():
        if i not in layout or j not in layout:
            continue
        Ri, ti = layout[i]["R"], layout[i]["t"]
        Rj, tj = layout[j]["R"], layout[j]["t"]
        tmp = []
        for (e0, e1) in edge_list:
            pi0 = tuple((Ri @ np.asarray(e0[0], float) + ti).tolist())
            pi1 = tuple((Ri @ np.asarray(e0[1], float) + ti).tolist())
            pj0 = tuple((Rj @ np.asarray(e1[0], float) + tj).tolist())
            pj1 = tuple((Rj @ np.asarray(e1[1], float) + tj).tolist())
            tmp.append(((pi0, pi1), (pj0, pj1)))
        out[(i, j)] = tmp
    return out


def add_midpoints_sparse(merged_contour, stitch_points, corr_map, adj_graph_2d_world,
                         min_gap=20, max_gap=40, max_insertions=2, close_gap=15,
                         tol=1e-9, distance_threshold=None):

    def norm_item(p):
        pid, pt = p
        pt = np.asarray(pt, float).reshape(-1)[:2]
        return (pid, (float(pt[0]), float(pt[1])))

    merged_contour = [norm_item(p) for p in merged_contour]
    stitch_set = set(norm_item(p) for p in stitch_points)

    contour_index = {}
    for i, it in enumerate(merged_contour):
        contour_index[it] = i

    insert_spans = []
    start = -1
    cur_pid = None

    for i, (pid, pt) in enumerate(merged_contour):
        if pid != cur_pid:
            if start != -1 and i - start >= min_gap:
                nins = 1 if i - start <= max_gap else 2
                insert_spans.append((start, i, min(max_insertions, nins), cur_pid))
            start = -1
            cur_pid = pid

        if (pid, pt) in stitch_set:
            if start != -1 and i - start >= min_gap:
                nins = 1 if i - start <= max_gap else 2
                insert_spans.append((start, i, min(max_insertions, nins), cur_pid))
            start = -1
        elif start == -1:
            start = i

    if start != -1 and len(merged_contour) - start >= min_gap:
        nins = 1 if len(merged_contour) - start <= max_gap else 2
        insert_spans.append((start, len(merged_contour), min(max_insertions, nins), cur_pid))

    new_points = set()
    for s, e, nins, pid in insert_spans:
        for j in range(1, nins + 1):
            idx = s + (e - s) * j // (nins + 1)
            p = merged_contour[idx]
            if p not in stitch_set:
                new_points.add(p)

    if not new_points:
        return list(stitch_set), defaultdict(set, {k: set(v) for k, v in corr_map.items()})

    def on_seg(p, a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        p = np.asarray(p, float)
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 <= tol:
            dist = float(np.linalg.norm(p - a))
            return (dist <= tol, 0.0, dist)
        t = float(np.dot(p - a, ab) / L2)
        if t < -tol or t > 1 + tol:
            return (False, None, None)
        proj = a + t * ab
        dist = float(np.linalg.norm(proj - p))
        if dist <= tol:
            return (True, max(0.0, min(1.0, t)), dist)
        return (False, None, None)
    
    def qkey(pt, q=1e-9):
        pt = np.asarray(pt, float).reshape(-1)[:2]
        if not np.all(np.isfinite(pt)):
            return None
        v = pt / q
        if not np.all(np.isfinite(v)):
            return None
        return tuple(np.round(v).astype(np.int64))


    new_lookup = defaultdict(dict)
    for pid, pt in new_points:
        new_lookup[pid][qkey(pt)] = pt

    matched_pairs = defaultdict(set)
    matched_points = set()

    for (i, j), edges in adj_graph_2d_world.items():
        for (ei, ej) in edges:
            ai = np.asarray(ei[0], float); bi = np.asarray(ei[1], float)
            aj = np.asarray(ej[0], float); bj = np.asarray(ej[1], float)

            for (pid, pt) in list(new_points):
                if pid == i:
                    ok, t, _ = on_seg(pt, ai, bi)
                    if ok:
                        pj = aj + t * (bj - aj)
                        k = qkey(pj)
                        if k in new_lookup[j]:
                            qpt = new_lookup[j][k]
                            pkey = (i, pt)
                            qkey2 = (j, qpt)
                            matched_pairs[pkey].add(qkey2)
                            matched_pairs[qkey2].add(pkey)
                            matched_points.update([pkey, qkey2])
                elif pid == j:
                    ok, t, _ = on_seg(pt, aj, bj)
                    if ok:
                        pi = ai + t * (bi - ai)
                        k = qkey(pi)
                        if k in new_lookup[i]:
                            ppt = new_lookup[i][k]
                            pkey = (i, ppt)
                            qkey2 = (j, pt)
                            matched_pairs[pkey].add(qkey2)
                            matched_pairs[qkey2].add(pkey)
                            matched_points.update([pkey, qkey2])

    orphans = [p for p in new_points if p not in matched_points]

    updated_set = set(stitch_set)
    updated_corr = defaultdict(set, {k: set(v) for k, v in corr_map.items()})
    newly_added = set()
    added_pairs = []

    for p in matched_points:
        if p not in updated_set:
            newly_added.add(p)
        updated_set.add(p)

    for p, qs in matched_pairs.items():
        for q2 in qs:
            if p == q2:
                continue
            if q2 not in updated_corr[p]:
                updated_corr[p].add(q2)
                updated_corr[q2].add(p)
                added_pairs.append((p, q2))

    def try_pair_orphan(orphan):
        pid_o, pt_o = orphan
        best = None
        best_t = None

        for (i, j), edges in adj_graph_2d_world.items():
            for (ei, ej) in edges:
                ai = np.asarray(ei[0], float); bi = np.asarray(ei[1], float)
                aj = np.asarray(ej[0], float); bj = np.asarray(ej[1], float)

                if pid_o == i:
                    ok, t, dist = on_seg(pt_o, ai, bi)
                    if ok and (best is None or dist < best[0]):
                        best = (dist, (i, j), (ai, bi, aj, bj), "i"); best_t = t
                elif pid_o == j:
                    ok, t, dist = on_seg(pt_o, aj, bj)
                    if ok and (best is None or dist < best[0]):
                        best = (dist, (i, j), (ai, bi, aj, bj), "j"); best_t = t

        if best is None:
            # print(f"[add_midpoints_sparse] failed, skip: pid={pid_o}, pt={pt_o}")
            return

        _, (i, j), (ai, bi, aj, bj), side = best
        t = best_t

        if side == "i":
            pj = aj + t * (bj - aj)
            target = (j, (float(pj[0]), float(pj[1])))
        else:
            pi = ai + t * (bi - ai)
            target = (i, (float(pi[0]), float(pi[1])))

        okey = (pid_o, pt_o)

        tpid, tpt = target
        tk = qkey(tpt)
        aligned = None

        if tk in new_lookup[tpid]:
            aligned = (tpid, new_lookup[tpid][tk])
        else:
            for (pid_s, pt_s) in updated_set:
                if pid_s == tpid and qkey(pt_s) == tk:
                    aligned = (pid_s, pt_s)
                    break
        if aligned is None:
            aligned = target

        if okey not in updated_set:
            newly_added.add(okey)
            updated_set.add(okey)
        if aligned not in updated_set:
            newly_added.add(aligned)
            updated_set.add(aligned)

        updated_corr[okey].add(aligned)
        updated_corr[aligned].add(okey)
        added_pairs.append((okey, aligned))

    for o in orphans:
        try_pair_orphan(o)

    def close_same_pid(u, v, gap=close_gap):
        if u[0] != v[0]:
            return False
        iu = contour_index.get(u, None)
        iv = contour_index.get(v, None)
        if iu is None or iv is None:
            return False
        return abs(iu - iv) <= gap

    to_remove = set()
    for i in range(len(added_pairs)):
        A, B = added_pairs[i]
        for j in range(i + 1, len(added_pairs)):
            C, D = added_pairs[j]
            if (close_same_pid(A, C) or close_same_pid(A, D) or close_same_pid(B, C) or close_same_pid(B, D)):
                to_remove.add((C, D))

    for p, q2 in to_remove:
        updated_corr[p].discard(q2)
        updated_corr[q2].discard(p)

    def maybe_drop(pt):
        if pt in newly_added and len(updated_corr.get(pt, set())) == 0:
            updated_corr.pop(pt, None)
            updated_set.discard(pt)

    for p, q2 in to_remove:
        maybe_drop(p)
        maybe_drop(q2)

    if distance_threshold is not None:
        pts = list(updated_set)
        keep = set()
        seen = set()

        for p in pts:
            if p in seen:
                continue
            grp = set(updated_corr.get(p, set())) | {p}
            seen |= grp
            if len(grp) <= 1:
                continue

            grp_list = list(grp)
            chosen = []
            for x in grp_list:
                wx = np.asarray(x[1], float)
                ok = True
                for y in chosen:
                    wy = np.asarray(y[1], float)
                    if np.linalg.norm(wx - wy) < distance_threshold:
                        ok = False
                        break
                if ok:
                    chosen.append(x)
            if len(chosen) >= 2:
                keep |= set(chosen)

        updated_set = keep
        updated_corr = defaultdict(set, {p: {q2 for q2 in updated_corr.get(p, set()) if q2 in keep} for p in keep})

    return list(updated_set), updated_corr


def export_svg_contour_and_stitches(merged_contour, stitch_points=None, svg_filename="layout.svg",
                                   target_width_mm=410.0, target_height_mm=200.0, padding_mm=5.0,
                                   contour_color="#000000", contour_stroke_mm=0.2, close_contour=False,
                                   stitch_fill="#ff0000", stitch_radius_mm=0.5,
                                   add_border=True, border_color="blue", border_stroke_mm=0.01):
    all_pts = []
    if merged_contour:
        all_pts += [tuple(pt) for _, pt in merged_contour]
    if stitch_points:
        all_pts += [tuple(pt) for _, pt in stitch_points]
    if not all_pts:
        raise ValueError("export_svg_contour_and_stitches: empty input")

    pts = np.asarray(all_pts, float)
    xmin, ymin = float(pts[:, 0].min()), float(pts[:, 1].min())
    xmax, ymax = float(pts[:, 0].max()), float(pts[:, 1].max())

    w = max(xmax - xmin, 1e-12)
    h = max(ymax - ymin, 1e-12)
    avail_w = max(target_width_mm - 2 * padding_mm, 1e-6)
    avail_h = max(target_height_mm - 2 * padding_mm, 1e-6)
    s = min(avail_w / w, avail_h / h)

    def world_to_mm(p):
        x, y = float(p[0]), float(p[1])
        X = padding_mm + (x - xmin) * s
        Y = padding_mm + (y - ymin) * s
        Y = target_height_mm - Y
        return X, Y

    contour_path = ""
    if merged_contour and len(merged_contour) >= 2:
        xy = [world_to_mm(pt) for _, pt in merged_contour]
        cmds = [f"M {xy[0][0]:.6f} {xy[0][1]:.6f}"] + [f"L {x:.6f} {y:.6f}" for (x, y) in xy[1:]]
        if close_contour:
            cmds.append("Z")
        contour_path = " ".join(cmds)

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{target_width_mm}mm" height="{target_height_mm}mm" '
        f'viewBox="0 0 {target_width_mm} {target_height_mm}">'
    )

    if add_border:
        svg.append(
            f'<rect x="0" y="0" width="{target_width_mm}" height="{target_height_mm}" '
            f'fill="none" stroke="{border_color}" stroke-width="{border_stroke_mm}mm" />'
        )

    if contour_path:
        svg.append(
            f'<path d="{contour_path}" stroke="{contour_color}" stroke-width="{contour_stroke_mm}mm" '
            f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    if stitch_points:
        for _, pt in stitch_points:
            cx, cy = world_to_mm(pt)
            svg.append(f'<circle cx="{cx:.6f}" cy="{cy:.6f}" r="{stitch_radius_mm}mm" fill="{stitch_fill}" />')

    svg.append("</svg>")

    with open(svg_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    return svg_filename


def export_svg_with_stitch_vectors_mm(
        merged_contour,
        stitch_points_world,
        shifted_in_stitch_points,
        shifted_out_stitch_points,
        svg_filename="pointset.svg",
        target_width_mm=130.0,
        target_height_mm=100.0,
        padding_mm=5.0,
        stroke_width_pt="0.1pt",
        contour_color="#000000",
        stitch_fill="#ff0000",
        stitch_radius_world=0.5,
        vector_color="#0000ff",
        vector_stroke_mm=0.05,
        close_contour=False,
):

    all_pts = []
    if merged_contour:
        all_pts += [tuple(pt) for _, pt in merged_contour]
    if stitch_points_world:
        all_pts += [tuple(pt) for _, pt in stitch_points_world]
    if shifted_in_stitch_points:
        all_pts += [tuple(pt) for _, pt in shifted_in_stitch_points]
    if shifted_out_stitch_points:
        all_pts += [tuple(pt) for _, pt in shifted_out_stitch_points]
    if not all_pts:
        raise ValueError("export_svg_with_stitch_vectors_mm: empty input")

    pts = np.asarray(all_pts, float)
    xmin, ymin = float(pts[:, 0].min()), float(pts[:, 1].min())
    xmax, ymax = float(pts[:, 0].max()), float(pts[:, 1].max())

    w = max(xmax - xmin, 1e-12)
    h = max(ymax - ymin, 1e-12)
    avail_w = max(target_width_mm - 2 * padding_mm, 1e-6)
    avail_h = max(target_height_mm - 2 * padding_mm, 1e-6)
    s = min(avail_w / w, avail_h / h)

    stitch_radius_mm = stitch_radius_world


    def world_to_mm(p):
        x, y = float(p[0]), float(p[1])
        X = padding_mm + (x - xmin) * s
        Y = padding_mm + (y - ymin) * s
        Y = target_height_mm - Y
        return X, Y

    def contour_path():
        if not merged_contour or len(merged_contour) < 2:
            return ""
        xy = [world_to_mm(pt) for _, pt in merged_contour]
        cmds = [f"M {xy[0][0]:.6f} {xy[0][1]:.6f}"] + [f"L {x:.6f} {y:.6f}" for (x, y) in xy[1:]]
        if close_contour:
            cmds.append("Z")
        return " ".join(cmds)

    path_d = contour_path()

    def build_svg(include_vectors: bool, filename: str):
        svg = []
        svg.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{target_width_mm}mm" height="{target_height_mm}mm" '
            f'viewBox="0 0 {target_width_mm} {target_height_mm}">'
        )

        if path_d:
            svg.append(
                f'<path d="{path_d}" stroke="{contour_color}" stroke-width="{vector_stroke_mm}mm" '
                f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            )

        if stitch_points_world:
            for _, pt in stitch_points_world:
                cx, cy = world_to_mm(pt)
                svg.append(f'<circle cx="{cx:.6f}" cy="{cy:.6f}" r="{stitch_radius_mm*10}mm" fill="{stitch_fill}" />')

        if include_vectors and stitch_points_world and shifted_in_stitch_points and shifted_out_stitch_points:
            for (_, pout), (_, pin) in zip(shifted_out_stitch_points, shifted_in_stitch_points):
                x0, y0 = world_to_mm(pout)
                x1, y1 = world_to_mm(pin)
                svg.append(
                    f'<line x1="{x0:.6f}" y1="{y0:.6f}" x2="{x1:.6f}" y2="{y1:.6f}" '
                    f'stroke="{vector_color}" stroke-width="{vector_stroke_mm}mm" stroke-linecap="round" />'
                )

        svg.append("</svg>")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(svg))
        return filename
    
    build_svg(include_vectors=True, filename=svg_filename)
    return svg_filename


# def export_pointset_debug_png(
#         merged_contour,
#         stitch_points_world,
#         shifted_in_stitch_points=None,
#         shifted_out_stitch_points=None,
#         png_filename="pointset_debug.png",
#         dpi=200,
#         show_vectors=True,
# ):
#     """
#     py 项目里不做交互式可视化，直接把 pointset debug 图保存成 png。
#     """
#     import matplotlib.pyplot as plt

#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     if merged_contour and len(merged_contour) >= 2:
#         C = np.asarray([pt for _, pt in merged_contour], float)
#         ax.plot(C[:, 0], C[:, 1])

#     if stitch_points_world:
#         P = np.asarray([pt for _, pt in stitch_points_world], float)
#         ax.scatter(P[:, 0], P[:, 1], s=8)

#     if show_vectors and shifted_in_stitch_points and shifted_out_stitch_points:
#         for (_, pout), (_, pin) in zip(shifted_out_stitch_points, shifted_in_stitch_points):
#             ax.plot([pout[0], pin[0]], [pout[1], pin[1]])

#     ax.set_aspect("equal", adjustable="box")
#     ax.axis("off")
#     fig.savefig(png_filename, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
#     plt.close(fig)
#     return png_filename


def export_dxf(layout, outlines, stitch_points, radius, filename="layout_with_stitch.dxf", scale=1.0):
    import ezdxf

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    for pid, polyline_list in outlines.items():
        if pid not in layout:
            continue
        R, t = layout[pid]["R"], layout[pid]["t"]
        for poly in polyline_list:
            poly = _to2(poly)
            W = _apply(R, t, poly) * float(scale)
            pts = [tuple(p) for p in W]
            is_closed = np.linalg.norm(W[0] - W[-1]) < 1e-8
            msp.add_lwpolyline(pts, close=is_closed, dxfattribs={"color": 1})

    for _, pt in stitch_points:
        msp.add_circle(center=(pt[0] * scale, pt[1] * scale), radius=radius, dxfattribs={"color": 1})

    doc.saveas(filename)
    return filename


def global_dedupe_across_components(final_local, corr_local, layout, dist_th, score, min_keep_multi=2):

    alive = set()
    pos = {}
    for p in final_local:
        pid, pt = p
        if pid not in layout:
            continue
        R, t = layout[pid]["R"], layout[pid]["t"]
        w = R @ np.asarray(pt, float) + t
        alive.add(p)
        pos[p] = np.asarray(w, float)

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in list(alive):
        for q in corr_local.get(p, ()):
            if q in alive:
                union(p, q)

    comp = defaultdict(list)
    for p in alive:
        comp[find(p)].append(p)

    comps = [set(v) for v in comp.values()] 

    comp_is_multi = []
    for g in comps:
        is_multi = False
        for p in g:
            if len(corr_local.get(p, ())) >= 2:
                is_multi = True
                break
        comp_is_multi.append(is_multi)

    def closure_of_point(p):
        rem = {p}
        for q in corr_local.get(p, ()):
            if q in alive:
                rem.add(q)
        return rem
    
    def find_colliding_points(A, B):
        collA = set()
        collB = set()
        for p in A:
            wp = pos.get(p, None)
            if wp is None:
                continue
            for q in B:
                wq = pos.get(q, None)
                if wq is None:
                    continue
                if np.linalg.norm(wp - wq) < dist_th:
                    collA.add(p)
                    collB.add(q)
        return collA, collB

    def try_remove_one_pair_to_resolve(self_comp, other_comp, self_is_multi):
        coll_self, _ = find_colliding_points(self_comp, other_comp)
        if not coll_self:
            return None  

        best_rem = None
        best_cost = None

        for p in coll_self:
            rem = closure_of_point(p)
            remaining = [x for x in self_comp if x not in rem]
            if self_is_multi:
                if len(remaining) < min_keep_multi:
                    continue
            else:
                if len(remaining) < 1:
                    continue

            still = False
            for x in remaining:
                wx = pos.get(x, None)
                if wx is None:
                    continue
                for y in other_comp:
                    wy = pos.get(y, None)
                    if wy is None:
                        continue
                    if np.linalg.norm(wx - wy) < dist_th:
                        still = True
                        break
                if still:
                    break
            if still:
                continue

            cost = 0.0
            for r in rem:
                cost += float(score.get(r, 0.0)) 
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_rem = rem

        return best_rem

    changed = True
    while changed:
        changed = False

        for i in range(len(comps)):
            comps[i] = {p for p in comps[i] if p in alive}

        active = [(i, comps[i]) for i in range(len(comps)) if len(comps[i]) > 0]
        if len(active) <= 1:
            break

        did_any = False

        for ai in range(len(active)):
            i, A = active[ai]
            for bi in range(ai + 1, len(active)):
                j, B = active[bi]

                collA, collB = find_colliding_points(A, B)
                if not collA:
                    continue 

                A_multi = comp_is_multi[i]
                B_multi = comp_is_multi[j]

                if A_multi != B_multi:
                    if A_multi and (not B_multi):
                        rem = try_remove_one_pair_to_resolve(B, A, self_is_multi=False)
                        if rem is None:
                            rem = set(B)
                        alive -= rem
                        did_any = True
                        changed = True
                        break
                    if B_multi and (not A_multi):
                        rem = try_remove_one_pair_to_resolve(A, B, self_is_multi=False)
                        if rem is None:
                            rem = set(A)
                        alive -= rem
                        did_any = True
                        changed = True
                        break
                else:
                    if A_multi and B_multi:
                        remA = try_remove_one_pair_to_resolve(A, B, self_is_multi=True)
                        remB = try_remove_one_pair_to_resolve(B, A, self_is_multi=True)

                        if remA is not None or remB is not None:
                            def rem_cost(remset):
                                if remset is None:
                                    return (10**9, 10**18)
                                c = 0.0
                                for r in remset:
                                    c += float(score.get(r, 0.0))
                                return (len(remset), c)

                            if rem_cost(remA) <= rem_cost(remB):
                                alive -= remA
                            else:
                                alive -= remB
                            did_any = True
                            changed = True
                            break
                        else:
                            if len(A) <= len(B):
                                alive -= set(A)
                            else:
                                alive -= set(B)
                            did_any = True
                            changed = True
                            break

                    else:
                        remA = try_remove_one_pair_to_resolve(A, B, self_is_multi=False)
                        remB = try_remove_one_pair_to_resolve(B, A, self_is_multi=False)

                        if remA is not None or remB is not None:
                            def rem_cost(remset):
                                if remset is None:
                                    return (10**9, 10**18)
                                c = 0.0
                                for r in remset:
                                    c += float(score.get(r, 0.0))
                                return (len(remset), c)

                            if rem_cost(remA) <= rem_cost(remB):
                                alive -= remA
                            else:
                                alive -= remB
                        else:
                            if len(A) <= len(B):
                                alive -= set(A)
                            else:
                                alive -= set(B)
                        did_any = True
                        changed = True
                        break

            if did_any:
                break

        if not did_any:
            break 

    final_local2 = set(alive)

    corr2 = defaultdict(set)
    for p in final_local2:
        for q in corr_local.get(p, ()):
            if q in final_local2:
                corr2[p].add(q)

    return final_local2, dict(corr2)


def build_pointset(best_layout, patches, patch_faces, contours_raw, adj_graph_2d, best_choice,
                   radius=0.1, min_gap=25, max_gap=40, max_insertions=2, close_gap=20, curv_percentage=80):
    
    stitch_points_world, corr_world = collect_stitch_points(
        contours_raw, best_layout, patches, adj_graph_2d,
        q=curv_percentage, distance_threshold=radius*2, debug_plot=False
    )

    contours = build_contours_pieces(contours_raw, best_choice, adj_graph_2d)
    contours_world = {}
    for pid, poly_list in contours.items():
        if pid not in best_layout:
            continue
        R, t = best_layout[pid]["R"], best_layout[pid]["t"]
        tmp = []
        for poly in poly_list:
            W = _apply(R, t, poly)
            if len(W) >= 2:
                tmp.append(W)
        if tmp:
            contours_world[pid] = tmp

    try:
        merged_contour = outline_euler(best_layout, patches, patch_faces, adj_graph_2d, best_choice, min_seg=0.0)
        _, merged_contour = merge_all_contours(contours_world)
        
    except Exception:
        _, merged_contour = merge_all_contours(contours_world)

    adj_graph_2d_world = adj_to_world(adj_graph_2d, best_layout)


    stitch_points_world, corr_world = add_midpoints_sparse(
        merged_contour=merged_contour,
        stitch_points=stitch_points_world,
        corr_map=corr_world,
        adj_graph_2d_world=adj_graph_2d_world,
        min_gap=min_gap,
        max_gap=max_gap,
        max_insertions=max_insertions,
        close_gap=close_gap,
        distance_threshold=radius*2,
    )

    # 6) shift in/out
    shifted_in = shift_stitch_points(
        stitch_points_world, best_layout, patches, patch_faces, contours_raw,
        shift_eps=radius, ward="inward"
    )
    shifted_out = shift_stitch_points(
        stitch_points_world, best_layout, patches, patch_faces, contours_raw,
        shift_eps=radius, ward="outward"
    )

    

    return (
        merged_contour,
        stitch_points_world,
        corr_world,
        shifted_in,
        shifted_out,
        contours,
        contours_world,
        adj_graph_2d_world,
    )