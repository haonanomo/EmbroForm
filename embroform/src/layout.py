from __future__ import annotations

import os
import copy
import random
from pathlib import Path
from collections import defaultdict, Counter, deque
from itertools import combinations

import numpy as np
import trimesh
import networkx as nx

try:
    from shapely.geometry import Polygon
except Exception as e:
    Polygon = None


def edge_curvature(p_prev, p_curr, p_next):
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return -np.inf
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    return 1 - abs(cos_angle)


def max_curvature_index(edge_seq):
    pts = [np.array(e[0]) for e in edge_seq] + [np.array(edge_seq[-1][1])]
    best_i = 0
    best_c = -np.inf
    for i in range(1, len(pts) - 1):
        c = edge_curvature(pts[i - 1], pts[i], pts[i + 1])
        if c > best_c:
            best_c = c
            best_i = i - 1
    return best_i, best_c


def rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def align_edge(edge_from, edge_to):
    v1 = edge_from[1] - edge_from[0]
    v2 = edge_to[1] - edge_to[0]
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])
    theta = a1 - a2
    R = rotation_matrix(theta)
    t = edge_from[0] - R @ edge_to[0]
    return R, t


def apply_transform(points, R, t):
    return (R @ points.T).T + t


def parse_patch_id(name):
    s = name
    if s.startswith("patch_"):
        s = s[len("patch_"):]
    s = s.split("_")[0]
    try:
        return int(s)
    except:
        return None


def load_patch_data(root_dir, patches_pp_dir="patches_pp", patches_dir="patches"):
    root_dir = Path(root_dir)
    pp_dir = root_dir / patches_pp_dir
    p_dir = root_dir / patches_dir

    patch_2d_mesh = {}
    patch_vmap = {}

    pp_files = sorted(pp_dir.glob("patch_*_pp_para.obj"))
    for f in pp_files:
        pid = parse_patch_id(f.stem)
        if pid is None:
            continue
        vmap_path = p_dir / f"patch_{pid}_vmap.txt"
        if not vmap_path.exists():
            continue
        patch_2d_mesh[pid] = trimesh.load_mesh(f, process=False)
        patch_vmap[pid] = np.loadtxt(vmap_path, dtype=int)

    return patch_2d_mesh, patch_vmap


def build_adj_graph_3d(root_dir, final_obj="final.obj", final_seg="final_seg.txt"):
    root_dir = Path(root_dir)
    mesh = trimesh.load_mesh(root_dir / final_obj, process=False)
    face_segments = np.loadtxt(root_dir / final_seg, dtype=int)

    if len(face_segments) != len(mesh.faces):
        raise ValueError("final_seg and final faces length mismatch")

    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
        a, b, c = face
        for u, v in ((a, b), (b, c), (c, a)):
            u, v = (u, v) if u < v else (v, u)
            edge_to_faces[(u, v)].append(fi)

    adj_graph_3d = defaultdict(list)
    for edge, faces in edge_to_faces.items():
        if len(faces) != 2:
            continue
        f1, f2 = faces
        p1 = int(face_segments[f1])
        p2 = int(face_segments[f2])
        if p1 == p2:
            continue
        key = tuple(sorted((p1, p2)))
        adj_graph_3d[key].append(edge)

    return adj_graph_3d


def build_adj_graph_2d(adj_graph_3d, patch_2d_mesh, patch_vmap):
    adj_graph_2d = defaultdict(list)

    for (pi, pj), edges in adj_graph_3d.items():
        if pi not in patch_2d_mesh or pj not in patch_2d_mesh:
            continue

        Vi = patch_2d_mesh[pi].vertices
        Vj = patch_2d_mesh[pj].vertices
        vmap_i = patch_vmap[pi]
        vmap_j = patch_vmap[pj]

        for v0g, v1g in edges:
            idx_i0 = np.where(vmap_i == v0g)[0]
            idx_i1 = np.where(vmap_i == v1g)[0]
            idx_j0 = np.where(vmap_j == v0g)[0]
            idx_j1 = np.where(vmap_j == v1g)[0]
            if len(idx_i0) == 0 or len(idx_i1) == 0 or len(idx_j0) == 0 or len(idx_j1) == 0:
                continue

            ii0 = int(idx_i0[0])
            ii1 = int(idx_i1[0])
            ij0 = int(idx_j0[0])
            ij1 = int(idx_j1[0])

            pi0, pi1 = Vi[ii0][:2], Vi[ii1][:2]
            pj0, pj1 = Vj[ij0][:2], Vj[ij1][:2]

            adj_graph_2d[(pi, pj)].append(((pi0, pi1), (pj0, pj1)))
            adj_graph_2d[(pj, pi)].append(((pj1, pj0), (pi1, pi0)))

    return adj_graph_2d


def build_weighted_graph(adj_graph_2d):
    G = nx.Graph()
    for (i, j), pairs in adj_graph_2d.items():
        if not pairs:
            continue
        total = 0.0
        for (p0, p1), _ in pairs:
            p0 = np.array(p0)
            p1 = np.array(p1)
            total += np.linalg.norm(p1 - p0)
        if total <= 0:
            continue
        G.add_edge(i, j, weight=-float(total))
    return G


def extract_single_contour_from_faces(patches, patch_faces):
    contours = {}

    for pid, F in patch_faces.items():
        V = np.asarray(patches[pid], float)
        F = np.asarray(F, int)
        if V.size == 0 or F.size == 0:
            continue

        edge_count = Counter()
        for a, b, c in F:
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v:
                    u, v = v, u
                edge_count[(u, v)] += 1

        boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
        if not boundary_edges:
            continue

        adj = defaultdict(list)
        for u, v in boundary_edges:
            adj[u].append(v)
            adj[v].append(u)

        used_edge = set()
        loops = []

        def walk(start):
            path = [start]
            cur = start
            prev = -1
            while True:
                nxt = None
                for n in adj[cur]:
                    e = (min(cur, n), max(cur, n))
                    if e in used_edge:
                        continue
                    if n == prev and len(adj[cur]) > 1:
                        continue
                    nxt = n
                    used_edge.add(e)
                    break
                if nxt is None:
                    break
                path.append(nxt)
                prev, cur = cur, nxt
                if len(path) > 2 and path[-1] == path[0]:
                    break
            return path

        visited = set()
        for u, v in boundary_edges:
            for s in (u, v):
                if s in visited:
                    continue
                visited.add(s)
                p = walk(s)
                if len(p) >= 2:
                    loops.append(p)

        if not loops:
            continue

        def score(p):
            closed = (len(p) > 2 and p[0] == p[-1])
            L = 0.0
            for i in range(len(p) - 1):
                L += np.linalg.norm(V[p[i + 1]] - V[p[i]])
            return (1 if closed else 0, L, len(p))

        loops.sort(key=score, reverse=True)
        best = loops[0]

        coords = [V[i] for i in best]
        if len(best) > 2 and best[0] == best[-1]:
            if not np.allclose(coords[0], coords[-1]):
                coords.append(coords[0])

        contours[pid] = np.asarray(coords, float)

    return contours


def initialize_edge_pair_choice_greedy(mst, adj_graph_2d):
    choice = {}
    for u, v in mst.edges():
        if (u, v) in adj_graph_2d:
            key = (u, v)
        elif (v, u) in adj_graph_2d:
            key = (v, u)
        else:
            continue

        pairs = adj_graph_2d.get(key, [])
        if not pairs:
            continue
        if len(pairs) == 1:
            choice[key] = 0
            continue

        patch_a_edges = [pair[0] for pair in pairs]
        patch_b_edges = [pair[1] for pair in pairs]

        ia, ca = max_curvature_index(patch_a_edges)
        ib, cb = max_curvature_index(patch_b_edges)

        choice[key] = ia if ca >= cb else ib

    return choice


def construct_layout_from_choices(mst, adj_graph_2d, edge_pair_choice):
    layout = {}
    root = max(dict(mst.degree()).items(), key=lambda x: x[1])[0]
    layout[root] = {"R": np.eye(2), "t": np.zeros(2)}
    q = deque([root])

    while q:
        cur = q.popleft()
        Rc = layout[cur]["R"]
        tc = layout[cur]["t"]

        for nb in mst.neighbors(cur):
            if nb in layout:
                continue

            if (cur, nb) in adj_graph_2d:
                key, reverse = (cur, nb), False
            elif (nb, cur) in adj_graph_2d:
                key, reverse = (nb, cur), True
            else:
                continue

            pairs = adj_graph_2d[key]
            if not pairs:
                continue

            pref = edge_pair_choice.get(key, edge_pair_choice.get((key[1], key[0]), 0))
            order = list(range(len(pairs)))
            if pref in order:
                order.remove(pref)
                order = [pref] + order

            placed = False
            for idx in order:
                edge_from, edge_to = pairs[idx]
                if reverse:
                    edge_from, edge_to = edge_to[::-1], edge_from[::-1]

                edge_from_g = apply_transform(np.array(edge_from), Rc, tc)
                R, t = align_edge(edge_from_g, np.array(edge_to))
                layout[nb] = {"R": R, "t": t}
                q.append(nb)
                placed = True
                break

            if not placed:
                continue

    return layout


def compute_overlap_area(layout, contours_raw):
    if Polygon is None:
        raise RuntimeError("shapely is required for overlap evaluation")

    polys = {}
    for i in layout:
        if i not in contours_raw:
            continue
        R = layout[i]["R"]
        t = layout[i]["t"]
        pts = (R @ np.array(contours_raw[i]).T).T + t
        try:
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 0:
                polys[i] = poly
        except:
            pass

    total = 0.0
    pairs = []
    ids = list(polys.keys())
    for a, b in combinations(ids, 2):
        pa = polys[a]
        pb = polys[b]
        if not pa.intersects(pb):
            continue
        try:
            inter = pa.intersection(pb)
            if inter.area > 1e-10:
                total += float(inter.area)
                pairs.append((a, b))
        except:
            pass

    return total, pairs


def evaluate_layout(layout, contours_raw):
    overlap, pairs = compute_overlap_area(layout, contours_raw)
    return overlap, pairs


def perturb_edge_pair_choice(edge_pair_choice, adj_graph_2d, num_changes=7):
    new_choice = edge_pair_choice.copy()
    keys = list(new_choice.keys())
    if not keys:
        return new_choice

    for _ in range(int(num_changes)):
        key = random.choice(keys)
        cands = adj_graph_2d.get(key, []) or adj_graph_2d.get((key[1], key[0]), [])
        if len(cands) <= 1:
            continue
        new_idx = random.randint(0, len(cands) - 1)
        new_choice[key] = new_idx

    return new_choice


def optimize_layout_lns_sa(contours_raw, mst, adj_graph_2d, init_choice,
                           T_start=1.0, T_end=1e-10, cooling_rate=0.99, max_iters=3000, verbose=True):

    best_choice = copy.deepcopy(init_choice)
    best_layout = construct_layout_from_choices(mst, adj_graph_2d, best_choice)
    best_score, _ = evaluate_layout(best_layout, contours_raw)

    curr_choice = copy.deepcopy(best_choice)
    curr_score = float(best_score)
    T = float(T_start)

    if verbose:
        print(f"[Init] score = {best_score}")

    for it in range(int(max_iters)):
        new_choice = perturb_edge_pair_choice(curr_choice, adj_graph_2d, num_changes=7)
        new_layout = construct_layout_from_choices(mst, adj_graph_2d, new_choice)
        new_score, _ = evaluate_layout(new_layout, contours_raw)

        delta = float(new_score) - float(curr_score)
        accept = (delta < 0) or (random.random() < np.exp(-delta / max(T, 1e-12)))

        if accept:
            curr_choice = copy.deepcopy(new_choice)
            curr_score = float(new_score)

            if new_score < best_score:
                best_score = float(new_score)
                best_choice = copy.deepcopy(new_choice)
                best_layout = copy.deepcopy(new_layout)

        if verbose and it % 200 == 0:
            print(f"[{it:04d}] best = {best_score}")

        T *= float(cooling_rate)
        if T < float(T_end) or best_score < 1e-12:
            break

    best_layout = construct_layout_from_choices(mst, adj_graph_2d, best_choice)
    return best_layout, best_choice, best_score


def save_layout_obj(layout, patches, patch_faces, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    all_vertices = []
    all_faces = []
    v_offset = 0

    for pid in sorted(layout.keys()):
        if pid not in patches or pid not in patch_faces:
            continue
        R = layout[pid]["R"]
        t = layout[pid]["t"]
        verts = apply_transform(np.asarray(patches[pid], float), R, t)
        faces = np.asarray(patch_faces[pid], int)

        all_vertices.extend(verts.tolist())
        all_faces.extend((faces + v_offset).tolist())
        v_offset += verts.shape[0]

    with open(filename, "w") as f:
        for v in all_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} 0.000000\n")
        for fa in all_faces:
            f.write(f"f {fa[0]+1} {fa[1]+1} {fa[2]+1}\n")

    return filename


def generate_random_spanning_trees(G, n=20, seed_base=2):
    trees = []
    seen = set()

    for i in range(int(n) * 5):
        seed = int(seed_base) + i
        T_raw = nx.random_spanning_tree(G, seed=seed)

        if isinstance(T_raw, nx.Graph):
            T = T_raw
        elif isinstance(T_raw, dict):
            T = nx.Graph()
            for child, parent in T_raw.items():
                if child != parent:
                    T.add_edge(child, parent)
        else:
            T = nx.Graph()
            for k in range(len(T_raw)):
                if T_raw[k] != k:
                    T.add_edge(k, T_raw[k])

        edge_key = frozenset(tuple(sorted((u, v))) for u, v in T.edges())
        if edge_key in seen:
            continue
        seen.add(edge_key)
        trees.append(T)
        if len(trees) >= int(n):
            break

    return trees


def build_layout(root_dir, out_obj="layout_final.obj",
           n_trees=20, seed_base=2,
           prefilter_pairs=True,
           sa_pair_iters=3000, sa_topo_iters=4000,
           T_start=1.0, T_end=1e-10, cooling_rate=0.99,
           verbose=True):

    root_dir = Path(root_dir)

    adj_graph_3d = build_adj_graph_3d(root_dir)
    patch_2d_mesh, patch_vmap = load_patch_data(root_dir)

    if verbose:
        print(f"[INFO] patches loaded: {len(patch_2d_mesh)}")

    adj_graph_2d = build_adj_graph_2d(adj_graph_3d, patch_2d_mesh, patch_vmap)
    if verbose:
        print(f"[INFO] adj_graph_2d pairs: {len(adj_graph_2d)}")

    G = build_weighted_graph(adj_graph_2d)
    if verbose:
        print(f"[INFO] graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    patches = {i: patch_2d_mesh[i].vertices[:, :2] for i in patch_2d_mesh}
    patch_faces = {i: patch_2d_mesh[i].faces for i in patch_2d_mesh}
    contours_raw = extract_single_contour_from_faces(patches, patch_faces)

    if prefilter_pairs and G.number_of_edges() > 0:
        bad = []
        for u, v in list(G.edges()):
            if verbose:
                print(f"[PAIR] test ({u},{v})")
            T = nx.Graph()
            T.add_edge(u, v)

            try:
                init_choice = initialize_edge_pair_choice_greedy(T, adj_graph_2d)
                best_layout, best_choice, best_score = optimize_layout_lns_sa(
                    contours_raw, T, adj_graph_2d, init_choice,
                    T_start=T_start, T_end=T_end, cooling_rate=cooling_rate, max_iters=sa_pair_iters,
                    verbose=False
                )
                if best_score >= 1e-10:
                    bad.append((u, v))
                    if verbose:
                        print(f"[PAIR] collision ({u},{v}) score={best_score}")
                else:
                    if verbose:
                        print(f"[PAIR] ok ({u},{v})")
            except Exception as e:
                bad.append((u, v))
                if verbose:
                    print(f"[PAIR] fail ({u},{v}) {e}")

        if bad:
            G = G.copy()
            G.remove_edges_from(bad)
            if verbose:
                print(f"[INFO] removed {len(bad)} edges, remain edges={G.number_of_edges()}")

    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    if isolated:
        raise RuntimeError(f"Graph infeasible after collision filtering. Isolated patches: {isolated}")

    if not nx.is_connected(G):
        comps = [sorted(list(c)) for c in nx.connected_components(G)]
        raise RuntimeError(f"Graph infeasible after collision filtering. Disconnected components: {comps}")


    random_trees = generate_random_spanning_trees(G, n=n_trees, seed_base=seed_base)
    if verbose:
        print(f"[INFO] trees: {len(random_trees)}")

    best_score = float("inf")
    best_layout = None
    best_choice = None
    best_idx = -1

    all_pids = set(patches.keys())

    for i, mst in enumerate(random_trees):
        try:
            init_choice = initialize_edge_pair_choice_greedy(mst, adj_graph_2d)
            lay, ch, sc = optimize_layout_lns_sa(
                contours_raw, mst, adj_graph_2d, init_choice,
                T_start=T_start, T_end=T_end, cooling_rate=cooling_rate, max_iters=sa_topo_iters,
                verbose=False
            )
        except Exception as e:
            if verbose:
                print(f"[TOPO {i}] fail {e}")
            continue

        if lay is None:
            continue

        if set(lay.keys()) != all_pids:
            sc = 1e12

        if verbose:
            print(f"[TOPO {i}] score={sc}")

        if sc < 1e-10:
            best_score = sc
            best_layout = lay
            best_choice = ch
            best_idx = i
            break

        if sc < best_score:
            best_score = sc
            best_layout = lay
            best_choice = ch
            best_idx = i

    if best_layout is None:
        raise RuntimeError("all topos failed")

    out_path = root_dir / out_obj
    save_layout_obj(best_layout, patches, patch_faces, out_path)

    if verbose:
        print(f"[DONE] best topo={best_idx} score={best_score}")
        print(f"[OK] wrote: {out_path}")
        
    return  best_layout, best_choice, adj_graph_3d, adj_graph_2d, G, patches, patch_faces, contours_raw
