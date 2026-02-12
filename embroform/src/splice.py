import numpy as np
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def find_index_by_patch_and_position(pt, metadata, target_patch, tol=None):
    pt_arr = np.array(pt)
    candidate_indices = [
        i for i, (pid, coord) in enumerate(metadata)
        if pid == target_patch
    ]
    if not candidate_indices:
        raise ValueError(f"No points for patch {target_patch} found in metadata")

    dists = [np.linalg.norm(np.array(metadata[i][1]) - pt_arr) for i in candidate_indices]
    best_i = candidate_indices[np.argmin(dists)]

    if tol is not None and min(dists) > tol:
        raise ValueError(
            f"Point {pt} not found near patch {target_patch} (min dist={min(dists):.6f})"
        )

    return best_i


def extract_splicing_area(
    patchA,
    patchB,
    merged_contour,
    correspondence_map,
    start_coord_A,
    start_coord_B,
    dir_A,
    dir_B,
    radius
):
    N = len(merged_contour)
    idx_A = find_index_by_patch_and_position(start_coord_A, merged_contour, patchA)
    idx_B = find_index_by_patch_and_position(start_coord_B, merged_contour, patchB)

    region_A = []
    region_B = []

    stitch_A = None
    stitch_B = None
    last_set_A = set()
    last_set_B = set()
    repeat_map = {}

    for _ in range(N):
        pid_A, local_A = merged_contour[idx_A]
        pid_B, local_B = merged_contour[idx_B]

        key_A = (pid_A, local_A)
        key_B = (pid_B, local_B)

        if stitch_A is None:
            if key_A in correspondence_map and not (correspondence_map[key_A] & last_set_A):
                stitch_A = key_A
            elif key_A in correspondence_map and (correspondence_map[key_A] & last_set_A):
                inter = correspondence_map[key_A] & last_set_A
                if all(np.linalg.norm(np.array(local_A) - np.array(pt[1])) > radius for pt in inter):
                    if len(last_set_A) >= 2:
                        raise ValueError("The size of last_set_A >= 2")
                    prev = next(iter(last_set_A))
                    last_set_A.remove(prev)
                    repeat_map[prev] = (pid_A + 0.5, local_A)
                idx_A = (idx_A + dir_A) % N
            else:
                idx_A = (idx_A + dir_A) % N

        if stitch_B is None:
            if key_B in correspondence_map and not (correspondence_map[key_B] & last_set_B):
                stitch_B = key_B
            elif key_B in correspondence_map and (correspondence_map[key_B] & last_set_B):
                inter = correspondence_map[key_B] & last_set_B
                if all(np.linalg.norm(np.array(local_B) - np.array(pt[1])) > radius for pt in inter):
                    if len(last_set_B) >= 2:
                        raise ValueError("The size of last_set_B >= 2")
                    prev = next(iter(last_set_B))
                    last_set_B.remove(prev)
                    repeat_map[prev] = (pid_B + 0.5, local_B)
                idx_B = (idx_B + dir_B) % N
            else:
                idx_B = (idx_B + dir_B) % N

        if stitch_A is not None and stitch_B is not None:
            if key_A in correspondence_map and key_B in correspondence_map[key_A]:
                region_A.append(key_A)
                region_B.append(key_B)

                stitch_A = None
                stitch_B = None
                idx_A = (idx_A + dir_A) % N
                idx_B = (idx_B + dir_B) % N

                last_set_A = {key_A}
                last_set_B = {key_B}
            else:
                break

    return region_A, region_B, repeat_map


def extract_all_splicing_areas(
    best_choice,
    adj_graph_2d,
    merged_contour,
    correspondence_map,
    radius
):
    repeat_map_list = []
    areas = []
    area_id = 0
    N = len(merged_contour)

    for (i, j), idx in best_choice.items():
        if (i, j) not in adj_graph_2d:
            continue

        edge_pair_list = adj_graph_2d[(i, j)]
        if idx >= len(edge_pair_list):
            continue

        (pi0, pi1), (pj0, pj1) = edge_pair_list[idx]

        try:
            idx_pi1 = find_index_by_patch_and_position(tuple(pi1), merged_contour, i)
            idx_pj1 = find_index_by_patch_and_position(tuple(pj1), merged_contour, j)
        except ValueError:
            continue

        if (idx_pj1 - idx_pi1) % N == 1:
            dir_A, dir_B = -1, 1
        elif (idx_pi1 - idx_pj1) % N == 1:
            dir_A, dir_B = 1, -1
        else:
            continue

        seq_A, seq_B, repeat_map1 = extract_splicing_area(
            i, j, merged_contour, correspondence_map,
            tuple(pi1), tuple(pj1), dir_A, dir_B,
            radius
        )
        repeat_map_list.append((area_id, repeat_map1))

        if seq_A and seq_B:
            areas.append({
                'area_id': area_id,
                'splicing_point': [(i, tuple(pi0)), (j, tuple(pj0))],
                'patch_pair': (i, j),
                'source_patch': i,
                'stitch_sequence': seq_A,
                'corresponding_sequence': seq_B
            })
            area_id += 1

        try:
            idx_pj0 = find_index_by_patch_and_position(tuple(pj0), merged_contour, j)
            idx_pi0 = find_index_by_patch_and_position(tuple(pi0), merged_contour, i)
        except ValueError:
            continue

        if (idx_pi0 - idx_pj0) % N == 1:
            dir_A_rev, dir_B_rev = -1, 1
        elif (idx_pj0 - idx_pi0) % N == 1:
            dir_A_rev, dir_B_rev = 1, -1
        else:
            continue

        seq_B2, seq_A2, repeat_map2 = extract_splicing_area(
            j, i, merged_contour, correspondence_map,
            tuple(pj0), tuple(pi0), dir_A_rev, dir_B_rev,
            radius
        )
        repeat_map_list.append((area_id, repeat_map2))

        if seq_A2 and seq_B2:
            areas.append({
                'area_id': area_id,
                'splicing_point': [(j, tuple(pj0)), (i, tuple(pi0))],
                'patch_pair': (j, i),
                'source_patch': j,
                'stitch_sequence': seq_B2,
                'corresponding_sequence': seq_A2
            })
            area_id += 1

    return areas, repeat_map_list


def filter_close_point_areas(areas, threshold=1e-3):
    filtered = []
    for area in areas:
        seq_A = area['stitch_sequence']
        seq_B = area['corresponding_sequence']

        new_seq_A = []
        new_seq_B = []

        for (pidA, ptA), (pidB, ptB) in zip(seq_A, seq_B):
            dist = np.linalg.norm(np.array(ptA) - np.array(ptB))
            if dist >= threshold:
                new_seq_A.append((pidA, ptA))
                new_seq_B.append((pidB, ptB))

        if new_seq_A:
            new_area = area.copy()
            new_area['stitch_sequence'] = new_seq_A
            new_area['corresponding_sequence'] = new_seq_B
            filtered.append(new_area)

    return filtered


def filter_single_point_areas(areas):
    return [
        area for area in areas
        if len(area['stitch_sequence']) > 1 or len(area['corresponding_sequence']) > 1
    ]


def filter_overlapping_areas(areas):
    n = len(areas)
    keep_flags = [True] * n
    area_sets = []

    for area in areas:
        combined_seq = area['stitch_sequence'] + area['corresponding_sequence']
        area_sets.append(set(combined_seq))

    for i in range(n):
        if not keep_flags[i]:
            continue
        for j in range(i + 1, n):
            if not keep_flags[j]:
                continue
            if len(area_sets[i] & area_sets[j]) > 1:
                len_i = len(area_sets[i])
                len_j = len(area_sets[j])

                if len_i > len_j:
                    keep_flags[j] = False
                elif len_j > len_i:
                    keep_flags[i] = False
                    break
                else:
                    pt_i1 = np.array(areas[i]['stitch_sequence'][0][1])
                    pt_i2 = np.array(areas[i]['corresponding_sequence'][0][1])
                    pt_j1 = np.array(areas[j]['stitch_sequence'][0][1])
                    pt_j2 = np.array(areas[j]['corresponding_sequence'][0][1])

                    dist_i = np.linalg.norm(pt_i1 - pt_i2)
                    dist_j = np.linalg.norm(pt_j1 - pt_j2)

                    if dist_i >= dist_j:
                        keep_flags[j] = False
                    else:
                        keep_flags[i] = False
                        break

    return [area for area, keep in zip(areas, keep_flags) if keep]

def visualize_splicing_areas_save_svg(
    areas,
    merged_contour=None,
    correspondence_map=None,
    out_svg=None,
    figsize=(10, 10),
    show_labels=False
):


    if out_svg is None:
        raise ValueError("out_svg is required")

    fig, ax = plt.subplots(figsize=figsize)
    cmap = cm.get_cmap('tab10', max(len(areas), 10))

    if merged_contour is not None:
        poly = np.array([pt[1] for pt in merged_contour], float)
        ax.plot(poly[:, 0], poly[:, 1], '-', color='blue', linewidth=0.8, alpha=0.5)

    if correspondence_map is not None and len(correspondence_map) > 0:
        red_pts = np.array([list(coord) for (_, coord) in correspondence_map.keys()], float)
        ax.plot(red_pts[:, 0], red_pts[:, 1], '.', color='red', alpha=0.35, markersize=2)

    for area in areas:
        area_id = area['area_id']
        color = cmap(area_id % 10)

        seqA = area['stitch_sequence']
        seqB = area['corresponding_sequence']

        if seqA:
            coordsA = np.array([pt[1] for pt in seqA], float)
            if len(coordsA) > 1:
                ax.plot(coordsA[:, 0], coordsA[:, 1], '-', color=color, linewidth=1.2)
            ax.plot(coordsA[:, 0], coordsA[:, 1], 'o', color=color, markersize=4, markeredgewidth=0)

            if show_labels:
                for k, pt in enumerate(coordsA):
                    ax.text(pt[0], pt[1], f'A{area_id}-{k}', color=color, fontsize=6)

        if seqB:
            coordsB = np.array([pt[1] for pt in seqB], float)
            if len(coordsB) > 1:
                ax.plot(coordsB[:, 0], coordsB[:, 1], '--', color=color, linewidth=1.2)
            ax.plot(coordsB[:, 0], coordsB[:, 1], 'x', color=color, markersize=4)

            if show_labels:
                for k, pt in enumerate(coordsB):
                    ax.text(pt[0], pt[1], f'B{area_id}-{k}', color=color, fontsize=6)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def run_splicing(
    best_choice,
    adj_graph_2d_world,
    merged_contour,
    correspondence_map_world,
    radius=0.1,
    filter_close_threshold=None,
    svg3=None,
    show_labels=True
):
    areas_raw, repeat_map_list = extract_all_splicing_areas(
        best_choice,
        adj_graph_2d_world,
        merged_contour,
        correspondence_map_world,
        radius
    )

    areas = areas_raw
    areas = filter_overlapping_areas(areas)
    areas = filter_single_point_areas(areas)

    if filter_close_threshold is not None:
        areas = filter_close_point_areas(areas, threshold=filter_close_threshold)

    if svg3 is not None:
        visualize_splicing_areas_save_svg(
            areas,
            merged_contour=merged_contour,
            correspondence_map=correspondence_map_world,
            out_svg=svg3,
            show_labels=True
        )


    valid_area_ids = {area['area_id'] for area in areas}
    filtered_repeat_map_list = [
        (idx, content) for idx, content in repeat_map_list
        if idx in valid_area_ids and content
    ]
    repeat_map = {k: v for _, d in filtered_repeat_map_list for k, v in d.items()}

    return areas, repeat_map
