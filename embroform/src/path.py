import math
import random
from typing import Optional, List, Tuple, Dict, Set

import numpy as np
import networkx as nx

from shapely.geometry import LineString, Point, Polygon, MultiLineString
from shapely.strtree import STRtree

from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

def is_intersect_zigzag(seg1, seg2, close=False, tol=0.001):

    if isinstance(seg1[0], np.ndarray):
        p1 = seg1[0]
        p2 = seg1[1]
    else:
        p1 = np.array(seg1[0][1])
        p2 = np.array(seg1[1][1])
    if isinstance(seg2[1], np.ndarray):
        q1 = seg2[0]
        q2 = seg2[1]
    else:
        q1 = np.array(seg2[0][1])
        q2 = np.array(seg2[1][1])

    line1 = LineString([p1, p2])
    line2 = LineString([q1, q2])

    if not line1.intersects(line2):
        if close:
            if line2.distance(Point(p1)) < tol or line2.distance(Point(p2)) < tol:
                return True
            if line1.distance(Point(q1)) < tol or line1.distance(Point(q2)) < tol:
                return True
        return False


    inter = line1.intersection(line2)

    if inter.geom_type == 'Point':
        pt = (inter.x, inter.y)

        is_on_p = any(np.linalg.norm(np.array(pt) - ep) < 1e-8 for ep in [p1, p2])
        is_on_q = any(np.linalg.norm(np.array(pt) - ep) < 1e-8 for ep in [q1, q2])
        return not (is_on_p and is_on_q)

    return True



def is_colinear(A, B, C, D, angle_threshold=5):

    v1 = B - A
    v2 = D - C
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return False

    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    angle_deg = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
    return angle_deg < angle_threshold or angle_deg > (180 - angle_threshold)



def cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]



def is_same_side(K1, K2, P1, P2):

    v  = K2 - K1
    v1 = P1 - K1
    v2 = P2 - K1

    cross1 = cross_2d(v, v1)
    cross2 = cross_2d(v, v2)

    return cross1 * cross2 > 0



def extend_along_direction(src, tgt, epsilon=0.5):
    direction = tgt - src
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("Too close to determine direction.")
    direction /= norm
    return src + direction * (norm + epsilon)



def compute_foot_point(P, A, B, inside=False):
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)

    AB = B - A
    AP = P - A
    AB_norm_sq = np.dot(AB, AB)

    if AB_norm_sq < 1e-10:
        return A

    proj_len = np.dot(AP, AB) / AB_norm_sq

    if inside:
        if proj_len < -1e-5 or proj_len > 1+1e-5:
            return None

    H = A + proj_len * AB
    return H





def reroute_zigzag(
    positive_direction_right,
    accepted_segments,
    pt_start,
    pt_prev,
    pt_curr,
    intersection_seg,
    stitch_points_world,
    epsilon=0.5,
    radius=1.0
):
    B = np.array(pt_prev[1])
    C = np.array(pt_curr[1])
    S = np.array(pt_start)
    pid = -1

    detour_pts = n_points_detour(positive_direction_right, accepted_segments, intersection_seg, C, S, epsilon=epsilon, radius=radius, strategy="full")
    reroute = [pt_prev] + [(-1, tuple(p)) for p in detour_pts] + [pt_curr]
    return reroute






def find_intersecting_stitch_point(stitch_points_world, seg, radius=0.5):
    seg_start, seg_end = seg
    stitch_points_world_array = [np.array(pt) for _, pt in stitch_points_world]
    for p in stitch_points_world_array:
        if np.array_equal(p, seg_start) or np.array_equal(p, seg_end):
            continue
        if is_intersect_zigzag_circle(p, seg, radius=radius):
            return p
    return None






def is_intersect_zigzag_circle(
    center_point,
    segment,
    radius=0.5
) -> bool:

    if isinstance(center_point, tuple) and isinstance(center_point[1], (np.ndarray, tuple)):
        pt_center = tuple(center_point[1]) if isinstance(center_point[1], np.ndarray) else center_point[1]
    else:
        pt_center = tuple(center_point) if isinstance(center_point, np.ndarray) else center_point

    pt1_raw = segment[0]
    if isinstance(pt1_raw, tuple) and isinstance(pt1_raw[1], (np.ndarray, tuple)):
        pt1 = tuple(pt1_raw[1]) if isinstance(pt1_raw[1], np.ndarray) else pt1_raw[1]
    else:
        pt1 = tuple(pt1_raw) if isinstance(pt1_raw, np.ndarray) else pt1_raw

    pt2_raw = segment[1]
    if isinstance(pt2_raw, tuple) and isinstance(pt2_raw[1], (np.ndarray, tuple)):
        pt2 = tuple(pt2_raw[1]) if isinstance(pt2_raw[1], np.ndarray) else pt2_raw[1]
    else:
        pt2 = tuple(pt2_raw) if isinstance(pt2_raw, np.ndarray) else pt2_raw

    circle = Point(pt_center).buffer(radius)
    line = LineString([pt1, pt2])
    return circle.intersects(line)



def is_point_in_quad(edge1, edge2, point):

    pt1 = np.array(edge1[0][1])
    pt2 = np.array(edge1[1][1])
    pt3 = np.array(edge2[0][1])
    pt4 = np.array(edge2[1][1])
    test_pt = np.array(point[1])

    quad = Polygon([pt1, pt2, pt4, pt3])
    if not quad.is_valid:
        quad = Polygon([pt1, pt2, pt3, pt4])
    return quad.contains(Point(test_pt)) or quad.touches(Point(test_pt))


def is_left(O, R, P, tol=1e-8):
    O, R, P = np.asarray(O), np.asarray(R), np.asarray(P)
    d = R - O
    OP = P - O
    cross = np.cross(d, OP)

    if abs(cross) < tol:
        return "on the ray"
    return cross > 0  # True=left


def direction_reroute(i, raw_path, pt_start):
    if is_left(np.array(raw_path[0][1]),np.array(raw_path[1][1]),np.array(pt_start)):
        if (i%2==0 and is_left(np.array(raw_path[i - 2][1]),np.array(raw_path[i-1][1]),np.array(raw_path[i][1]))):
            return True
        elif (i%2!=0 and not is_left(np.array(raw_path[i - 2][1]),np.array(raw_path[i-1][1]),np.array(raw_path[i][1]))):
            return True
        else:
            return False
    else:
        if (i%2==0 and is_left(np.array(raw_path[i - 2][1]),np.array(raw_path[i-1][1]),np.array(raw_path[i][1]))):
            return False
        elif (i%2!=0 and not is_left(np.array(raw_path[i - 2][1]),np.array(raw_path[i-1][1]),np.array(raw_path[i][1]))):
            return False
        else:
            return True




def generate_zigzag_path_simple(area, mode= 0):


    seqA = area['stitch_sequence']
    seqB = area['corresponding_sequence']

    if len(seqA) != len(seqB):
        raise ValueError(f"Area {area['area_id']} stitch/corresponding sequence length are different, cannot do zigzag.")

    if mode in [2, 3]:
        seqA = seqA[::-1]
        seqB = seqB[::-1]

    path = []
    if mode % 2 == 0:
        for a, b in zip(seqA, seqB):
            path.append(a)
            path.append(b)
    else:
        for a, b in zip(seqA, seqB):
            path.append(b)
            path.append(a)

    return path



def visualize_zigzag_paths(zigzag_paths, contours_raw, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)

    from matplotlib import colormaps
    cmap = colormaps['tab10'].resampled(len(zigzag_paths))

    x_all, y_all = [], []
    for contour in contours_raw.values():
        contour = np.array(contour)
        if len(contour.shape) > 2:
            contour = contour[0]
        x_all.extend(contour[:, 0])
        y_all.extend(contour[:, 1])

    for path_data in zigzag_paths:
        for _, (x, y) in path_data['point_seq']:
            x_all.append(x)
            y_all.append(y)

    x_min, x_max = min(x_all), max(x_all)
    y_min, y_max = min(y_all), max(y_all)
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin_ratio = 0.1

    ax.set_xlim(x_min - x_range * margin_ratio, x_max + x_range * margin_ratio)
    ax.set_ylim(y_min - y_range * margin_ratio, y_max + y_range * margin_ratio)

    for pid, contour in contours_raw.items():
        contour = np.array(contour)
        if len(contour.shape) > 2:
            contour = contour[0]
        if not np.allclose(contour[0], contour[-1], atol=1e-6):
            contour = np.vstack([contour, contour[0]])
        ax.plot(contour[:, 0], contour[:, 1], '-', color='gray', linewidth=0.5, label='Contour' if pid == 0 else "")

    start_markers = []
    end_markers = []

    for path_data in zigzag_paths:
        area_id = path_data['area_id']
        path = path_data['point_seq']
        color = cmap(area_id % 10)

        if not path:
            continue

        for i in range(len(path) - 1):
            x0, y0 = path[i][1]
            x1, y1 = path[i + 1][1]
            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1)
            )

        x_start, y_start = path[0][1]
        start = ax.plot(x_start, y_start, 'o', color='red', markersize=6)
        if area_id == 0:
            start_markers = start

        x_end, y_end = path[-1][1]
        end = ax.plot(x_end, y_end, 's', color='blue', markersize=6)
        if area_id == 0:
            end_markers = end

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1, label='Contour'),
        Line2D([0], [0], marker='o', color='red', label='Start Point',
               markersize=6, linestyle=''),
        Line2D([0], [0], marker='s', color='blue', label='End Point',
               markersize=6, linestyle='')
    ]
    ax.legend(handles=legend_elements)

    ax.set_aspect('equal')
    ax.set_title("Zigzag Path Visualization")
    plt.tight_layout()
    plt.show()





def generate_expanded_path_2(stitch_path, expanded_path, stitch_points_world,
                            shifted_in_stitch_points, is_bridge, best_modes, offset=0.5):
    sp_index_map = {(pid, tuple(pt)): idx for idx, (pid, pt) in enumerate(stitch_points_world)}
    new_path = []
    n = len(expanded_path)
    prev_bridge = -0.5
    cnt = -1
    current_mode = None

    for i, item in enumerate(expanded_path):
        if isinstance(item, list) and len(item) == 2:
            if i < len(is_bridge) and is_bridge[i] < 0 and is_bridge[i] != prev_bridge:
                prev_bridge = is_bridge[i]
                cnt += 1
                current_mode = best_modes[cnt]

            (pid1, B1), (pid2, B2) = item
            key = stitch_path[i]
            key = (int(key[0]), key[1])
            idx = sp_index_map[key]

            original = np.array(key[1])
            shifted = shifted_in_stitch_points[idx][1]
            direction = shifted - original
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0])

            candidates_B0 = [
                np.array(B1) + offset * direction,
                np.array(B1) - offset * direction,
            ]
            candidates_B3 = [
                np.array(B2) + offset * direction,
                np.array(B2) - offset * direction,
            ]

            rect_poly = Polygon([
                tuple(candidates_B0[0]), tuple(candidates_B3[0]),
                tuple(candidates_B3[1]), tuple(candidates_B0[1])
            ])

            ptA = new_path[-1][-1][1] if i > 0 and isinstance(new_path[-1], list) else (
                new_path[-1][1] if i > 0 else None)
            ptC = expanded_path[i + 1][0][1] if i + 1 < n and isinstance(expanded_path[i + 1], list) else (
                expanded_path[i + 1][1] if i + 1 < n else None)

            B0, B3, B00, B33, evil_A, evil_C = determine_connection_points(
                ptA, ptC, B1, B2, pid1, pid2, candidates_B0, candidates_B3,
                rect_poly, current_mode
            )

            result = build_result_path(
                B0, B3, B00, B33, pid1, pid2, B1, B2, evil_A, evil_C, current_mode
            )

            new_path.append(result)
        else:
            new_path.append(item)

    return new_path





def determine_connection_points(ptA, ptC, B1, B2, pid1, pid2, candidates_B0,
                               candidates_B3, rect_poly, mode):
    B0, B3, B00, B33 = None, None, None, None
    evil_A, evil_C = False, False

    if mode > 1:
        temp_ptA, temp_ptC = ptC, ptA
        temp_B1, temp_B2 = B2, B1
        temp_pid1, temp_pid2 = pid2, pid1
        temp_candidates_B0 = candidates_B3
        temp_candidates_B3 = candidates_B0

        temp_B0, temp_B3, temp_B00, temp_B33, temp_evil_C, temp_evil_A = determine_connection_points(
            temp_ptA, temp_ptC, temp_B1, temp_B2, temp_pid1, temp_pid2,
            temp_candidates_B0, temp_candidates_B3, rect_poly, mode-2
        )

        B0 = temp_B3
        B3 = temp_B0
        B00 = temp_B33
        B33 = temp_B00
        evil_A = temp_evil_C
        evil_C = temp_evil_A

        return B0, B3, B00, B33, evil_A, evil_C

    if ptA is not None:
        lineA = LineString([ptA, B1])
        intersection_A = rect_poly.intersection(lineA)
        if rect_poly.intersects(lineA) and not is_tangent_intersection(intersection_A, B1):


            angles = [angle_between(ptA, B1, cand) for cand in candidates_B0]
            idx = int(np.argmin(angles))
            candidate_B0 = candidates_B0[idx]
            alt_candidate_B0 = candidates_B0[1 - idx]




            B0 = (pid1, tuple(candidate_B0.tolist()))

            if not rect_poly.contains(Point(ptA)):
                lineAA = LineString([ptA, np.array(B0[1])])
                if rect_poly.intersects(lineAA):
                    inter = rect_poly.intersection(lineAA)
                    if not is_tangent_intersection(inter, B0[1]):
                        angles = [angle_between(ptA, cand, B0[1]) for cand in candidates_B3]
                        B33_candidate = candidates_B3[int(np.argmax(angles))]
                        alt_B33_candidate = candidates_B3[int(np.argmin(angles))]

                        if ptC is not None:
                            line_A_to_B33 = LineString([ptA, B33_candidate])
                            line_B33_to_B0 = LineString([B33_candidate, B0[1]])
                            line_B2_to_ptC = LineString([B2, ptC])

                            if (line_A_to_B33.intersects(line_B2_to_ptC) or
                                line_B33_to_B0.intersects(line_B2_to_ptC)):
                                B33_candidate = alt_B33_candidate

                        B33 = (pid1, tuple(B33_candidate.tolist()))

                        if is_intersect_zigzag((np.array(B2), np.array(B1)),
                                              (np.array(B33[1]), np.array(B0[1]))):
                            B33 = (pid1, tuple(alt_B33_candidate.tolist()))

                        if is_intersect_zigzag((np.array(B2), np.array(B1)),
                                              (np.array(B33[1]), np.array(ptA)), close=True, tol=1e-3):
                            evil_A = True
            elif ptC is not None:
                if is_intersect_zigzag((np.array(ptA), np.array(B1)), (np.array(B2), np.array(ptC))):
                    angles = [angle_between(ptA, cand, B1) for cand in candidates_B0]
                    B0 = (pid1, tuple(candidates_B0[int(np.argmin(angles))].tolist()))

                    angles1 = [angle_between(ptA, B1, cand) for cand in candidates_B3]
                    B33 = (pid1, tuple(candidates_B3[int(np.argmax(angles1))].tolist()))
                    evil_A = True




    if ptC is not None:
        lineC = LineString([ptC, B2])
        intersection_C = rect_poly.intersection(lineC)
        if rect_poly.intersects(lineC) and not is_tangent_intersection(intersection_C, B2):
            angles = [angle_between(ptC, cand, B2) for cand in candidates_B3]
            candidate_B3 = candidates_B3[int(np.argmax(angles))]
            alt_candidate_B3 = candidates_B3[int(np.argmin(angles))]

            if ptA is not None:
                line_B3_to_ptC = LineString([candidate_B3, ptC])
                line_A_to_B1 = LineString([ptA, B1])

                if line_B3_to_ptC.intersects(line_A_to_B1):
                    candidate_B3 = alt_candidate_B3

            B3 = (pid2, tuple(candidate_B3.tolist()))

            if ptA is not None:
                if is_intersect_zigzag((np.array(ptA), np.array(B1)), (np.array(B2), np.array(ptC))):
                    B3 = (pid2, tuple(alt_candidate_B3.tolist()))

            if not rect_poly.contains(Point(ptC)):
                lineBB = LineString([ptC, np.array(B3[1])])
                if rect_poly.intersects(lineBB):
                    inter = rect_poly.intersection(lineBB)
                    if not is_tangent_intersection(inter, B3[1]):
                        angles = [angle_between(ptC, cand, B3[1]) for cand in candidates_B0]
                        B00_candidate = candidates_B0[int(np.argmax(angles))]
                        alt_B00_candidate = candidates_B0[int(np.argmin(angles))]

                        if ptA is not None:
                            line_B00_to_ptC = LineString([B00_candidate, ptC])
                            line_A_to_B1 = LineString([ptA, B1])

                            if line_B00_to_ptC.intersects(line_A_to_B1):
                                B00_candidate = alt_B00_candidate

                        B00 = (pid2, tuple(B00_candidate.tolist()))

                        if is_intersect_zigzag((np.array(B2), np.array(B1)),
                                              (np.array(B00[1]), np.array(B3[1]))):
                            B00 = (pid1, tuple(alt_B00_candidate.tolist()))

                        if is_intersect_zigzag((np.array(B2), np.array(B1)),
                                              (np.array(B00[1]), np.array(ptC))):
                            evil_C = True
            elif ptA is not None:
                if is_intersect_zigzag((np.array(ptA), np.array(B1)), (np.array(B2), np.array(ptC))):
                    angles = [angle_between(ptC, cand, B2) for cand in candidates_B3]
                    B3 = (pid2, tuple(candidates_B3[int(np.argmin(angles))].tolist()))

                    angles1 = [angle_between(ptC, B2, cand) for cand in candidates_B0]
                    B00 = (pid2, tuple(candidates_B0[int(np.argmax(angles1))].tolist()))
                    evil_C = True


    if all(x is None for x in [B0, B3, B00, B33]):
        return B0, B3, B00, B33, evil_A, evil_C
    if ptA is not None and ptC is not None:
        path1_segments = []
        if B33 is not None:
            path1_segments.append((np.array(ptA), np.array(B33[1])))
            path1_segments.append((np.array(B33[1]), np.array(B0[1])))
            path1_segments.append((np.array(B0[1]), np.array(B1)))
        elif B0 is not None:
            path1_segments.append((np.array(ptA), np.array(B0[1])))
            path1_segments.append((np.array(B0[1]), np.array(B1)))
        else:
            path1_segments.append((np.array(ptA), np.array(B1)))

        path2_segments = []

        if B00 is not None:
            path2_segments.append((np.array(B1), np.array(B2)))
            path2_segments.append((np.array(B2), np.array(B3[1])))
            path2_segments.append((np.array(B3[1]), np.array(B00[1])))
            path2_segments.append((np.array(B00[1]), np.array(ptC)))
        elif B3 is not None:
            path2_segments.append((np.array(B1), np.array(B2)))
            path2_segments.append((np.array(B2), np.array(B3[1])))
            path2_segments.append((np.array(B3[1]), np.array(ptC)))
        else:
            path2_segments.append((np.array(B1), np.array(B2)))
            path2_segments.append((np.array(B2), np.array(ptC)))

        collision_detected = False
        for seg1 in path1_segments:
            for seg2 in path2_segments:
                if is_intersect_zigzag(seg1, seg2):
                    collision_detected = True
                    break
            if collision_detected:
                break

        if collision_detected and B0 is not None:
            idx = next((i for i, cand in enumerate(candidates_B0) if np.array_equal(cand, np.array(B0[1]))), None)
            candidate_B0 = candidates_B0[1 - idx]
            B0 = (pid1, tuple(candidate_B0.tolist()))

            if not rect_poly.contains(Point(ptA)):
                lineAA = LineString([ptA, np.array(B0[1])])
                if rect_poly.intersects(lineAA):
                    inter = rect_poly.intersection(lineAA)
                    if not is_tangent_intersection(inter, B0[1]):
                        angles = [angle_between(ptA, cand, B0[1]) for cand in candidates_B3]
                        B33_candidate = candidates_B3[int(np.argmax(angles))]
                        alt_B33_candidate = candidates_B3[int(np.argmin(angles))]

                        if ptC is not None:
                            line_A_to_B33 = LineString([ptA, B33_candidate])
                            line_B33_to_B0 = LineString([B33_candidate, B0[1]])
                            line_B2_to_ptC = LineString([B2, ptC])

                            if (line_A_to_B33.intersects(line_B2_to_ptC) or
                                line_B33_to_B0.intersects(line_B2_to_ptC)):
                                B33_candidate = alt_B33_candidate

                        B33 = (pid1, tuple(B33_candidate.tolist()))

                        if is_intersect_zigzag((np.array(B2), np.array(B1)),
                                              (np.array(B33[1]), np.array(B0[1]))):
                            B33 = (pid1, tuple(alt_B33_candidate.tolist()))

                        if is_intersect_zigzag((np.array(B2), np.array(B1)),
                                              (np.array(B33[1]), np.array(ptA)), close=True, tol=1e-3):
                            evil_A = True
            elif ptC is not None:
                if is_intersect_zigzag((np.array(ptA), np.array(B1)), (np.array(B2), np.array(ptC))):
                    angles = [angle_between(ptA, cand, B1) for cand in candidates_B0]
                    B0 = (pid1, tuple(candidates_B0[int(np.argmin(angles))].tolist()))

                    angles1 = [angle_between(ptA, B1, cand) for cand in candidates_B3]
                    B33 = (pid1, tuple(candidates_B3[int(np.argmax(angles1))].tolist()))
                    evil_A = True

    return B0, B3, B00, B33, evil_A, evil_C



def build_result_path(B0, B3, B00, B33, pid1, pid2, B1, B2, evil_A, evil_C, mode):
    result = []

    if B33 is not None:
        if mode > 1 and evil_A:
            dis1 = np.linalg.norm(np.array(B0[1]) - np.array(B33[1])) * 0.2
            pt1 = extend_along_direction(np.array(B0[1]), np.array(B33[1]), dis1)
            pt1d5 = extend_along_direction(np.array(B1), np.array(B2), dis1)
            dis2 = np.linalg.norm(pt1d5 - pt1)
            pt2 = extend_along_direction(pt1, pt1d5, dis2 + dis1)
            result.extend([(pid1, tuple(pt2)), (pid1, tuple(pt1))])
        result.append(B33)

    if B0 is None and B3 is None:
        result.extend([(pid1, B1), (pid2, B2)])
    elif B0 is None:
        result.extend([(pid1, B1), (pid2, B2), B3])
    elif B3 is None:
        result.extend([B0, (pid1, B1), (pid2, B2)])
    else:
        result.extend([B0, (pid1, B1), (pid2, B2), B3])

    if B00 is not None:
        if mode <= 1 and evil_C:
            dis1 = np.linalg.norm(np.array(B3[1]) - np.array(B00[1])) * 0.2
            pt1 = extend_along_direction(np.array(B3[1]), np.array(B00[1]), dis1)
            pt1d5 = extend_along_direction(np.array(B2), np.array(B1), dis1)
            dis2 = np.linalg.norm(pt1d5 - pt1)
            pt2 = extend_along_direction(pt1, pt1d5, dis2 + dis1)
            result.extend([(pid2, tuple(pt1)), (pid2, tuple(pt2))])
        else:
            result.append(B00)

    return result



def is_tangent_intersection(intersection, point):
    return (intersection.geom_type == "Point" and
            intersection.distance(Point(point)) < 1e-4)






def angle_between(p1, p2, p3):

    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-8:
        return 0.0
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return np.arccos(cos_theta)



def generate_expanded_path(
    stitch_path,
    stitch_points_world,
    shifted_out_stitch_points,
    is_bridge,
    best_modes,
    offset = 0.2
):

    sp_index_map = {
        (pid, tuple(pt)): idx
        for idx, (pid, pt) in enumerate(stitch_points_world)
    }

    new_path = []
    current_zigzag_order = 1  # 1 for left-right, -1 for right-left
    n = len(stitch_path)
    cnt = -1
    prev_bridge = -0.5

    for i, (pid, pt) in enumerate(stitch_path):
        if pid < 0:
            new_path.append((pid, pt))
            continue

        if i < len(is_bridge) and is_bridge[i] < 0 and is_bridge[i] != prev_bridge:
            prev_bridge = is_bridge[i]
            current_zigzag_order = 1  # Reset to left-right order for new segment
            cnt += 1


        if i < len(is_bridge) and is_bridge[i] == 1:
            new_path.append((pid, pt))
            continue
        key = (int(pid), pt)

        if key in sp_index_map:
            idx = sp_index_map[key]
            original = np.array(pt)
            shifted_out = shifted_out_stitch_points[idx][1]

            direction = shifted_out - original
            perp = np.array([-direction[1], direction[0]])  # Perpendicular vector

            norm = np.linalg.norm(perp)
            if norm > 1e-8:
                perp /= norm
            else:
                perp = np.array([0.0, 0.0])

            left_pt = original + offset * perp
            right_pt = original - offset * perp

            if current_zigzag_order == 1:
                pair = [(pid, tuple(left_pt)), (pid, tuple(right_pt))]
            else:
                pair = [(pid, tuple(right_pt)), (pid, tuple(left_pt))]

            new_path.append(pair)


            if i < len(is_bridge) and is_bridge[i] == prev_bridge:
                if best_modes[cnt] <= 1:
                    if i < len(stitch_path)-1 and stitch_path[i+1][0] % 1 == 0:
                        current_zigzag_order *= -1
                else:
                    if stitch_path[i][0] % 1 == 0:
                        current_zigzag_order *= -1
        else:
            raise ValueError(f"Point not found in stitch points: {key}")

    return new_path







def generate_zigzag_path(area, repeat_map, stitch_points_world, shifted_in_stitch_points, mode= 0, pull_all= True, epsilon=0.002, radius=0.007):

    seqA = area['stitch_sequence']
    seqB = area['corresponding_sequence']



    pt_start = area['splicing_point'][0][1]

    if len(seqA) != len(seqB):
        raise ValueError(f"Area {area['area_id']} stitch/corresponding sequence length are different, cannot do zigzag.")


    raw_path = []
    if mode == 0 or mode == 3:
        for a, b in zip(seqA, seqB):
            raw_path.append(a)
            raw_path.append(b)
    else:
        for a, b in zip(seqA, seqB):
            raw_path.append(b)
            raw_path.append(a)



    accepted_segments = []
    final_path = []
    intersection_seg = [raw_path[0], raw_path[1]]


    if not pull_all and (mode == 0 or mode == 1):
        return generate_zigzag_path_simple(area, mode)

    for i in range(1, len(raw_path)):
        pt_prev = raw_path[i - 1]
        pt_curr = raw_path[i]
        seg = (pt_prev, pt_curr)


        intersect = any(is_intersect_zigzag(seg, s) for s in accepted_segments)



        if ((i==1) or (i>1 and not direction_reroute(i,raw_path, pt_start))) and (not intersect) and (len(intersection_seg) == 2) \
            and (i==1 or not is_intersect_zigzag((pt_prev, pt_curr), (raw_path[0], (0, pt_start)))\
            and not is_point_in_quad((pt_prev, pt_curr), (seqA[0], seqB[0]), (-1, pt_start))
                ):
            accepted_segments.append(seg)
            if not final_path:
                final_path.append(pt_prev)


            final_path.append(pt_curr)


            if i == 1:
                continue
            else:
                SM = (np.array(intersection_seg[0][1]) + np.array(intersection_seg[1][1])) / 2.0
                S0 = np.array(pt_start)
                K1 = np.array(pt_prev[1])
                K2 = np.array(pt_curr[1])

                if is_same_side(K1, K2, S0, SM):
                    intersection_seg = []
                    intersection_seg.append(pt_prev)
                    intersection_seg.append(pt_curr)
                else:
                    continue

        elif (not direction_reroute(i,raw_path, pt_start)) and (not intersect) and (len(intersection_seg) > 2) \
            and (not is_intersect_zigzag((pt_prev, pt_curr), (raw_path[0], (0, pt_start)))
                ):
            accepted_segments.append(seg)
            final_path.append(pt_curr)
            intersection_seg = []
            intersection_seg.append(pt_prev)
            intersection_seg.append(pt_curr)
        else:
            positive_direction_right = xnor(i%2!=0,is_left(np.array(raw_path[0][1]), np.array(raw_path[1][1]), np.array(pt_start)))
            rerouted = reroute_zigzag(positive_direction_right, accepted_segments, pt_start, pt_prev, pt_curr,  \
                                      intersection_seg, stitch_points_world, epsilon=epsilon, radius=radius)

            if len(rerouted) < 2:
                raise ValueError(f"Reroute from {pt_prev} to {pt_curr} failed or invalid.")


            intersection_seg = []
            if rerouted[0] in repeat_map:
                rerouted[0] = repeat_map[rerouted[0]]
            for j in range(1, len(rerouted)):
                reroute_seg = (rerouted[j - 1], rerouted[j])
                accepted_segments.append(reroute_seg)
                intersection_seg.append(rerouted[j-1])
            intersection_seg.append(rerouted[len(rerouted)-1])





            final_path.extend(rerouted[1:])


    final_path = insert_repeat_points(final_path, repeat_map)
    if mode >= 2:
        final_path = final_path[::-1]




    return final_path






def n_points_detour(
    positive_direction_right,
    accepted_segments,
    intersection_points,
    pt_curr,
    pt_start,
    epsilon,
    radius,
    strategy = 'full'
):


    seg = [np.array(pt) for (_, pt) in intersection_points]
    if len(seg) < 2:
        raise ValueError("n_points_detour requires at least 2 intersection points")

    if len(seg) == 2:
        tan_inters = tangent_reroute(pt_curr, seg[1], seg[0], radius)
        for k in range(len(tan_inters)):
            pp = tan_inters[k]
            if positive_direction_right:
                if (not is_left(seg[1], seg[0], pp)) and (not is_left(seg[0], pt_curr, pp)):
                    return [pp]
            else:
                if (is_left(seg[1], seg[0], pp)) and (is_left(seg[0], pt_curr, pp)):
                    return [pp]
        # raise ValueError("len(seg)==2 WRONG!")


    detour_points = []



    if strategy == 'full':
        for i in range(len(seg)-2, -1, -1):
            pivot = seg[i]
            if i==len(seg)-2:
                A = seg[i]
                B = seg[-1]
                C = pt_curr
                BC = C - B
                BC_norm_sq = np.dot(BC, BC)
                if BC_norm_sq < 1e-10:
                    raise ValueError("B and C are too close together")
                proj_len = np.dot(A - B, BC) / BC_norm_sq
                H = B + proj_len * BC
                if np.array_equal(A, H):
                    raise ValueError("A == H!!!!!!!!!!!!!!")
                pt1 = extend_along_direction(H, A, epsilon)

                pt2 = (extend_along_direction(pt_curr, pivot, epsilon*2) +
                          extend_along_direction(seg[-1], pivot, epsilon*2)) / 2.0

                d1 = np.linalg.norm(np.array(pt1) - np.array(pivot))
                d2 = np.linalg.norm(np.array(pt2) - np.array(pivot))
                if d1 > d2:
                    pt = pt1
                else:
                    pt = pt2
                detour_points.append(pt)

            elif is_same_side(seg[i+1],seg[i],seg[-1],pt_curr):
                if i == 0:
                    tan_inters = tangent_reroute(pt_curr, detour_points[-1], seg[0], radius)
                    for k in range(len(tan_inters)):
                        pp = tan_inters[k]
                        seg1 = [detour_points[-1], pp]
                        seg2 = [pt_curr, pp]
                        if any(is_intersect_zigzag(seg1, s) for s in accepted_segments) or \
                            any(is_intersect_zigzag(seg2, s) for s in accepted_segments):
                            continue
                        else:
                            pt = pp
                            break
                    detour_points.append(pt)
                else:
                    pt = (extend_along_direction(pt_curr, pivot, epsilon*2) +
                          extend_along_direction(seg[-1], pivot, epsilon*2)) / 2.0
                    detour_points.append(pt)
            else:
                if i == 0:
                    if is_intersect_zigzag_circle(seg[i], (detour_points[-1], pt_curr), radius=radius):
                        tan_inters = tangent_reroute(pt_curr, detour_points[-1], seg[0], radius)
                        for k in range(len(tan_inters)):
                            pp = tan_inters[k]
                            seg1 = [detour_points[-1], pp]
                            seg2 = [pt_curr, pp]
                            if any(is_intersect_zigzag(seg1, s) for s in accepted_segments) or \
                                any(is_intersect_zigzag(seg2, s) for s in accepted_segments):
                                continue
                            else:
                                pt = pp
                                break
                        detour_points.append(pt)

                else:
                    continue


    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return detour_points



def insert_repeat_points(path, repeat_map):
    new_path = []
    for point in path:
        pid, pt = point
        new_path.append(point)

        if pid >= 0 and (pid, pt) in repeat_map:
            new_path.append(repeat_map[(pid, pt)])

    return new_path




def xnor(a, b):
    return True if a == b else False


def find_tangent_lines(point, circle_center, circle_radius):
    x0, y0 = point
    xc, yc = circle_center
    r = circle_radius

    dx = xc - x0
    dy = yc - y0
    d_squared = dx**2 + dy**2

    if d_squared <= r**2:
        raise ValueError("Point is inside or on the circle, no tangents.")

    a = dx**2 - r**2
    b = -2 * dx * dy
    c = dy**2 - r**2

    discriminant = b**2 - 4 * a * c
    m1 = (-b + np.sqrt(discriminant)) / (2 * a)
    m2 = (-b - np.sqrt(discriminant)) / (2 * a)



    def line(m):
        return lambda x: m * (x - x0) + y0

    tangent1 = line(m1)
    tangent2 = line(m2)

    return m1, m2, tangent1, tangent2



def find_intersection(m1, point1, m2, point2):
    x1, y1 = point1
    x2, y2 = point2

    if m1 == m2:
        return None  # Parallel lines

    x = (m1*x1 - m2*x2 + y2 - y1) / (m1 - m2)
    y = m1 * (x - x1) + y1

    return (x, y)



def plot_tangents_and_circle(A, C, O, r, intersections):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))

    circle = plt.Circle(O, r, fill=False, color='blue', linestyle='-')
    ax.add_patch(circle)

    ax.plot(A[0], A[1], 'ro', label='Point A')
    ax.plot(C[0], C[1], 'go', label='Point C')
    ax.plot(O[0], O[1], 'bo', label='Center O')

    m1_A, m2_A, tan1_A, tan2_A = find_tangent_lines(A, O, r)
    x_vals = np.linspace(O[0] - 2*r, O[0] + 2*r, 100)
    ax.plot(x_vals, tan1_A(x_vals), 'r--', label='Tangent from A (1)')
    ax.plot(x_vals, tan2_A(x_vals), 'r-.', label='Tangent from A (2)')

    m1_C, m2_C, tan1_C, tan2_C = find_tangent_lines(C, O, r)
    ax.plot(x_vals, tan1_C(x_vals), 'g--', label='Tangent from C (1)')
    ax.plot(x_vals, tan2_C(x_vals), 'g-.', label='Tangent from C (2)')

    for i, (x, y) in enumerate(intersections, 1):
        ax.plot(x, y, 'k*', markersize=10, label=f'Intersection {i}')

    ax.set_aspect('equal')
    ax.grid()
    ax.legend()
    plt.title('Tangents from Points to Circle and Their Intersections')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()





def is_on_ray(P, ray_start, ray_end, tol=1e-10):

    P = np.array(P, dtype=np.float64)
    start = np.array(ray_start, dtype=np.float64)
    end = np.array(ray_end, dtype=np.float64)

    vec_P = P - start
    vec_ray = end - start


    cross = np.abs(np.cross(vec_P, vec_ray))
    if cross > tol * np.linalg.norm(vec_ray):
        return False

    t = np.dot(vec_P, vec_ray) / (np.dot(vec_ray, vec_ray) + 1e-15)

    return (t >= -tol) and (np.linalg.norm(vec_P) <= np.linalg.norm(vec_ray) * (t + 1.0 + tol))



def tangent_reroute(A, C, O, r):
    A = np.array(A, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    O = np.array(O, dtype=np.float64)

    eps = 1e-9

    vA = A - O
    dA = np.linalg.norm(vA)
    if dA <= r + eps:
        if dA < eps:
            vA = np.array([1.0, 0.0])
            dA = 1.0
        A = O + vA / dA * (r + 1e-6)

    vC = C - O
    dC = np.linalg.norm(vC)
    if dC <= r + eps:
        if dC < eps:
            vC = np.array([1.0, 0.0])
            dC = 1.0
        C = O + vC / dC * (r + 1e-6)

    A = (float(A[0]), float(A[1]))
    C = (float(C[0]), float(C[1]))
    O = (float(O[0]), float(O[1]))

    m1_A, m2_A, tan1_A, tan2_A = find_tangent_lines(A, O, r)
    m1_C, m2_C, tan1_C, tan2_C = find_tangent_lines(C, O, r)



    def get_tangent_point(point, m, O, r):
        x0, y0 = point
        xc, yc = O
        if np.isinf(m):
            x_tangent = 2*x0 - xc
            y_diff_sq = r**2 - (x_tangent - xc)**2
            if y_diff_sq < 0:
                raise ValueError("No real tangent point")
            y_diff = np.sqrt(y_diff_sq)
            return (x_tangent, yc + y_diff)

        a = 1 + m**2
        b = 2*(m*(y0 - yc) + (x0 - xc))
        c = (x0 - xc)**2 + (y0 - yc)**2 - r**2
        discriminant = max(0, b**2 - 4*a*c)
        t = (-b + np.sqrt(discriminant)) / (2*a + 1e-10)
        return (x0 + t, y0 + m*t)

    A_tan1 = get_tangent_point(A, m1_A, O, r)
    A_tan2 = get_tangent_point(A, m2_A, O, r)
    C_tan1 = get_tangent_point(C, m1_C, O, r)
    C_tan2 = get_tangent_point(C, m2_C, O, r)

    intersections = [
        find_intersection(m1_A, A, m1_C, C),
        find_intersection(m1_A, A, m2_C, C),
        find_intersection(m2_A, A, m1_C, C),
        find_intersection(m2_A, A, m2_C, C)
    ]

    ray_intersections = []
    for P in intersections:
        if P is None:
            continue

        on_A_ray = is_on_ray(P, A, A_tan1) or is_on_ray(P, A, A_tan2)

        on_C_ray = is_on_ray(P, C, C_tan1) or is_on_ray(P, C, C_tan2)

        if on_A_ray and on_C_ray:
            ray_intersections.append(P)

    return ray_intersections





def get_area_endpoint(area, mode):

    path = generate_zigzag_path_simple(area, mode)
    return path[0][1], path[-1][1]




def path_length(areas, order, modes):
    total = 0.0
    for i in range(len(order) - 1):
        _, exit_pt = get_area_endpoint(areas[order[i]], modes[i])
        entry_pt, _ = get_area_endpoint(areas[order[i+1]], modes[i+1])
        total += np.linalg.norm(np.array(exit_pt) - np.array(entry_pt))

    _, exit_last = get_area_endpoint(areas[order[-1]], modes[-1])
    entry_first, _ = get_area_endpoint(areas[order[0]], modes[0])
    total += np.linalg.norm(np.array(exit_last) - np.array(entry_first))

    return total





def simulated_annealing_tsp_areas(areas,
                                   max_iter = 5000,
                                   init_temp = 10.0,
                                   cooling_rate = 0.995):
    n = len(areas)

    current_order = list(range(n))
    current_modes = [random.randint(0, 3) for _ in range(n)]
    current_cost = path_length(areas, current_order, current_modes)

    best_order = current_order[:]
    best_modes = current_modes[:]
    best_cost = current_cost

    T = init_temp

    for step in range(max_iter):
        new_order = current_order[:]
        new_modes = current_modes[:]

        if random.random() < 0.5:
            i, j = random.sample(range(n), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_modes[i], new_modes[j] = new_modes[j], new_modes[i]
        else:
            i = random.randint(0, n - 1)
            new_modes[i] = random.randint(0, 3)

        new_cost = path_length(areas, new_order, new_modes)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_order, current_modes, current_cost = new_order, new_modes, new_cost
            if new_cost < best_cost:
                best_order, best_modes, best_cost = new_order[:], new_modes[:], new_cost

        T *= cooling_rate

    return best_order, best_modes, best_cost





def visualize_area_connections(order, modes, areas, contours_raw, figsize=(8, 8)):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)
    cmap = cm.get_cmap('tab10', len(order))


    for pid, contour in contours_raw.items():
        contour = np.array(contour)
        if len(contour.shape) > 2:
            contour = contour[0]
        if not np.allclose(contour[0], contour[-1], atol=1e-6):
            contour = np.vstack([contour, contour[0]])
        ax.plot(contour[:, 0], contour[:, 1], '-', color='gray', linewidth=0.5)


    for i in range(len(order) - 1):
        idx0, mode0 = order[i], modes[i]
        idx1, mode1 = order[i + 1], modes[i + 1]

        _, exit_pt = get_area_endpoint(areas[idx0], mode0)
        entry_pt, _ = get_area_endpoint(areas[idx1], mode1)

        color = cmap(i % 10)

        ax.annotate(
            '', xy=entry_pt, xytext=exit_pt,
            arrowprops=dict(arrowstyle='->', color=color, lw=1)
        )


        ax.plot(*exit_pt, 'o', color='red', markersize=4)
        ax.plot(*entry_pt, 's', color='blue', markersize=4)

    idx_last, mode_last = order[-1], modes[-1]
    idx_first, mode_first = order[0], modes[0]
    _, exit_last = get_area_endpoint(areas[idx_last], mode_last)
    entry_first, _ = get_area_endpoint(areas[idx_first], mode_first)
    color = cmap((len(order) - 1) % 10)

    ax.annotate(
        '', xy=entry_first, xytext=exit_last,
        arrowprops=dict(arrowstyle='->', color=color, lw=1, linestyle='dashed')
    )
    ax.plot(*exit_last, 'o', color='red', markersize=4)
    ax.plot(*entry_first, 's', color='blue', markersize=4)

    ax.plot(*exit_last, 'o', color='red', markersize=6, label='start')
    ax.plot(*entry_first, 's', color='blue', markersize=6, label='end')

    ax.set_aspect('equal')
    ax.set_title("Area-to-Area Connection Visualization (with TSP loop)")
    ax.legend()
    plt.tight_layout()
    plt.show()






def is_intersection(
    line: LineString,
    combined_edges: MultiLineString,
    contour_polygon: Polygon,
    tol = 1e-6,
    inside_check_samples = 5
) -> bool:
    if not line.intersects(combined_edges):
        return False

    intersection = line.intersection(combined_edges)
    line_endpoints = {tuple(line.coords[0]), tuple(line.coords[-1])}



    def pt_at_endpoint(pt):
        return any(np.linalg.norm(np.array(pt) - np.array(ep)) < tol for ep in line_endpoints)

    if intersection.geom_type == "Point":
        pt = (intersection.x, intersection.y)
        if not pt_at_endpoint(pt):
            return True

    elif intersection.geom_type == "MultiPoint":
        if len(intersection.geoms) > 2:
            return True
        else:
            for p in intersection.geoms:
                pt = (p.x, p.y)
                if not pt_at_endpoint(pt):
                    return True
            for i in range(1, inside_check_samples + 1):
                t = i / (inside_check_samples + 1)
                pt = line.interpolate(t, normalized=True)
                if contour_polygon.covers(pt):
                    return False
                else:
                    return True
    elif intersection.geom_type == "LineString":
        return False

    return False






def merge_patch_meshes(patches, patch_faces, layout):

    V_all = []
    F_all = []
    vertex_map_world = {}
    v_offset = 0

    for pid in sorted(patches.keys()):
        verts = patches[pid]
        faces = patch_faces[pid]
        R, t = layout[pid]['R'], layout[pid]['t']


        verts_transformed = (R @ verts.T).T + t
        V_all.append(verts_transformed)


        faces_global = faces + v_offset
        F_all.append(faces_global)


        for i in range(verts_transformed.shape[0]):
            coord = tuple(verts_transformed[i])
            vertex_map_world[(pid, coord)] = v_offset + i

        v_offset += verts.shape[0]

    V_global = np.vstack(V_all)
    F_global = np.vstack(F_all)

    return V_global, F_global, vertex_map_world






def build_path(
    areas,
    repeat_map,
    path_order,
    area_modes,
    G: 'networkx.Graph',
    V_global,
    merged_contour,
    stitch_points_world,
    shifted_in_stitch_points,
    radius,
    epsilon
):



    contour_edges = []
    points_only = [pt for (_, pt) in merged_contour]
    for i in range(len(points_only) - 1):
        contour_edges.append((np.array(points_only[i]), np.array(points_only[i+1])))
    contour_edges.append((np.array(points_only[-1]), np.array(points_only[0])))

    line_segments = [LineString([tuple(p1), tuple(p2)]) for p1, p2 in contour_edges]
    edge_tree = STRtree(line_segments)
    contour_polygon = Polygon(points_only)

    stitch_path = []
    is_bridge_flags = []
    point_to_index = {tuple(pt): i for i, pt in enumerate(V_global)}

    zigzag_paths = [
        generate_zigzag_path(areas[idx], repeat_map, stitch_points_world, \
                             shifted_in_stitch_points, mode=area_modes[i], epsilon=epsilon, radius=radius)
        for i, idx in enumerate(path_order)
    ]


    all_zigzag_lines = []
    for path in zigzag_paths:
        pts = [pt[1] for pt in path]
        for j in range(1, len(pts)):
            seg = LineString([pts[j - 1], pts[j]])
            all_zigzag_lines.append(seg)
    zigzag_lines = MultiLineString(all_zigzag_lines) if all_zigzag_lines else None

    for i in range(len(path_order)):
        idx = path_order[i]
        mode = area_modes[i]
        area = areas[idx]

        zigzag = zigzag_paths[i]
        for j, pt in enumerate(zigzag):
            stitch_path.append(pt)
            is_bridge_flags.append(-i-1)

        if i < len(path_order) - 1:




            current_path = zigzag_paths[i]
            next_path = zigzag_paths[i + 1]

            exit_pt = current_path[-1][1]

            entry_pt = next_path[0][1]

            line = LineString([tuple(exit_pt), tuple(entry_pt)])
            candidate_edges = edge_tree.query(line)
            combined_edges = MultiLineString([edge_tree.geometries[k] for k in candidate_edges])

            if zigzag_lines:
                all_lines = list(combined_edges.geoms) + list(zigzag_lines.geoms)
                all_lines = [ls for ls in all_lines if not ls.is_empty]
                combined = MultiLineString(all_lines)
            else:
                combined = combined_edges

            intersects = is_intersection(line, combined, contour_polygon)

            if intersects:
                idx_exit = point_to_index[tuple(exit_pt)]
                idx_entry = point_to_index[tuple(entry_pt)]
                path_coords = search_bridge_path(G, V_global, idx_exit, idx_entry)
                path_coords = simplify_path(path_coords, combined, contour_polygon)


                for pt in path_coords[1:-1]:
                    stitch_path.append((-1, tuple(pt)))
                    is_bridge_flags.append(1)
            else:
                pass

    return stitch_path, is_bridge_flags



def build_direct_vertex_connection_edges(
    best_choice,
    adj_graph_2d_world,
    vertex_map_world
):

    connection_edges = []

    for (i, j), idx in best_choice.items():
        edge_pair = adj_graph_2d_world[(i, j)][idx]
        (ptA0, ptA1), (ptB0, ptB1) = edge_pair

        vA0 = vertex_map_world.get((i, tuple(ptA0)))
        vA1 = vertex_map_world.get((i, tuple(ptA1)))
        vB0 = vertex_map_world.get((j, tuple(ptB0)))
        vB1 = vertex_map_world.get((j, tuple(ptB1)))

        if None in (vA0, vA1, vB0, vB1):
            continue

        connection_edges.append((vA0, vB0))
        connection_edges.append((vA1, vB1))

    return connection_edges



def extract_zigzag_lines(zigzag_paths):
    zigzag_lines = []
    for area in zigzag_paths:


        path = area.get('point_seq') or area.get('zigzag_path')
        if not path or len(path) < 2:
            continue

        pts = [pt[1] for pt in path]
        for i in range(1, len(pts)):
            seg = LineString([pts[i - 1], pts[i]])
            zigzag_lines.append(seg)

    return MultiLineString(zigzag_lines)





def build_mesh_graph_with_direct_connections(
    V_global,
    F_global,
    connection_edges,
    contour_edges,
    zigzag_paths
):

    zigzag_lines = extract_zigzag_lines(zigzag_paths)

    G = nx.Graph()
    boundary_weight = 20
    zigzag_penalty = 20.0
    zigzag_tol = 0.5



    def compute_weight(u, v):
        dist = np.linalg.norm(V_global[u] - V_global[v])
        edge = tuple(sorted((u, v)))
        if edge in contour_edges:
            dist *= boundary_weight

        if zigzag_lines:
            seg = LineString([V_global[u], V_global[v]])
            if seg.distance(zigzag_lines) < zigzag_tol:
                dist *= zigzag_penalty

        return max(dist, 1e-6)

    for tri in F_global:
        for i in range(3):
            u = tri[i]
            v = tri[(i + 1) % 3]
            G.add_edge(u, v, weight=compute_weight(u, v))

    for u, v in connection_edges:
        G.add_edge(u, v, weight=compute_weight(u, v))

    return G






def _pt(xy):
    return (float(xy[0]), float(xy[1]))



def _segment_is_safe(a, b,
                     contour_polygon: Polygon,
                     combined_edges: Optional[MultiLineString],
                     safe_margin: float,
                     tol: float) -> bool:
    seg = LineString([a, b])

    if not seg.within(contour_polygon):
        return False

    if safe_margin > 0 and contour_polygon.boundary.distance(seg) + tol < safe_margin:
        return False

    if combined_edges is not None:
        inter = seg.intersection(combined_edges)
        if not inter.is_empty:
            if inter.geom_type == "Point":
                pts = [inter]
            elif inter.geom_type == "MultiPoint":
                pts = list(inter.geoms)
            else:
                return False
            ax, ay = a; bx, by = b
            for p in pts:
                px, py = p.x, p.y
                if not ((abs(px-ax) <= tol and abs(py-ay) <= tol) or
                        (abs(px-bx) <= tol and abs(py-by) <= tol)):
                    return False
    return True



def _build_safe_subgraph(G: 'networkx.Graph',
                         V: np.ndarray,
                         contour_polygon: Polygon,
                         combined_edges: Optional[MultiLineString],
                         safe_margin: float,
                         tol: float) -> 'networkx.Graph':
    is_safe_node = {}
    for u in G.nodes():
        x, y = _pt(V[u])
        inside = Point(x, y).within(contour_polygon)
        far = (contour_polygon.boundary.distance(Point(x, y)) + tol >= safe_margin) if safe_margin > 0 else True
        is_safe_node[u] = bool(inside and far)

    Gs = nx.Graph()
    for u in G.nodes():
        if is_safe_node[u]:
            Gs.add_node(u, **G.nodes[u])

    for u, v, data in G.edges(data=True):
        if not (is_safe_node.get(u, False) and is_safe_node.get(v, False)):
            continue
        a = _pt(V[u]); b = _pt(V[v])
        if _segment_is_safe(a, b, contour_polygon, combined_edges, safe_margin, tol):
            Gs.add_edge(u, v, **data)
    return Gs





def search_bridge_path(G: 'networkx.Graph',
                       V: np.ndarray,
                       start_idx: int,
                       end_idx: int,
                       contour_polygon = None,
                       combined_edges = None,
                       safe_margin = 1e-3,
                       tol = 1e-6,
                       use_astar = True) -> np.ndarray:


    if contour_polygon is None:
        nodes_path = nx.shortest_path(G, source=start_idx, target=end_idx, weight='weight')
        return V[nodes_path]

    Gs = _build_safe_subgraph(G, V, contour_polygon, combined_edges, safe_margin, tol)



    def nearest_safe(u):
        if (u in Gs) and (u in Gs.nodes):
            return u
        ux, uy = _pt(V[u])
        best, best_d2 = None, float('inf')
        for v in Gs.nodes():
            vx, vy = _pt(V[v])
            d2 = (vx-ux)*(vx-ux) + (vy-uy)*(vy-uy)
            if d2 < best_d2:
                best_d2, best = d2, v
        if best is None:
            raise RuntimeError("No safe node available in the safe subgraph.")
        return best

    s = nearest_safe(start_idx)
    t = nearest_safe(end_idx)

    if use_astar:




        def h(u, v=t):
            ux, uy = _pt(V[u]); vx, vy = _pt(V[v])
            return np.hypot(ux - vx, uy - vy)
        try:
            nodes_path = nx.astar_path(Gs, s, t, heuristic=h, weight='weight')
        except nx.NetworkXNoPath:
            nodes_path = nx.shortest_path(Gs, s, t, weight='weight')
    else:
        nodes_path = nx.shortest_path(Gs, s, t, weight='weight')

    return V[nodes_path]




def merged_contour_with_idx(
    merged_contour: List[Tuple[int, Tuple[float, float]]],
    vertex_map_world: Dict[Tuple[int, Tuple[float, float]], int]
) -> Set[Tuple[int, int]]:

    contour_edge_ids = set()
    n = len(merged_contour)

    for i in range(n):
        pid1, pt1 = merged_contour[i]
        pid2, pt2 = merged_contour[(i + 1) % n]

        v1 = vertex_map_world.get((pid1, tuple(pt1)))
        v2 = vertex_map_world.get((pid2, tuple(pt2)))

        if v1 is not None and v2 is not None:
            edge = tuple(sorted((v1, v2)))
            contour_edge_ids.add(edge)
        else:

            pass

    return contour_edge_ids




def simplify_path_once(
    path: List[Tuple[float, float]],
    combined_edges: MultiLineString,
    contour_polygon: Polygon,
    max_lookahead: int,
    tol = 1e-6
) -> List[Tuple[float, float]]:

    simplified = [path[0]]
    n = len(path)
    i = 0
    global_start = path[0]
    global_end = path[-1]

    while i < n - 1:
        found = False
        max_j = min(n - 1, i + max_lookahead)

        for j in reversed(range(i + 1, max_j + 1)):
            line = LineString([path[i], path[j]])

            intersects = is_intersection(line, combined_edges, contour_polygon)


            if not intersects:
                simplified.append(path[j])
                i = j
                found = True
                break

        if not found:
            i += 1
            simplified.append(path[i])

    return simplified





def simplify_path(
    path: List[np.ndarray],
    combined_edges: MultiLineString,
    contour_polygon: Polygon,
    max_lookahead = 3,
    tol = 1e-6,
    max_iter = 10
) -> List[Tuple[float, float]]:

    path = [tuple(pt) for pt in path]
    for _ in range(max_iter):
        new_path = simplify_path_once(path, combined_edges, contour_polygon, max_lookahead)
        if new_path == path:
            break
        path = new_path
    return path



def visualize_merged_mesh_with_path_and_connections(
    V: np.ndarray,
    F: np.ndarray,
    full_path: List[Tuple[int, Tuple[float, float]]],
    figsize=(10, 10),
    show_points=True,
    line_color='blue'
):


    fig, ax = plt.subplots(figsize=figsize)

    patch_polys = [mPolygon(V[face], closed=True) for face in F]
    patch_collection = PatchCollection(
        patch_polys, facecolor='lightgray', edgecolor='black', linewidth=1, alpha=0.6
    )
    ax.add_collection(patch_collection)

    path_coords = np.array([pt for _, pt in full_path])
    ax.plot(path_coords[:, 0], path_coords[:, 1], color=line_color, linewidth=2, label='Stitch Path')

    if show_points:
        ax.scatter(path_coords[:, 0], path_coords[:, 1], color=line_color, s=10)

    ax.plot(path_coords[0, 0], path_coords[0, 1], 'ro', markersize=6, label='Start')
    ax.plot(path_coords[-1, 0], path_coords[-1, 1], 'bs', markersize=6, label='End')

    ax.set_aspect('equal')
    ax.set_title("Merged Mesh with Stitching Path")
    ax.legend()
    plt.axis('off')
    plt.show()



def shorten_segment(p1, p2, shorten_ratio=0.2, which='both'):

    A = np.array(p1)
    B = np.array(p2)
    v = B - A

    if which == 'both':
        new_start = A + shorten_ratio * v
        new_end = A + (1 - shorten_ratio) * v
    elif which == 'start':
        new_start = A + shorten_ratio * v
        new_end = B
    elif which == 'end':
        new_start = A
        new_end = A + (1 - shorten_ratio) * v
    else:
        raise ValueError(f"Invalid 'which' value: {which}. Use 'both', 'start', or 'end'.")

    return tuple(new_start), tuple(new_end)




def export_svg_with_bridge_flags_mm_raw(
    stitch_path_expanded,
    is_bridge: List[int],
    stitch_points_world: List[Tuple[int, Tuple[float, float]]],
    shifted_out_stitch_points: List[Tuple[int, Tuple[float, float]]],
    merged_contour,
    svg_filename = 'stitch_path.svg',
    pdf_filename = None,
    target_width_mm = 130.0,
    target_height_mm = 100.0,
    padding_mm = 5.0,
    stroke_width = '0.1pt',
    vector_stroke_width = '0.1pt',
    contour_stroke_width='0.01pt'
):
    from xml.dom.minidom import Document

    width_mm = target_width_mm
    height_mm = target_height_mm
    padding = padding_mm

    all_points = []
    for item in stitch_path_expanded:
        pts = item if isinstance(item, list) else [item]
        all_points.extend([pt[1] for pt in pts])
    all_points.extend([p[1] for p in stitch_points_world])
    all_points.extend([p[1] for p in shifted_out_stitch_points])

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    scale = min((width_mm - 2 * padding) / bbox_w, (height_mm - 2 * padding) / bbox_h) if bbox_w and bbox_h else 1.0
    offset_x = (width_mm - scale * bbox_w) / 2 - scale * min_x
    offset_y = (height_mm - scale * bbox_h) / 2 + scale * max_y



    def to_svg_point(pt):
        x = pt[0] * scale + offset_x
        y = -pt[1] * scale + offset_y
        return f"{x:.4f}", f"{y:.4f}"

    doc = Document()
    svg = doc.createElement("svg")
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("version", "1.1")
    svg.setAttribute("width", f"{width_mm}mm")
    svg.setAttribute("height", f"{height_mm}mm")
    svg.setAttribute("viewBox", f"0 0 {width_mm} {height_mm}")
    doc.appendChild(svg)



    prev_pt = None
    if stroke_width:
        for i, item in enumerate(stitch_path_expanded):
            pts = item if isinstance(item, list) else [item]
            for j in range(len(pts)):
                pt_curr = pts[j][1]
                if j > 0:
                    x1, y1 = to_svg_point(pts[j - 1][1])
                    x2, y2 = to_svg_point(pt_curr)
                    line = doc.createElement("line")
                    line.setAttribute("x1", x1)
                    line.setAttribute("y1", y1)
                    line.setAttribute("x2", x2)
                    line.setAttribute("y2", y2)
                    line.setAttribute("stroke", "red")
                    line.setAttribute("stroke-width", stroke_width)
                    svg.appendChild(line)

                elif prev_pt is not None:


                    color = "red"

                    x1, y1 = to_svg_point(prev_pt)
                    x2, y2 = to_svg_point(pt_curr)

                    line = doc.createElement("line")
                    line.setAttribute("x1", x1)
                    line.setAttribute("y1", y1)
                    line.setAttribute("x2", x2)
                    line.setAttribute("y2", y2)
                    line.setAttribute("stroke", color)
                    line.setAttribute("stroke-width", stroke_width)
                    svg.appendChild(line)

            prev_pt = pts[-1][1]

    if contour_stroke_width:
        if isinstance(merged_contour[0], tuple) and isinstance(merged_contour[0][1], tuple):
            contour_coords = [pt[1] for pt in merged_contour]
        else:
            contour_coords = merged_contour

        contour_str = " ".join([
            f"{to_svg_point(pt)[0]},{to_svg_point(pt)[1]}" for pt in contour_coords
        ])

        poly = doc.createElement("polyline")
        poly.setAttribute("points", contour_str)
        poly.setAttribute("fill", "none")
        poly.setAttribute("stroke", "red")
        poly.setAttribute("stroke-width", contour_stroke_width)
        poly.setAttribute("stroke-linejoin", "round")
        svg.appendChild(poly)

    if vector_stroke_width and len(stitch_points_world) == len(shifted_out_stitch_points):
        for (pid1, pt1), (pid2, pt2) in zip(stitch_points_world, shifted_out_stitch_points):
            assert pid1 == pid2, f"PID mismatch: {pid1} vs {pid2}"

            x1, y1 = to_svg_point(pt1)
            x2, y2 = to_svg_point(pt2)
            line = doc.createElement('line')
            line.setAttribute('x1', x1)
            line.setAttribute('y1', y1)
            line.setAttribute('x2', x2)
            line.setAttribute('y2', y2)
            line.setAttribute('stroke', 'blue')
            line.setAttribute('stroke-width', vector_stroke_width)
            svg.appendChild(line)



    svg_content = doc.toprettyxml(indent='  ')

    if svg_filename is not None:
        with open(svg_filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)

    if pdf_filename is not None:
        if svg_filename is None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.svg', mode='w', encoding='utf-8') as tmp_svg:
                tmp_svg.write(svg_content)
                tmp_svg_path = tmp_svg.name
        else:
            tmp_svg_path = svg_filename
        drawing = svg2rlg(tmp_svg_path)
        renderPDF.drawToFile(drawing, pdf_filename)

    return svg_filename, pdf_filename




def export_svg_with_bridge_flags_mm(
    stitch_path_expanded,
    is_bridge: List[int],
    stitch_points_world: List[Tuple[int, Tuple[float, float]]],
    shifted_out_stitch_points: List[Tuple[int, Tuple[float, float]]],
    merged_contour,
    svg_filename = 'stitch_path.svg',
    pdf_filename = None,
    target_width_mm = 130.0,
    target_height_mm = 100.0,
    padding_mm = 5.0,
    stroke_width = '0.1pt',
    vector_stroke_width = '0.1pt',
    contour_stroke_width='0.01pt'
):
    from xml.dom.minidom import Document


    def _merge_svg_lines_into_polylines(svg_node, allowed_colors=('red', 'purple'), tol=1e-6):



        def _float(attr):
            try:
                return float(attr)
            except Exception:
                return None



        def _same_style(a, b):

            for key in ('stroke', 'stroke-width', 'stroke-dasharray'):
                if a.getAttribute(key) != b.getAttribute(key):
                    return False
            return True



        def _close(p, q):
            return (abs(p[0] - q[0]) <= tol) and (abs(p[1] - q[1]) <= tol)

        lines = []
        for n in list(svg_node.childNodes):
            if getattr(n, "nodeName", "") != "line":
                continue
            stroke = n.getAttribute("stroke") or ""
            if stroke not in allowed_colors:
                continue
            x1 = _float(n.getAttribute("x1")); y1 = _float(n.getAttribute("y1"))
            x2 = _float(n.getAttribute("x2")); y2 = _float(n.getAttribute("y2"))
            if None in (x1, y1, x2, y2):
                continue
            lines.append(n)

        runs = []
        for ln in lines:
            if not runs:
                runs.append({'style_node': ln, 'elems': [ln]})
                continue
            last_run = runs[-1]
            last_ln = last_run['elems'][-1]

            if not _same_style(last_run['style_node'], ln):
                runs.append({'style_node': ln, 'elems': [ln]})
                continue

            lx2 = _float(last_ln.getAttribute("x2")); ly2 = _float(last_ln.getAttribute("y2"))
            nx1 = _float(ln.getAttribute("x1")); ny1 = _float(ln.getAttribute("y1"))
            if _close((lx2, ly2), (nx1, ny1)):
                last_run['elems'].append(ln)
            else:
                runs.append({'style_node': ln, 'elems': [ln]})

        for run in runs:
            elems = run['elems']
            if len(elems) <= 1:
                continue
            style_src = run['style_node']
            pts = []
            first = elems[0]
            pts.append((_float(first.getAttribute("x1")), _float(first.getAttribute("y1"))))
            for ln in elems:
                pts.append((_float(ln.getAttribute("x2")), _float(ln.getAttribute("y2"))))
            pts_str = " ".join([f"{x:.4f},{y:.4f}" for (x, y) in pts])
            poly = svg_node.ownerDocument.createElement("polyline")
            poly.setAttribute("points", pts_str)
            poly.setAttribute("fill", "none")
            for key in ('stroke', 'stroke-width', 'stroke-dasharray'):
                val = style_src.getAttribute(key)
                if val:
                    poly.setAttribute(key, val)
            poly.setAttribute("stroke-linejoin", "round")
            poly.setAttribute("stroke-linecap", "round")
            svg_node.appendChild(poly)

            for ln in elems:
                try:
                    svg_node.removeChild(ln)
                except Exception:
                    pass

    width_mm = target_width_mm
    height_mm = target_height_mm
    padding = padding_mm

    all_points = []
    for item in stitch_path_expanded:
        pts = item if isinstance(item, list) else [item]
        all_points.extend([pt[1] for pt in pts])
    all_points.extend([p[1] for p in stitch_points_world])
    all_points.extend([p[1] for p in shifted_out_stitch_points])

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    scale = min((width_mm - 2 * padding) / bbox_w, (height_mm - 2 * padding) / bbox_h) if bbox_w and bbox_h else 1.0
    offset_x = (width_mm - scale * bbox_w) / 2 - scale * min_x
    offset_y = (height_mm - scale * bbox_h) / 2 + scale * max_y



    def to_svg_point(pt):
        x = pt[0] * scale + offset_x
        y = -pt[1] * scale + offset_y
        return f"{x:.4f}", f"{y:.4f}"

    doc = Document()
    svg = doc.createElement("svg")
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("version", "1.1")
    svg.setAttribute("width", f"{width_mm}mm")
    svg.setAttribute("height", f"{height_mm}mm")
    svg.setAttribute("viewBox", f"0 0 {width_mm} {height_mm}")
    doc.appendChild(svg)



    border_rect = doc.createElement('rect')
    border_rect.setAttribute('x', '0')
    border_rect.setAttribute('y', '0')
    border_rect.setAttribute('width', f'{target_width_mm}')
    border_rect.setAttribute('height', f'{target_height_mm}')
    border_rect.setAttribute('fill', 'none')
    border_rect.setAttribute('stroke', 'blue')
    border_rect.setAttribute('stroke-width', '0.01pt')
    svg.appendChild(border_rect)

    prev_pt = None
    if stroke_width:
        for i, item in enumerate(stitch_path_expanded):
            pts = item if isinstance(item, list) else [item]
            for j in range(len(pts)):
                pt_curr = pts[j][1]
                if j > 0:
                    x1, y1 = to_svg_point(pts[j - 1][1])
                    x2, y2 = to_svg_point(pt_curr)
                    line = doc.createElement("line")
                    line.setAttribute("x1", x1)
                    line.setAttribute("y1", y1)
                    line.setAttribute("x2", x2)
                    line.setAttribute("y2", y2)
                    line.setAttribute("stroke", "red")
                    line.setAttribute("stroke-width", stroke_width)
                    svg.appendChild(line)

                elif prev_pt is not None:


                    color = "red"
                    x1, y1 = to_svg_point(prev_pt)
                    x2, y2 = to_svg_point(pt_curr)
                    line = doc.createElement("line")
                    line.setAttribute("x1", x1)
                    line.setAttribute("y1", y1)
                    line.setAttribute("x2", x2)
                    line.setAttribute("y2", y2)
                    line.setAttribute("stroke", color)
                    line.setAttribute("stroke-width", stroke_width)
                    svg.appendChild(line)

            prev_pt = pts[-1][1]

    _merge_svg_lines_into_polylines(svg_node=svg, allowed_colors=('red', 'purple'), tol=1e-6)

    if contour_stroke_width:
        if isinstance(merged_contour[0], tuple) and isinstance(merged_contour[0][1], tuple):
            contour_coords = [pt[1] for pt in merged_contour]
        else:
            contour_coords = merged_contour

        contour_str = " ".join([
            f"{to_svg_point(pt)[0]},{to_svg_point(pt)[1]}" for pt in contour_coords
        ])

        poly = doc.createElement("polyline")
        poly.setAttribute("points", contour_str)
        poly.setAttribute("fill", "none")
        poly.setAttribute("stroke", "red")
        poly.setAttribute("stroke-width", contour_stroke_width)
        poly.setAttribute("stroke-linejoin", "round")
        svg.appendChild(poly)

    if vector_stroke_width and len(stitch_points_world) == len(shifted_out_stitch_points):
        for (pid1, pt1), (pid2, pt2) in zip(stitch_points_world, shifted_out_stitch_points):
            assert pid1 == pid2, f"PID mismatch: {pid1} vs {pid2}"

            x1, y1 = to_svg_point(pt1)
            x2, y2 = to_svg_point(pt2)
            line = doc.createElement('line')
            line.setAttribute('x1', x1)
            line.setAttribute('y1', y1)
            line.setAttribute('x2', x2)
            line.setAttribute('y2', y2)
            line.setAttribute('stroke', 'blue')
            line.setAttribute('stroke-width', vector_stroke_width)
            svg.appendChild(line)

    svg_content = doc.toprettyxml(indent='  ')

    if svg_filename is not None:
        with open(svg_filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)

    if pdf_filename is not None:
        if svg_filename is None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.svg', mode='w', encoding='utf-8') as tmp_svg:
                tmp_svg.write(svg_content)
                tmp_svg_path = tmp_svg.name
        else:
            tmp_svg_path = svg_filename
        drawing = svg2rlg(tmp_svg_path)
        renderPDF.drawToFile(drawing, pdf_filename)

    return svg_filename, pdf_filename



def sort_stitch_and_shifted_by_path(stitch_path, stitch_points_world, shifted_in_stitch_points, shifted_out_stitch_points, tol=1e-6):
    sorted_stitch = []
    sorted_shifted_in = []
    sorted_shifted_out = []

    for pid_path, coord_path in stitch_path:
        if pid_path < 0:
            continue
        for i, (pid_world, coord_world) in enumerate(stitch_points_world):
            if int(pid_path) == pid_world and np.allclose(
                np.asarray(coord_path).flatten(), np.asarray(coord_world).flatten(), atol=tol
            ):
                sorted_stitch.append((pid_world, coord_world))
                sorted_shifted_in.append(shifted_in_stitch_points[i])
                sorted_shifted_out.append(shifted_out_stitch_points[i])
                break

    return sorted_stitch, sorted_shifted_in, sorted_shifted_out




def sort_stitch_and_shifted_by_merged_contour(merged_contour, stitch_points_world, shifted_in_stitch_points, shifted_out_stitch_points, tol=1e-4):
    sorted_stitch = []
    sorted_shifted_in = []
    sorted_shifted_out = []

    for pid_path, coord_path in merged_contour:
        if pid_path < 0:
            continue
        for i, (pid_world, coord_world) in enumerate(stitch_points_world):
            if int(pid_path) == pid_world and np.allclose(
                np.asarray(coord_path).flatten(), np.asarray(coord_world).flatten(), atol=tol
            ):
                sorted_stitch.append((pid_world, coord_world))
                sorted_shifted_in.append(shifted_in_stitch_points[i])
                sorted_shifted_out.append(shifted_out_stitch_points[i])
                break

    return sorted_stitch, sorted_shifted_in, sorted_shifted_out



# zigzag_paths = [
#     {
#         'area_id': area['area_id'],
#         'point_seq': generate_zigzag_path(area, repeat_map, stitch_points_world, shifted_in_stitch_points, mode=1, pull_all=True, epsilon=0.002, radius=0.004)
#     }
#     for area in areas
# ]




# best_order, best_modes, best_cost = simulated_annealing_tsp_areas(areas)

# visualize_area_connections(best_order, best_modes, areas, contours_raw_world)
# print("Area mode:", best_modes)



# V_global, F_global, vertex_map_world = merge_patch_meshes(patches, patch_faces, best_layout)
# contour_edges = merged_contour_with_idx(
#     merged_contour, vertex_map_world
# )
# connection_edges = build_direct_vertex_connection_edges(
#     best_choice, adj_graph_2d_world, vertex_map_world
# )

# G = build_mesh_graph_with_direct_connections(
#     V_global, F_global, connection_edges, contour_edges, zigzag_paths
# )


# stitch_path, is_bridge = build_path(
#     areas,
#     repeat_map,
#     best_order,
#     best_modes,
#     G,
#     V_global,
#     merged_contour,
#     stitch_points_world,
#     shifted_in_stitch_points,
#     radius=0.0022,
#     epsilon=0.0027
# )




# cleaned_path = []
# cleaned_flags = []

# for i, (pid, pt) in enumerate(stitch_path):
#     if not cleaned_path or pt != cleaned_path[-1][1]:
#         cleaned_path.append((pid, pt))
#         cleaned_flags.append(is_bridge[i])

# stitch_path = cleaned_path
# is_bridge = cleaned_flags
# print(len(stitch_points_world))
# print(len(shifted_out_stitch_points))
# stitch_points_world, shifted_in_stitch_points, shifted_out_stitch_points = sort_stitch_and_shifted_by_merged_contour(merged_contour, stitch_points_world, shifted_in_stitch_points, shifted_out_stitch_points)
# print(len(shifted_in_stitch_points))



# expanded_path = generate_expanded_path(
#     stitch_path,
#     stitch_points_world,
#     shifted_out_stitch_points,
#     is_bridge,
#     best_modes,
#     offset=0.0012
# )
# expanded_path = generate_expanded_path_2(
#     stitch_path,
#     expanded_path,
#     stitch_points_world,
#     shifted_in_stitch_points,
#     is_bridge,
#     best_modes,
#     offset=0.0012
# )



# expanded_path = [pt for item in expanded_path for pt in (item if isinstance(item, list) else [item])]
# expanded_path[248] = (-1, (-0.013394357295223364, 0.06967154283114134))
# expanded_path[191] = (-1, (0.001067, 0.032664))
# del expanded_path[183]
# expanded_path[183] = (1, (0.03644244530435348, -0.007036412365219204))
# expanded_path.insert(183, (-1, (0.02029921921321714147, -0.005217245141294878)))
# del expanded_path[83]
# del expanded_path[83]
# expanded_path[65] = (-1, (0.19745549881169, -0.02922218295229))

# expanded_path.insert(39, (0, (0.2093221643966633, -0.03309929529509851)))
# expanded_path.insert(36, (0, (0.2107221643966633, -0.03309929529509851)))
# expanded_path[236:243] = expanded_path[236:243][::-1]



# export_svg_with_bridge_flags_mm(
#     expanded_path,
#     is_bridge,
#     stitch_points_world=stitch_points_world,
#     shifted_out_stitch_points=shifted_out_stitch_points,
#     merged_contour=merged_contour,
#     svg_filename=None,#'tendon.svg',
#     pdf_filename='Kit/contour_kitten.pdf',
#     target_width_mm=400.0,
#     target_height_mm=210.0,
#     padding_mm=3,
#     stroke_width=None,#'0.1pt',
#     vector_stroke_width=None,#'0.1pt',
#     contour_stroke_width='0.001pt',
# )

# export_svg_with_bridge_flags_mm(
#     expanded_path,
#     is_bridge,
#     stitch_points_world=stitch_points_world,
#     shifted_out_stitch_points=shifted_out_stitch_points,
#     merged_contour=merged_contour,
#     svg_filename=None,
#     pdf_filename='Kit/tendon_kitten.pdf',
#     target_width_mm=400.0,
#     target_height_mm=210.0,
#     padding_mm=3,
#     stroke_width='0.01pt',
#     vector_stroke_width=None,
#     contour_stroke_width=None,
# )

# export_svg_with_bridge_flags_mm_raw(
#     expanded_path,
#     is_bridge,
#     stitch_points_world=stitch_points_world,
#     shifted_out_stitch_points=shifted_out_stitch_points,
#     merged_contour=merged_contour,
#     svg_filename='Kit/vector_kitten.svg',
#     pdf_filename=None,
#     target_width_mm=400.0,
#     target_height_mm=210.0,
#     padding_mm=3,
#     stroke_width='0.01pt',
#     vector_stroke_width='0.01pt',
#     contour_stroke_width=None,
# )

# visualize_merged_mesh_with_path_and_connections(
#     V_global,
#     F_global,
#     expanded_path
# )



def _dedup_consecutive_points(stitch_path, is_bridge):
    cleaned_path = []
    cleaned_flags = []
    for i, (pid, pt) in enumerate(stitch_path):
        if not cleaned_path or pt != cleaned_path[-1][1]:
            cleaned_path.append((pid, pt))
            cleaned_flags.append(is_bridge[i])
    return cleaned_path, cleaned_flags


def run_routing(
    areas,
    repeat_map,
    best_layout,
    patches,
    patch_faces,
    merged_contour,
    stitch_points_world,
    shifted_in_stitch_points,
    shifted_out_stitch_points,
    best_choice,
    adj_graph_2d_world,
    radius,
    epsilon,
    offset=3.0,
    tsp_sa=True,
    best_order=None,
    best_modes=None,
    export_svg=None,
    export_pdf=None,
    export_w_mm=400.0,
    export_h_mm=210.0,
    export_pad_mm=3.0,
    stroke_width=None,
    vector_stroke_width=None,
    contour_stroke_width=None,
    manual_edits=None,
):
    if tsp_sa:
        best_order, best_modes, best_cost = simulated_annealing_tsp_areas(areas)
    else:
        if best_order is None or best_modes is None:
            raise ValueError("tsp_sa=False requires best_order and best_modes")

    zigzag_paths = [
        {
            "area_id": area["area_id"],
            "point_seq": generate_zigzag_path(
                area,
                repeat_map,
                stitch_points_world,
                shifted_in_stitch_points,
                mode=1,
                pull_all=True,
                epsilon=epsilon,
                radius=radius,
            ),
        }
        for area in areas
    ]

    V_global, F_global, vertex_map_world = merge_patch_meshes(patches, patch_faces, best_layout)
    contour_edges = merged_contour_with_idx(merged_contour, vertex_map_world)
    connection_edges = build_direct_vertex_connection_edges(best_choice, adj_graph_2d_world, vertex_map_world)

    G = build_mesh_graph_with_direct_connections(
        V_global, F_global, connection_edges, contour_edges, zigzag_paths
    )

    stitch_path, is_bridge = build_path(
        areas,
        repeat_map,
        best_order,
        best_modes,
        G,
        V_global,
        merged_contour,
        stitch_points_world,
        shifted_in_stitch_points,
        radius=radius,
        epsilon=epsilon,
    )

    stitch_path, is_bridge = _dedup_consecutive_points(stitch_path, is_bridge)

    stitch_points_world, shifted_in_stitch_points, shifted_out_stitch_points = sort_stitch_and_shifted_by_merged_contour(
        merged_contour,
        stitch_points_world,
        shifted_in_stitch_points,
        shifted_out_stitch_points,
    )

    expanded_path = generate_expanded_path(
        stitch_path,
        stitch_points_world,
        shifted_out_stitch_points,
        is_bridge,
        best_modes,
        offset=offset,
    )
    expanded_path = generate_expanded_path_2(
        stitch_path,
        expanded_path,
        stitch_points_world,
        shifted_in_stitch_points,
        is_bridge,
        best_modes,
        offset=offset,
    )

    expanded_path = [pt for item in expanded_path for pt in (item if isinstance(item, list) else [item])]

    if manual_edits is not None:
        for op in manual_edits:
            t = op.get("type")
            if t == "set":
                expanded_path[op["idx"]] = op["value"]
            elif t == "del":
                del expanded_path[op["idx"]]
            elif t == "insert":
                expanded_path.insert(op["idx"], op["value"])
            elif t == "reverse_slice":
                a, b = op["a"], op["b"]
                expanded_path[a:b] = expanded_path[a:b][::-1]

    if export_svg is not None or export_pdf is not None:
        export_svg_with_bridge_flags_mm(
            expanded_path,
            is_bridge,
            stitch_points_world=stitch_points_world,
            shifted_out_stitch_points=shifted_out_stitch_points,
            merged_contour=merged_contour,
            svg_filename=export_svg,
            pdf_filename=export_pdf,
            target_width_mm=export_w_mm,
            target_height_mm=export_h_mm,
            padding_mm=export_pad_mm,
            stroke_width=stroke_width,
            vector_stroke_width=vector_stroke_width,
            contour_stroke_width=contour_stroke_width,
        )

    return {
        "best_order": best_order,
        "best_modes": best_modes,
        "stitch_path": stitch_path,
        "is_bridge": is_bridge,
        "expanded_path": expanded_path,
    }
