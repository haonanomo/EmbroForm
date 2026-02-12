import argparse
from pathlib import Path

from embroform import DATA_DIR
from embroform.src.layout import build_layout
from embroform.src.pointset import (
    build_pointset,
    export_svg_with_stitch_vectors_mm,
)
from embroform.src.preprocess import preprocess
from embroform.src.splice import run_splicing
from embroform.src.path import run_routing


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--obj_name", type=str, default="Bir.obj")
    ap.add_argument("--radius", type=float, default=0.05)

    ap.add_argument("--export_pointset", default=True)
    ap.add_argument("--pointset_w_mm", type=float, default=410.0)
    ap.add_argument("--pointset_h_mm", type=float, default=200.0)
    ap.add_argument("--pointset_pad_mm", type=float, default=5.0)

    ap.add_argument("--export_path", default=True)
    ap.add_argument("--path_w_mm", type=float, default=400.0)
    ap.add_argument("--path_h_mm", type=float, default=210.0)
    ap.add_argument("--path_pad_mm", type=float, default=3.0)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--offset", type=float, default=0.05)

    return ap.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    radius = args.radius

    # STEP 0: preprocess input mesh
    preprocess(root_dir=root_dir, obj_name=args.obj_name)

    # STEP 1: layout optimization
    best_layout, best_choice, adj_graph_3d, adj_graph_2d, G, patches, patch_faces, contours_raw = build_layout(root_dir=root_dir)

    # STEP 2: build pointset
    (
        merged_contour,
        stitch_points_world,
        corr_world,
        shifted_in,
        shifted_out,
        contours,
        contours_world,
        adj_graph_2d_world,
    ) = build_pointset(
        radius=radius,
        best_layout=best_layout,
        patches=patches,
        patch_faces=patch_faces,
        contours_raw=contours_raw,
        adj_graph_2d=adj_graph_2d,
        best_choice=best_choice,
    )

    if args.export_pointset:
        out_dir = root_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        svg2 = str(out_dir / "pointset.svg")
        # png2 = str(out_dir / "pointset.png")

        export_svg_with_stitch_vectors_mm(
            merged_contour=merged_contour,
            stitch_points_world=stitch_points_world,
            shifted_in_stitch_points=shifted_in,
            shifted_out_stitch_points=shifted_out,
            svg_filename=svg2,
            target_width_mm=args.pointset_w_mm,
            target_height_mm=args.pointset_h_mm,
            padding_mm=args.pointset_pad_mm,
            stitch_radius_world=radius,
        )

        # export_pointset_debug_png(
        #     merged_contour=merged_contour,
        #     stitch_points_world=stitch_points_world,
        #     shifted_in_stitch_points=shifted_in,
        #     shifted_out_stitch_points=shifted_out,
        #     png_filename=png_dbg,
        # )

        print("[OK] wrote:", svg2)


    # STEP 3: splicing area extraction
    svg3 = str(out_dir / "splicing_areas.svg")

    areas, repeat_map = run_splicing(
        best_choice=best_choice,
        adj_graph_2d_world=adj_graph_2d_world,
        merged_contour=merged_contour,
        correspondence_map_world=corr_world,
        radius=radius,
        filter_close_threshold=radius*2,
        svg3=svg3,
        show_labels=True,
    )


    # STEP 4: routing / build tendon path
    if args.export_pointset:
        out_dir = root_dir
    else:
        out_dir = root_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_path:
        svg4 = str(out_dir / "tendon.svg")
        pdf4 = str(out_dir / "tendon.pdf")
        raw_svg4 = str(out_dir / "tendon_raw.svg")
    else:
        svg4 = None
        pdf4 = None
        raw_svg4 = None

    ret = run_routing(
        areas=areas,
        repeat_map=repeat_map,
        best_layout=best_layout,
        patches=patches,
        patch_faces=patch_faces,
        merged_contour=merged_contour,
        stitch_points_world=stitch_points_world,
        shifted_in_stitch_points=shifted_in,
        shifted_out_stitch_points=shifted_out,
        best_choice=best_choice,
        adj_graph_2d_world=adj_graph_2d_world,
        radius=radius,
        epsilon=args.epsilon,
        offset=args.offset,
        export_svg=svg4,
        export_pdf=pdf4,
        export_w_mm=args.path_w_mm,
        export_h_mm=args.path_h_mm,
        export_pad_mm=args.path_pad_mm,
        stroke_width="0.01pt" if args.export_path else None,
        vector_stroke_width=None,
        contour_stroke_width=None,
    )

    if args.export_path:
        print("[OK] wrote:", svg4)
        print("[OK] wrote:", pdf4)



if __name__ == "__main__":
    main()
