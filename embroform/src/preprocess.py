import os
import subprocess
from pathlib import Path
from collections import defaultdict
from embroform import PROJ_DIR
import numpy as np
import trimesh


def _run(command):
    command = [str(x) for x in command]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (code={process.returncode}): {command}")


def develop_mesh(root_dir, obj_name="Bird.obj"):
    root_dir = Path(root_dir)

    exe_evo = PROJ_DIR / "EvoDevelop.exe"
    exe_dev = PROJ_DIR / "DevelopApp.exe"
    if not exe_evo.exists():
        raise FileNotFoundError(f"Missing exe: {exe_evo}")
    if not exe_dev.exists():
        raise FileNotFoundError(f"Missing exe: {exe_dev}")

    input_mesh = root_dir / obj_name
    output_folder = root_dir / obj_name.replace(".obj", "Out")

    _run([exe_evo, input_mesh, output_folder, "0.025", "1000"])

    output_mesh = output_folder / "output_mesh.obj"
    output_seg = output_folder / "output_seg.txt"
    _run([exe_dev, output_mesh, output_seg])

    return output_folder


def extract_patches(root_dir):
    root_dir = Path(root_dir)

    mesh = trimesh.load(root_dir / "final.obj", process=False)
    face_segments = np.loadtxt(root_dir / "final_seg.txt", dtype=int)

    if len(face_segments) != len(mesh.faces):
        raise ValueError("final_seg.txt and final.obj faces length mismatch")

    output_folder = root_dir / "patches"
    os.makedirs(output_folder, exist_ok=True)

    patch_to_faces = defaultdict(list)
    for face_idx, patch_id in enumerate(face_segments):
        patch_to_faces[int(patch_id)].append(face_idx)

    global_vertices = mesh.vertices

    for patch_id, face_indices in patch_to_faces.items():
        face_global_indices = mesh.faces[face_indices]
        unique_global_ids = np.unique(face_global_indices)

        patch_filename = output_folder / f"patch_{patch_id}.obj"
        vmap_filename = output_folder / f"patch_{patch_id}_vmap.txt"

        with open(patch_filename, "w") as f:
            for gid in unique_global_ids:
                x, y, z = global_vertices[gid]
                f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")

            for face in face_global_indices:
                face_local = [np.where(unique_global_ids == vid)[0][0] + 1 for vid in face]
                f.write(f"f {face_local[0]} {face_local[1]} {face_local[2]}\n")

        np.savetxt(vmap_filename, unique_global_ids, fmt="%d")
        print(f"Saved: {patch_filename} + {vmap_filename}")

    return output_folder


def flatten_patches(root_dir):
    root_dir = Path(root_dir)

    exe_pp = PROJ_DIR / "PP.exe"
    if not exe_pp.exists():
        raise FileNotFoundError(f"Missing exe: {exe_pp}")

    input_folder = root_dir / "patches"
    patch_files = sorted(input_folder.glob("patch_*.obj"))

    for patch_file in patch_files:
        mesh_name = "patches/" + patch_file.name
        print(f"\n=== Running on {mesh_name} ===")
        _run([exe_pp, root_dir, mesh_name])

    return root_dir / "patches_pp"


def preprocess(root_dir, obj_name="Bird.obj"):
    develop_mesh(root_dir, obj_name)
    extract_patches(root_dir)
    flatten_patches(root_dir)