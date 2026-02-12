bl_info = {
    "name": "Research Design Tool — One Panel (5 Buttons)",
    "author": "You",
    "version": (0, 4, 0),
    "blender": (3, 0, 0),
    "location": "3D Viewport > N‑panel > EmbroForm",
    "description": "Single panel with five actions: Import Mesh, Deform & Segment, Develop & 2D Packing, Mark Points, Construct Routing Path.",
    "category": "3D View",
}

import bpy
import os
import sys
import math
import time  

class RDT_Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    project_dir: bpy.props.StringProperty(
        name="Project Folder", subtype='DIR_PATH',
        description="(Optional) Folder with your Python modules", default=""
    )
    output_dir: bpy.props.StringProperty(
        name="Results Folder", subtype='DIR_PATH',
        description="Root folder containing Results/import, Results/segment, Results/packing", default=""
    )
    add_to_sys_path: bpy.props.BoolProperty(
        name="Add project folder to sys.path", default=True,
        description="Append project folder to sys.path so you can import your modules",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "project_dir")
        layout.prop(self, "output_dir")
        layout.prop(self, "add_to_sys_path")

def _maybe_add_sys_path():
    try:
        prefs = bpy.context.preferences.addons[__name__].preferences  # type: ignore
        if prefs.add_to_sys_path and prefs.project_dir and os.path.isdir(prefs.project_dir):
            if prefs.project_dir not in sys.path:
                sys.path.append(prefs.project_dir)
    except Exception:
        pass


class RDT_SceneProps(bpy.types.PropertyGroup):
    collection_name: bpy.props.StringProperty(
        name="Target Collection", default="RDT_Results",
        description="Imported meshes will be placed here",
    )
    clear_collection: bpy.props.BoolProperty(
        name="Clear target collection before import", default=True,
    )
    color_by_patch: bpy.props.BoolProperty(
        name="Auto color by file index", default=True,
    )
    marker_radius: bpy.props.FloatProperty(
        name="Point Marker Radius",
        default=0.05, min=0.0001, soft_max=0.1,
        description="Radius for small circle markers (Blender units)",
    )
    routing_thickness: bpy.props.FloatProperty(
        name="Routing Thickness",
        default=0.006, min=0.0, soft_max=0.05,
        description="Curve bevel depth for routing path",
    )
    separation_distance: bpy.props.FloatProperty(
        name="Separation Distance",
        default=5.0, min=0.0, soft_max=10.0,
        description="Distance to separate segment and packing meshes along X-axis",
    )



def make_collection(name: str) -> bpy.types.Collection:
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def clear_collection(col: bpy.types.Collection):
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def get_or_create_material(name: str, color=(0.8, 0.8, 0.8, 1.0)) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs[0].default_value = color
    return mat


def material_for_index(idx: int) -> bpy.types.Material:
    import colorsys
    h = (idx * 0.1618033) % 1.0
    s, v = 0.7, 0.9
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return get_or_create_material(f"RDT_Mat_{idx:03d}", (r, g, b, 1))


def import_objs_from_subdir(subdir: str, offset_x: float = 0.0) -> int:
    prefs = bpy.context.preferences.addons[__name__].preferences  # type: ignore
    scn = bpy.context.scene.rdt

    base = prefs.output_dir or ""
    directory = os.path.join(base, subdir)

    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError(f"Folder not found: {directory}")

    col = make_collection(scn.collection_name)

    imported = 0
    for i, fname in enumerate(sorted(os.listdir(directory))):
        if not fname.lower().endswith(".obj"):
            continue
        path = os.path.join(directory, fname)
        prev_objs = set(bpy.data.objects)
        try:
            if hasattr(bpy.ops.wm, 'obj_import'):
                bpy.ops.wm.obj_import(filepath=path)
            else:
                bpy.ops.import_scene.obj(filepath=path)
        except Exception as e:
            print(f"[RDT] Failed to import {path}: {e}")
            continue

        new_objs = [o for o in bpy.data.objects if o not in prev_objs]
        for obj in new_objs:
            base_name = os.path.splitext(os.path.basename(path))[0]
            obj.name = base_name
            if obj.name not in col.objects:
                col.objects.link(obj)
            try:
                bpy.context.scene.collection.objects.unlink(obj)
            except Exception:
                pass
            
            # Apply X offset
            if offset_x != 0.0:
                obj.location.x += offset_x
            
            imported += 1
    return imported


def _ensure_subcollection(root: bpy.types.Collection, sub: str) -> bpy.types.Collection:
    if sub in bpy.data.collections:
        col = bpy.data.collections[sub]
        if col.name not in [c.name for c in root.children]:
            root.children.link(col)
        return col
    col = bpy.data.collections.new(sub)
    root.children.link(col)
    return col


def parse_point_line(line: str):
    # Accept formats: "x y z", "x,y,z", or "x y" (z=0)
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    if "," in line:
        parts = [p for p in line.replace("	", ",").split(",") if p]
    else:
        parts = line.split()
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        x, y = nums; z = 0.0
    elif len(nums) >= 3:
        x, y, z = nums[:3]
    else:
        return None
    return (x, y, z)


def load_points_from_folder(folder: str):
    """Return dict: {name:[(x,y,z), ...]} for all .txt under folder."""
    data = {}
    if not os.path.isdir(folder):
        return data
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.txt'):
            continue
        path = os.path.join(folder, fname)
        pts = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                pt = parse_point_line(line)
                if pt is not None:
                    pts.append(pt)
        if pts:
            key = os.path.splitext(fname)[0]
            data[key] = pts
    return data


def create_circle_marker(location, radius, name):
    # Create a mesh circle in XZ plane
    bpy.ops.mesh.primitive_circle_add(
        vertices=32, 
        radius=radius, 
        fill_type='NGON',
        location=location,
        rotation=(0, 0, 0)
    )
    obj = bpy.context.active_object
    obj.name = name
    
    mat = get_or_create_material("RDT_Point", (1.0, 0.0, 0.0, 1.0))  # Pure red
    
    # Assign material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return obj


def create_polyline_curve(points, name, bevel_depth=0.0):
    curve_data = bpy.data.curves.new(name=name+"_curve", type='CURVE')
    curve_data.dimensions = '3D'
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(points)-1)
    for i, (x, y, z) in enumerate(points):
        spline.points[i].co = (x, y, z, 1.0)
    curve_data.bevel_depth = bevel_depth
    curve_obj = bpy.data.objects.new(name, curve_data)
    
    # Create bright purple material for better visibility
    mat = get_or_create_material("RDT_Routing", (0.8, 0.2, 1.0, 1.0))  # Bright purple
    curve_obj.data.materials.append(mat)
    
    return curve_obj

class RDT_OT_import_mesh(bpy.types.Operator):
    bl_idname = "rdt.import_mesh"
    bl_label = "Import Mesh"

    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene.rdt
        col = make_collection(scn.collection_name)
        if scn.clear_collection:
            clear_collection(col) 
        try:
            n = import_objs_from_subdir("import")

            time.sleep(2)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}

class RDT_OT_import_patches_pp(bpy.types.Operator):
    bl_idname = "rdt.import_patches_pp"
    bl_label = "Import Patches PP"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene.rdt
        col = make_collection(scn.collection_name)
        if scn.clear_collection:
            clear_collection(col)
        
        separation = scn.separation_distance
        
        try:
            # Import segment meshes with negative X offset
            n_segment = import_objs_from_subdir("segment", offset_x=-separation/4)
            
            # Import patches_pp meshes with positive X offset
            n_patches_pp = import_objs_from_subdir("patches_pp", offset_x=separation/2)
            
            time.sleep(2)
            
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        return {'FINISHED'}
    

class RDT_OT_deform_segment_and_packing(bpy.types.Operator):
    bl_idname = "rdt.deform_segment_and_packing"
    bl_label = "Deform, Segment & Packing"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene.rdt
        col = make_collection(scn.collection_name)
        if scn.clear_collection:
            clear_collection(col)
        separation = scn.separation_distance
        
        try:
            # Import segment meshes with negative X offset
            n_segment = import_objs_from_subdir("segment", offset_x=-separation/4)
            
            # Import packing meshes with positive X offset
            n_packing = import_objs_from_subdir("packing", offset_x=separation/2)

            time.sleep(2)
            
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
    
        return {'FINISHED'}


class RDT_OT_mark_points(bpy.types.Operator):
    bl_idname = "rdt.mark_points"
    bl_label = "Mark Points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene.rdt
        col = make_collection(scn.collection_name)
        if scn.clear_collection:
            clear_collection(col)
        prefs = bpy.context.preferences.addons[__name__].preferences  # type: ignore
        root_col = make_collection(scn.collection_name)
        points_col = _ensure_subcollection(root_col, "RDT_Points")

        # Ensure packing meshes are present (also respects Clear Before Import option)
        try:
            import_objs_from_subdir("packing")
            
            time.sleep(2)
            
        except Exception as e:
            self.report({'ERROR'}, f"Packing import failed: {e}")
            return {'CANCELLED'}

        # Load points and create circle markers
        points_dir = os.path.join(prefs.output_dir or "", "points")
        datasets = load_points_from_folder(points_dir)
        if not datasets:
            self.report({'WARNING'}, f"No point .txt files found in {points_dir}")
        # Clear previous markers in subcollection
        clear_collection(points_col)

        count = 0
        for name, pts in datasets.items():
            for j, pt in enumerate(pts):
                obj = create_circle_marker(pt, scn.marker_radius, f"pt_{name}_{j:04d}")
                if obj.name not in points_col.objects:
                    points_col.objects.link(obj)
                try:
                    bpy.context.scene.collection.objects.unlink(obj)
                except Exception:
                    pass
                count += 1
        
        time.sleep(2)
        
        return {'FINISHED'}


class RDT_OT_construct_routing(bpy.types.Operator):
    bl_idname = "rdt.construct_routing"
    bl_label = "Construct Routing Path"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        time.sleep(2)
        prefs = bpy.context.preferences.addons[__name__].preferences  # type: ignore
        scn = context.scene.rdt
        root_col = make_collection(scn.collection_name)
        routing_col = _ensure_subcollection(root_col, "RDT_Routing")

        routing_dir = os.path.join(prefs.output_dir or "", "routing")
        datasets = load_points_from_folder(routing_dir)
        if not datasets:
            self.report({'WARNING'}, f"No routing .txt files found in {routing_dir}")
            return {'CANCELLED'}

        # Clear previous routing objects
        clear_collection(routing_col)

        made = 0
        for name, pts in datasets.items():
            if len(pts) < 2:
                continue
            curve_obj = create_polyline_curve(pts, f"route_{name}", bevel_depth=scn.routing_thickness)
            routing_col.objects.link(curve_obj)
            made += 1
    
        return {'FINISHED'}


class RDT_OT_export_embroidery_file(bpy.types.Operator):
    bl_idname = "rdt.export_embroidery_file"
    bl_label = "Export Embroidery File"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        time.sleep(2)
       
        return {'FINISHED'}


# Optional: quick color‑management helpers
class RDT_OT_set_view_standard(bpy.types.Operator):
    bl_idname = "rdt.set_view_standard"
    bl_label = "Use Standard View (sRGB)"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        scn = bpy.context.scene
        scn.display_settings.display_device = 'sRGB'
        scn.view_settings.view_transform = 'Standard'
        scn.view_settings.look = 'None'
        scn.view_settings.exposure = 0.0
        scn.view_settings.gamma = 1.0
        self.report({'INFO'}, "Color Management set to Standard (sRGB)")
        time.sleep(2)
        return {'FINISHED'}


class RDT_OT_set_view_filmic(bpy.types.Operator):
    bl_idname = "rdt.set_view_filmic"
    bl_label = "Use Filmic View"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        scn = bpy.context.scene
        scn.display_settings.display_device = 'sRGB'
        scn.view_settings.view_transform = 'Filmic'
        scn.view_settings.look = 'None'
        scn.view_settings.exposure = 0.0
        scn.view_settings.gamma = 1.0
        self.report({'INFO'}, "Color Management set to Filmic")
        time.sleep(2)
        return {'FINISHED'}


class RDT_PT_panel(bpy.types.Panel):
    bl_label = ""  
    bl_idname = "RDT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'EmbroForm'

    def draw(self, context):
        layout = self.layout
        
        
        title_box = layout.box()
        title_row = title_box.row()
        title_row.scale_y = 2.5  
        title_row.alignment = 'CENTER'
        title_row.label(text="EMBROFORM DESIGN TOOL", icon='TOOL_SETTINGS')
        
        box = layout.box()
        col = box.column(align=True)
        
        op_row = col.row(align=True)
        op_row.scale_y = 2.0 
        op_row.operator("rdt.import_mesh", text="IMPORT MESH", icon='EXPORT')
        
        op_row = col.row(align=True)
        op_row.scale_y = 2.0
        op_row.operator("rdt.import_patches_pp", text="SEGMENT", icon='SHADING_TEXTURE')
        
        op_row = col.row(align=True)
        op_row.scale_y = 2.0
        op_row.operator("rdt.deform_segment_and_packing", text="2D PACKING", icon='MESH_DATA')
         
        op_row = col.row(align=True)
        op_row.scale_y = 2.0
        op_row.operator("rdt.mark_points", text="IDENTIFY POINT PAIRS", icon='MARKER_HLT')
        
        op_row = col.row(align=True)
        op_row.scale_y = 2.0
        op_row.operator("rdt.construct_routing", text="GENERATE TENDON ROUTING", icon='IPO_ELASTIC')
        
        op_row = col.row(align=True)
        op_row.scale_y = 2.0
        op_row.operator("rdt.export_embroidery_file", text="EXPORT EMBROIDERY FILE", icon='IMPORT')

classes = (
    RDT_Preferences,
    RDT_SceneProps,
    RDT_OT_import_mesh,
    RDT_OT_import_patches_pp,
    RDT_OT_deform_segment_and_packing,
    RDT_OT_mark_points,
    RDT_OT_construct_routing,
    RDT_OT_export_embroidery_file,
    RDT_OT_set_view_standard,
    RDT_OT_set_view_filmic,
    RDT_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.rdt = bpy.props.PointerProperty(type=RDT_SceneProps)
    _maybe_add_sys_path()


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, 'rdt'):
        del bpy.types.Scene.rdt


if __name__ == "__main__":
    register()