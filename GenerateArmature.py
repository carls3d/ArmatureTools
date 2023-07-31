import bpy
import numpy as np
import bmesh
from math import dist as math_dist
from itertools import chain
from mathutils import Vector, Matrix
from typing import Iterable 
from dataclasses import dataclass
from io import StringIO 
import sys


def flatten_list(l:list[list]) -> list:
    """Flatten a nested list"""
    if not isinstance(l[0], Iterable) if isinstance(l, Iterable) else False: return l
    return list(chain.from_iterable(l))

def iter_depth(l:list) -> int:
    """Return the depth of a nested list"""
    if not isinstance(l, Iterable) or not any(l): return 0
    assert len(l) > 0, "List is empty" 
    return 1 + max(iter_depth(item) for item in l)

def sel_vertex(verts:bmesh.types.BMVertSeq) -> set[bmesh.types.BMVert]:
    """In range SELECTION, select vertex with smallest face area / most linked edges"""
    if mesh_vert_sel == 'Face area':           
        face_area = []
        for v in verts:
            area = np.sum([f.calc_area() for f in v.link_faces])
            face_area.append(area)
        sel_index = np.argmin(face_area)
    else: 
        sel_index = np.argmax([len(v.link_edges) for v in verts])
    return {verts[sel_index]}

def recursive_dict_print(item:dict) -> None:
    """Print a nested dict in a readable format"""
    def recursive_out(item, indent=0):
        prefix = "\t" * indent
        out = ""
        if type(item) == dict:
            for key, value in item.items():
                if indent: out += "\n"
                out += f"{prefix}{key}: {recursive_out(value, indent+1)}"
                if not indent: out += "\n\n"
        else: 
            out += f"{item}"
        return out
    print(recursive_out(item))

def nested_dict_update(dict1:dict,dict2:dict) -> None:
    """Update a nested dict with another nested dict"""
    for key, value in dict2.items():
        dict1.setdefault(key, {})
        dict1[key].update(value)

def reverse_dict_values(d:dict) -> dict:
    """Reverse the values of a dict"""
    return dict(reversed(list(d.items())))

class Capturing(list):
    """Capture stdout as a list to capture errors when running multiple operators"""
    #https://stackoverflow.com/a/16571630
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        
class Algo:
    
    @classmethod
    def optimize_coords(cls, coords_list:list[list[Vector]], dist:float=0.3, fac:float=0.0, resample_count:int=5):
        if debug: print("Optimize coords")
        #Note       ---- Algorithm ----
        #Note Spline Paramater  ->  [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, ...
        #Note Point Index       ->  [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, ...
        #Note Point Zip Index   ->  [(0, 8, 16, 24, 32, 40, 48), (1, 9, 17, ...
        #Note coords_list[i][0]  =  [0, 1, 2, 3, 4, 5, 6]
        #Note zipped[0]          =  [0, 8, 16, 24, 32, 40, 48]
        #Note 
        #Note dict filter   -> example
        #Note 0 [2, 3]      -> 0 [16, 24]
        #Note 1 [4, 5]      -> 1 [32, 40]
        #Note 2 [0, 1, 6]   -> 2 [0, 8, 48]
        #Note For each spline paramater
        #Note - using the zip list of that paramater, avarage the coords of the list grouped together by the dict filter
        
        fac_to_index = lambda a, b: round(a * (len(b) - 1))      # From ex.[0.0, 0.5, 1] -> [0, 3, 6]
        
        def list_to_indices(l:list) -> list:
            """Convert a nested list to a nested list of indices"""
            i, j, k = np.shape(l)
            a, b = np.indices((i, j))
            return a * j + b
        
        def create_filter(coords_list:list[list[Vector]], dist:float, fac:float) -> dict[int, list[int]]:
            """Filter: { New coord index: [ Spline paramater indices ], }"""
            #Note Creat a list of coords at a given parameter 
            #Note ex:    coords = [[0.1, 1.2, 4.0], [0.0, 1.3, 2.0]] -> coords[i][2] -> [4.0, 2.0]
            #TODO gen_index = lambda l, i: next(islice(l, i, None))                          ! if change to sets
            #Todo mergepoints = [gen_index(x, fac_to_index(fac, x)) for x in coords_list]
            mergepoints = [x[fac_to_index(fac, x)] for x in coords_list]
            bm = bmesh.new()
            for p in mergepoints: 
                bm.verts.new(p)
            parameter_coords = [v.co for v in bm.verts]    
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
            filter_coords = [v.co for v in bm.verts]
            bm.free()
            
            # Create filter
            coords_dict = {i:[] for i in range(len(filter_coords))} 

            # Fill out the dict filter
            for param_index, param_co in enumerate(parameter_coords):
                merge_dist_list = [(math_dist(filt_co, param_co)) for filt_co in filter_coords]
                closest_index = np.argmin(merge_dist_list)
                coords_dict[closest_index].append(param_index)
            return coords_dict
        
        coords_dict = create_filter(coords_list, dist, fac)
        # Resample the coords_list
        for i, spline_co in enumerate(coords_list):
            coords_list[i] = cls.resample_coords(spline_co, resample_count)
        point_index = list_to_indices(coords_list)
        zipped_points = list(zip(*point_index))
        
        new_coords = []
        point_coords = flatten_list(coords_list)
        for i, zp in enumerate(zipped_points):
            # i -> spline paramater
            layer = []
            for x in coords_dict.values(): 
                zip_coords = [point_coords[zp[j]] for j in x]
                layer.append(np.average(zip_coords, axis=0))
            new_coords.append(layer)
        new_coords = list(zip(*new_coords))
        coords = []
        for new_co in new_coords:
            if isinstance(new_co[0], Iterable):
                coords.append(list(map(Vector, new_co)))
        return coords

    @staticmethod
    def resample_coords(coords_list:list[list[Vector]], count:int) -> list[Vector]:
        """Resample point coordinates"""
        #TODO Resample length
        if debug: print("Resample coords")
        list_depth = iter_depth(coords_list)
        assert list_depth in [1, 2], f"Invalid depth: '{iter_depth(coords_list)}' -> Expected '1'"
        def run(point_coords):
            length_list = []
            for i in range(len(point_coords)-1):
                dist = math_dist(point_coords[i], point_coords[i+1])
                length_list.append(dist)
            length = np.sum(length_list)
            length_list = np.array(length_list)
            cum_sum = length_list.cumsum()
            cum_sum = np.insert(cum_sum,0,0)
            new_coords = []
            for i in range(count):
                target = length*(i / (count - 1))
                i1 = len(cum_sum[cum_sum <= target]) - 1
                if i1 >= len(length_list):
                    factor = 1
                    i1 = i1-1
                else:
                    factor = (target - cum_sum[i1]) / length_list[i1]
                point = point_coords[i1].lerp(point_coords[i1+1], factor)
                new_coords.append(point)
            return new_coords
        
        if list_depth == 2:
            return [run(coords) for coords in coords_list]
        else: 
            return run(coords_list)
    
    @staticmethod
    def gen_coords(obj:bpy.types.Object, verts:bmesh.types.BMVertSeq, max_sel:int) -> list[list[Vector], set[int]]:
        """Create a list of coords from the center of verts to the center of each loop from verts"""
        if debug: print("Gen coords")
        def center(self):   # Local avarage location of selection ignoring object offset
            coords = [obj.matrix_world @ v.co for v in self]
            return Vector(np.average(coords, 0))

        def linked_verts(self): # Linked verts
            return {v for vertex in self for edge in vertex.link_edges for v in edge.verts}
        
        coords = [center(verts)]
        island_indices = set()
        
        first = verts
        second = set()
        
        for i in range(max_sel):
            if second: first = second
            
            second = linked_verts(first)
            for vert in second: island_indices.add(vert.index)  # Add indices to set (does not add duplicates)
            if first == second: break 
            
            loop = second.difference(first)                     # Verts in second that are not in first
            skip = (i + 1) % (mod + 1)                          # Skip loops  
            if not skip: coords.append(center(loop))            # Add coords to list if not skipped
        return coords, island_indices
    
    @staticmethod
    def mesh_islands(bm:bmesh.types.BMesh, mode:str, max_isl:int, max_isl_verts:int) -> list[list]:
        """List of verts for every island"""
        if debug: print("Mesh islands")
        bm.verts.ensure_lookup_table()   
        copy_verts = {v for v in bm.verts if v.select} if mode == 'EDIT' else {*bm.verts}
        # if len(copy_verts) < max_isl: max_isl = len(copy_verts)
        # print(f"len(copy_verts): {len(copy_verts)} | max_isl: {max_isl} | max_isl_verts: {max_isl_verts}")
        islands = []
        for i in range(max_isl):
            if not copy_verts: break # Stop if no more verts
            first = set()
            first.add(copy_verts.pop())
            second = set()
            
            verts_ind = []
            for i in range(len(copy_verts)-1):
                if i: first = second
                second = {v for vertex in first for edge in vertex.link_edges for v in edge.verts}
                selected = first.intersection(second)
                
                for v in selected.copy():
                    verts_ind.append(bm.verts[v.index])
                    copy_verts.discard(v)
                if first == second: break
            islands.append(verts_ind)
        return islands
    
class GenerateCoords:
    reverse_bone_chain:bool
    
    @staticmethod
    def initiate_bmesh(obj:bpy.types.Object, mode:str, merge_dist:float = 0.0001) -> bmesh.types.BMesh:
        if debug: print(f"Initiating bmesh from {obj.name}")
        bpy.ops.object.mode_set(mode='OBJECT')
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        return bm
    
    @classmethod
    def from_mesh(cls, obj:bpy.types.Object, mode:str, resample:int = 0, max_sel:int = 100) -> list[Vector]:
        if debug: print(f"Generating coords from {obj.name}")
        bm = cls.initiate_bmesh(obj, mode)
        verts = [v for v in bm.verts]
        selected = {v for v in verts if v.select}
        assert verts, "No vertices"
        if mode == 'OBJECT': 
            selected = sel_vertex(verts)
        if not selected: return

        coords, island_indices = Algo.gen_coords(obj, selected, max_sel)
        if resample: coords = Algo.resample_coords(coords, resample)
        coords.reverse()
        grp_islands = {
            "island_0":{
                "bones":    {j:{"co":co, "name":None} for j, co in enumerate(coords)},
                "indices":  island_indices}}
        bm.free()
        return grp_islands
    
    @classmethod
    def from_islands(cls, obj:bpy.types.Object, mode:str, max_isl:int, max_loops:int, resample:int = 0) -> list[list[Vector]]:
        if debug: print(f"Generating coords from {obj.name}")
        if mode == 'EDIT': bpy.ops.mesh.select_linked()
        bm = cls.initiate_bmesh(obj, mode)
        islands_verts = Algo.mesh_islands(bm, mode, max_isl, max_loops)
        grp_islands = {}
        for i, verts in enumerate(islands_verts):
            if not verts: continue
            selected = sel_vertex(verts)
            coords, island_indices = Algo.gen_coords(obj, selected, max_loops)
            if resample: coords = Algo.resample_coords(coords, resample)
            coords.reverse()
            
            grp_islands.setdefault(
                f"island_{i}", {
                    "bones":    {j:{"co":co, "name":None} for j, co in enumerate(coords)}, 
                    "indices":  island_indices
                    })
    
        bm.free()
        return grp_islands
    
    @staticmethod
    def from_curves(obj:bpy.types.Object, resample:int) -> list[list[Vector]]:
        if debug: print(f"Generating coords from {obj.name}")
        # Runs if the obj type is a Hair curve
        matr = obj.matrix_world
        if hasattr(obj.data, 'curves'):
            hair_curves = obj.data.curves
            coords_list = [[matr @ p.position for p in c.points] for c in hair_curves]
            # indices = [[p.index for p in c.points] for c in hair_curves]
        
        # Runs if the obj type is a Bezier, Nurbs or Poly curve
        elif hasattr(obj.data, 'splines'):
            # Store arguments = [points coords, handles, attributes] depending on type
            # Make new splines with same type as original
            # Set arguments of new splines from stored arguments
            # Convert to mesh
            
            spline_points = {
                'POLY': [[[matr @ p.co for p in spline.points], 
                            (spline.use_cyclic_u,)]
                            for spline in obj.data.splines if spline.type == 'POLY'], 
                
                'NURBS': [[[matr @ p.co for p in spline.points], 
                            (spline.use_cyclic_u, 
                             spline.use_endpoint_u, 
                             spline.use_bezier_u, 
                             spline.order_u, 
                             spline.resolution_u)]
                            for spline in obj.data.splines if spline.type == 'NURBS'], 
                
                'BEZIER': [[[(matr @ p.co, matr @ p.handle_left, matr @ p.handle_right) for p in spline.bezier_points],
                            (spline.use_cyclic_u, 
                             spline.resolution_u)]
                            for spline in obj.data.splines if spline.type == 'BEZIER']
                }
                    
            _temp = bpy.data.objects.new('_temp', bpy.data.curves.new('_temp', type='CURVE'))
            bpy.context.collection.objects.link(_temp)
            bpy.context.view_layer.objects.active = _temp
            _temp.data.dimensions = '3D'
            
            for spline_type, splines in spline_points.items():
                if not splines: continue
                for points, attr in splines:
                    spline = _temp.data.splines.new(spline_type)
                    
                    if spline_type == 'POLY':
                        spline.points.add(len(points)-1)
                        spline.points.foreach_set('co', [co for p in points for co in p])
                        spline.use_cyclic_u,  = attr
                            
                    if spline_type == 'NURBS':
                        spline.points.add(len(points)-1)
                        spline.points.foreach_set('co', [co for p in points for co in p])
                        spline.use_cyclic_u, spline.use_endpoint_u, spline.use_bezier_u, spline.order_u, spline.resolution_u = attr
                       
                    if spline_type == 'BEZIER':
                        spline.bezier_points.add(len(points) - 1)
                        flatten = lambda x: [item for sublist in x for item in sublist] # Returns a flattened list [[1,2],[3,4]] -> [1,2,3,4]
                        coords, handles_left, handles_right = map(flatten, zip(*points))
                        spline.bezier_points.foreach_set('co', coords)
                        spline.bezier_points.foreach_set('handle_left', handles_left)
                        spline.bezier_points.foreach_set('handle_right', handles_right)
                        spline.use_cyclic_u, spline.resolution_u = attr
                        
            bpy.ops.object.select_all(action='DESELECT')
            _temp.select_set(True)
            bpy.ops.object.convert(target='MESH')
            bpy.ops.object.convert(target='CURVE')
            splines = _temp.data.splines
            coords_list = [[p.co.to_3d() for p in spline.points] for spline in splines]
            # indices = [[p.index for p in spline.points] for spline in splines]
            bpy.data.curves.remove(_temp.data)
            
        if resample: coords_list = [Algo.resample_coords(coords, resample) for coords in coords_list]
        grp_islands = {
            f"island_{i}":{
                "bones":   {j:{"co":co, "name":None} for j, co in enumerate(coords)},
                "indices":  {}} for i, coords in enumerate(coords_list)}
        return grp_islands
    
    
@dataclass
class ArmatureFuncs:
    armature_name:str
    bone_name:str
    armature:object = None
    
    @classmethod
    def create(cls, obj:bpy.types.Object, armature_name:str, bone_name:str, armature:bpy.types.Armature = None):
        """Creates a new armature or uses the existing one with the same name"""
        if debug: print("Create armature")
        if not armature_name:   armature_name = obj.name+'_rig'
        if not bone_name:       bone_name = 'bone'
        if armature:            
            armature_name = armature.name
            return cls(armature_name, bone_name, armature)
        
        else:
            if obj.users_collection[0] == bpy.context.scene.collection: #bpy.data.scenes['Scene'].collection
                collection = bpy.context.collection
            else:
                collection = bpy.data.collections[obj.users_collection[0].name]
            # Return new armature object with existing armature data if it exists
            if armature_name in bpy.data.armatures:
                armature_data = bpy.data.armatures[armature_name]
                
                # Generator -> does not get called unless users > 1
                # armature_objects = (ob for ob in bpy.context.view_layer.objects if ob.data == armature_data and ob.visible_get() == True)
                armature_objects = [ob for ob in bpy.context.view_layer.objects if ob.data == armature_data and ob.visible_get() == True]
                users = len(armature_objects)
                
                # If chosen armature has no object, create one
                if users == 0: 
                    armature = bpy.data.objects.new(armature_data.name, armature_data)
                    collection.objects.link(armature)
                    
                # If chosen armature has an object, use it
                elif users == 1: 
                    # Use first armature object that uses the chosen armature data
                    assert armature_objects, f"No armature object with name '{armature_name}' is visible in viewport"
                    armature = armature_objects[0]
                    
                # If chosen armature has more than one user(object), check if one of them is the parent of obj
                elif users > 1: # instad of else for readability
                    assert armature_objects, f"No armature objects with name '{armature_name}' are visible in viewport"
                    # Check for armatures that are parents of obj
                    for arm_obj in armature_objects:
                        if arm_obj == obj.parent:
                            armature = arm_obj
                            break
                    assert armature, f"Armature '{armature_name}' has multiple armature objects but none of them are the parent of '{obj.name}'"
                assert armature, f"Failed to create armature with name '{armature_name}'"
                armature.show_in_front = True
                return cls(armature_name, bone_name, armature)
            
            # Return armature object if it exists
            elif armature_name in bpy.context.view_layer.objects:
                assert bpy.context.view_layer.objects[armature_name].type == "ARMATURE", "Existing object with same name is not an armature"
                armature = bpy.context.view_layer.objects[armature_name]
                return cls(armature_name, bone_name, armature)
                
            # Else return a new armature object & data
            else:
                armature_data = bpy.data.armatures.new(armature_name)
                armature_name = armature_data.name
                armature = bpy.data.objects.new(armature_data.name, armature_data)
                collection.objects.link(armature)
                armature.show_in_front = True
                return cls(armature_name, bone_name, armature)
    
    def get_coords(self, mode:str) -> list[list]:
        """Return the armatures bone coordinates"""
        if debug: print("Armature get coords")
        assert self.armature.type == "ARMATURE", "Object is not an armature"

        if mode == 'POSE':
            bpy.context.view_layer
            bpy.ops.object.mode_set(mode='EDIT')
            mode = 'EDIT'
        if mode == 'EDIT': 
            bones = self.armature.data.edit_bones
            bones = {b for b in bones if b.select}
            root_bones = {b for b in bones if b.parent is None}
            bones_head = [[b.head, *[b2.head for b2 in b.children_recursive], b.children_recursive[-1].tail] for b in root_bones]
        else:
            bones = self.armature.data.bones
            root_bones = {b for b in bones if b.parent is None}
            bones_head = [[b.head_local, *[b2.head_local for b2 in b.children_recursive], b.children_recursive[-1].tail_local] for b in root_bones]
        assert bones, "No bones selected"
        # if reverse_bone_chain: 
        #     for l in bones_head: l.reverse()
        return bones_head
    
    def generate_bones(self, isl:dict, reverse:bool = False) -> None:
        """Generate the armatures bones"""
        if debug: print("Armature generate bones")
        
        # Set armature as active object and check if it is editable
        bpy.context.view_layer.objects.active = self.armature
        assert bpy.context.active_object, f"Expected armature '{self.armature_name}' to be editable"
        
        # Select armature and clear parent if it has one & report it
        bpy.ops.object.select_all(action='DESELECT')
        self.armature.select_set(True)
        if self.armature.parent: 
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            report({'INFO'}, "Armature parent cleared")
            
        # Apply armature transforms if it has any & report it
        if self.armature.matrix_world != Matrix.Identity(4):
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            report({'INFO'}, "Armature transforms applied")
        
        bpy.ops.object.mode_set(mode='EDIT')
        for i, value in enumerate(isl.values()):
            isl_bones = value["bones"]
            if reverse: isl_bones = reverse_dict_values(isl_bones)
            # isl_items = [*isl_bones.items()]
            isl_values = [*isl_bones.values()]
            
            for j, bone_values in enumerate(isl_values[:-1]):
                bone_name = self.bone_name+str(i)+'.000'
                bone = self.armature.data.edit_bones.new(bone_name)
                bone_values["name"] = bone.name
                bone.head = isl_values[j]["co"]
                bone.tail = isl_values[j+1]["co"]
                bone.use_connect = True
                if j: bone.parent = parent
                parent = bone
                
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        self.armature.select_set(True)
    
    def clear_bones(self, edit_bones = None) -> None:
        """Clears all bones from the armature"""
        if debug: print("Armature clear bones")
        
        if not edit_bones: edit_bones = self.armature.data.edit_bones
        bpy.context.view_layer.objects.active = self.armature
        assert bpy.context.active_object, "Expected active object to be editable. Is existing armature hidden?"
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in edit_bones:
            edit_bones.remove(bone)
        bpy.ops.object.mode_set(mode='OBJECT')

    @staticmethod
    def auto_weight(islands_grp:dict, arm_obj:object, mesh_obj:object):
        """
        islands_grp:      {f"Island_{i}":{"bones":{j:{"co":Vector((0,0,0)),"name":"bonename"}, ...},"indices":[]}, ...}\n
        temp_meshes:      {f"{i}":object, ...}\n
        temp_armatures:   {f"{i}":object, ...}"""
        if debug: print("Armatures auto weight")
        def set_active_selected(obj:object) -> None:
            bpy.context.view_layer.objects.active = obj
            for ob in bpy.data.objects: ob.select_set(ob == obj)
        
        def sort_islands(isl:dict) -> dict:
            """
            From: {f"Island_{i}":{"bones":{j:{"co":Vector((0,0,0)),"name":"bonename"}, ...},"indices":[]}, ...}\n
            To:   {f"Island_{i}":{"bones":[], "indices":[]}, ...}\n"""
            isl_data = {}
            for key, value in isl.items():
                bone_names = [*value["bones"].values()]
                
                isl_data.setdefault(key, {"bones":[], "indices":[]})
                isl_data[key]["bones"] = {item["name"] for item in bone_names if item["name"]}
                isl_data[key]["indices"] = value["indices"]
            return isl_data
  
        def temp_meshes() -> dict:
            set_active_selected(mesh_obj)
            
            # Store islands data in vertex groups #Note: (using index data when removing/creating verts doesn't work as the index order is different)
            vertex_groups = {}
            for i, (key,value) in enumerate(islands_data.items()):
                vgrp_name = f"_temp_{key}"
                if vgrp_name in mesh_obj.vertex_groups: 
                    vgrp = mesh_obj.vertex_groups[vgrp_name]
                else: 
                    vgrp = mesh_obj.vertex_groups.new(name=f"{vgrp_name}")
                vertex_groups.setdefault(key, vgrp)
                mesh_obj.vertex_groups[vgrp.name].add(list(value["indices"]), 1.0, 'REPLACE')

            # Separate by vertex groups
            mesh_objects = {}
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            for i, (key,value) in enumerate(vertex_groups.items()):
                mesh_obj.vertex_groups.active_index = value.index
                bpy.ops.object.vertex_group_select()
                bpy.ops.mesh.separate(type='SELECTED')
                for sel_obj in bpy.context.selected_objects: 
                    if sel_obj != bpy.context.object: 
                        mesh_objects.setdefault(i, sel_obj)
                        sel_obj.name = f"{mesh_obj.name}_{key}"
                        sel_obj.select_set(False)
                mesh_obj.vertex_groups.remove(value)
                
            # Clear temp vertex groups from separated meshes
            bpy.ops.object.mode_set(mode='OBJECT')
            for obj in mesh_objects.values():
                for key in islands_data.keys():
                    vgrp_name = f"_temp_{key}"
                    if vgrp_name in obj.vertex_groups:
                        obj.vertex_groups.remove(obj.vertex_groups[vgrp_name])
            return mesh_objects
            
        def temp_armatures() -> object:
            set_active_selected(arm_obj)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.armature.select_all(action='DESELECT')
            objects = {}
            for i, (key,value) in enumerate(islands_data.items()):
                for bone in arm_obj.data.edit_bones:
                    bone.select = bone.name in value["bones"]
                bpy.ops.armature.separate()
                for sel_obj in bpy.context.selected_objects: 
                    if sel_obj != bpy.context.object: 
                        sel_obj.name = f"{arm_obj.name}_{key}"
                        objects.setdefault(i, sel_obj)
                    sel_obj.select_set(sel_obj == bpy.context.object)
            bpy.ops.object.mode_set(mode='OBJECT')
            return objects
        
        def weights_islands():
            meshes = temp_meshes()
            armatures = temp_armatures()
            
            report({'INFO'}, "Applying auto weights")
            with Capturing() as output:                         # Capture error outputs from console
                for i, armature in enumerate(armatures.values()):
                    bpy.context.view_layer.objects.active = armature
                    # Only select mesh object to be parented
                    for obj in bpy.data.objects: 
                        obj.select_set(obj == meshes[i])
                    # Remove vertex groups if mix_mode 'REPLACE'
                    if mix_mode == 'REPLACE': 
                        meshes[i].vertex_groups.clear()
                    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
                    
                # Parent original mesh to original armature
                set_active_selected(arm_obj)
                mesh_obj.select_set(True)
                bpy.ops.object.parent_set(type='ARMATURE')
                
                # Join armatures
                set_active_selected(arm_obj)
                for arm in armatures.values(): arm.select_set(True)
                bpy.ops.object.join()
                
                # Join meshes
                set_active_selected(mesh_obj)
                for mesh in meshes.values(): mesh.select_set(True)
                bpy.ops.object.join()
            if not output: 
                report({'INFO'}, "Auto weights applied")
            return output
       
        islands_data = sort_islands(islands_grp)
        out = weights_islands()
        
        # Output errors
        if out: 
            [report({'ERROR'}, str(item)) for item in out]
            print("Error output:", out)
        set_active_selected(arm_obj)
        
        # recursive_dict_print(islands_grp)
        # recursive_dict_print(islands_data)


#----------------------------------------------------------------

#NOTE Variables set from outside script
_locals = locals()
def is_local(var_name:str, default_var):
    return _locals[var_name] if var_name in _locals else default_var

mod                  = is_local('mod', 0)
max_isl              = is_local('max_isl', 500)
max_loops            = is_local('max_loops', 100)
reverse_bone_chain   = is_local('reverse_bone_chain', False)
armature_name        = is_local('armature_name', '')
bone_name            = is_local('bone_name', 'bone')
mesh_vert_sel        = is_local('mesh_vert_sel', 'Face area')
resample_count       = is_local('resample_count', 8)
execute_type         = is_local('execute_type', 'ISLANDS')
auto_weights         = is_local('auto_weights', True)
mix_mode             = is_local('mix_mode', 'REPLACE')
dist                 = is_local('dist', 0.12)
fac                  = is_local('fac', 0.0)
self                 = is_local('self', None)
debug                = is_local('debug', True)
report = self.report if self else print

def main():
    print("------------- CTools Armature generator -------------\n")
    obj = bpy.context.active_object
    obj_mode = bpy.context.object.mode
    if not obj: 
        report({'ERROR'}, "No active object"); return
    if obj.type == 'MESH' and obj_mode == 'EDIT':
        obj.update_from_editmode()
        if not any([v.select for v in obj.data.vertices]): 
            report({'ERROR'}, "No vertices selected"); return
    # Using "if, elif" instead of "case" statements for downgradability
    if execute_type in ['CURVE', 'CURVES']:
        assert 'CURVE' in obj.type, "Object is not a curve"
        arm = ArmatureFuncs.create(obj, armature_name, bone_name)
        islands_dict = GenerateCoords.from_curves(obj, resample_count)
        arm.generate_bones(islands_dict, reverse_bone_chain)
        
    elif execute_type == 'MESH':
        assert obj.type == 'MESH', "Object is not a mesh"
        arm = ArmatureFuncs.create(obj, armature_name, bone_name)
        islands_dict = GenerateCoords.from_mesh(obj, obj_mode, resample_count, max_loops)
        arm.generate_bones(islands_dict, reverse_bone_chain)
        if auto_weights: ArmatureFuncs.auto_weight(islands_dict, arm.armature, obj)
        
    elif execute_type == 'ISLANDS':
        assert obj.type == 'MESH', "Object is not a mesh"
        arm = ArmatureFuncs.create(obj, armature_name, bone_name)
        islands_dict = GenerateCoords.from_islands(obj, obj_mode, max_isl, max_loops, resample_count)
        arm.generate_bones(islands_dict, reverse_bone_chain)
        if auto_weights: ArmatureFuncs.auto_weight(islands_dict, arm.armature, obj)
        
    elif execute_type == 'ARMATURE':
        assert obj.type == 'ARMATURE', "Object is not an armature"
        assert resample_count, "Resample count must be greater than 0"
        arm = ArmatureFuncs.create(obj, armature_name, bone_name, obj)
        
        coords = arm.get_coords(obj_mode)
        coords = Algo.optimize_coords(coords, dist, fac, resample_count)
        
        arm.clear_bones()
        arm.generate_bones(islands_dict, reverse_bone_chain)
    print("\n--------------------------")
main()
