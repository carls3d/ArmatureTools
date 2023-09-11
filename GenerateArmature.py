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
from collections import defaultdict

class ddict(defaultdict):
    __repr__ = dict.__repr__
    
if __name__ == '__main__':
    from EditArmature import EditFuncs

def flatten_list(l:list[list]) -> list:
    """Flatten a nested list"""
    if not isinstance(l[0], Iterable) if isinstance(l, Iterable) else False: return l
    return list(chain.from_iterable(l))

def iter_depth(l:list) -> int:
    """Return the depth of a nested list"""
    if not isinstance(l, Iterable) or not any(l): return 0
    assert len(l) > 0, "List is empty" 
    return 1 + max(iter_depth(item) for item in l)

def sel_vertex(verts:bmesh.types.BMVertSeq, vert_sel:str = 'Face area') -> set[bmesh.types.BMVert]:
    """In range SELECTION, select vertex with smallest face area / most linked edges"""
    if vert_sel == 'Face area':           
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

def popup_window(title:str = "Error", text:str|list = "Error message", icon:str = 'ERROR'):
    """Minimal popup error - less intrusive than assert, easier to read than print"""
    def popup(self, context):
        # Each element in text is a line in the popup window
        lines = text if type(text) == list else text.split("\n") 
        for line in lines:
            row = self.layout.row()
            row.label(text=line)
    bpy.context.window_manager.popup_menu(popup, title=title, icon=icon) 

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
    def optimize_coords(cls, coords_list:list[list[Vector]], dist:float=0.3, fac:float=0.0, resample:int=5):
        # if debug: print("Optimize coords")
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
            coords_list[i] = cls.resample_coords(spline_co, resample)
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
    def resample_coords(coords_list:list[list[Vector]], resample:int) -> list[Vector]:
        """Resample point coordinates"""
        #TODO Resample length
        # if debug: print("Resample coords")
        assert resample > 0, f"Invalid resample: '{resample}' -> Expected > 0"
        resample += 1
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
            for i in range(resample):
                target = length*(i / (resample - 1))
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
    def gen_coords(obj:bpy.types.Object, verts:bmesh.types.BMVertSeq, modulo:int, max_len:int) -> list[list[Vector], set[int]]:
        """Create a list of coords from the center of verts to the center of each loop from verts"""
        # if debug: print("Gen coords")
        def center(self):   # Local avarage location of selection ignoring object offset
            coords = [obj.matrix_world @ v.co for v in self]
            return Vector(np.average(coords, 0))

        def linked_verts(self): # Linked verts
            return {v for vertex in self for edge in vertex.link_edges for v in edge.verts}
        
        coords = [center(verts)]
        island_indices = set()
        
        first = verts
        second = set()
        
        for i in range(max_len):
            if second: first = second
            
            second = linked_verts(first)
            for vert in second: island_indices.add(vert.index)  # Add indices to set (does not add duplicates)
            if first == second: break 
            
            loop = second.difference(first)                     # Verts in second that are not in first
            skip = (i + 1) % (modulo + 1)                          # Skip loops  
            if not skip: coords.append(center(loop))            # Add coords to list if not skipped
        return coords, island_indices
    
    @staticmethod
    def mesh_islands(bm:bmesh.types.BMesh, mode:str, max_isl:int, max_isl_verts:int) -> list[list]:
        """List of verts for every island"""
        # if debug: print("Mesh islands")
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
    
    @staticmethod
    def calculate_vertex_weights(bones:list[bpy.types.Bone], vertices:list[bpy.types.MeshVertex], power:float, threshold:float) -> dict[int, dict[str, float]]:
        """Weight vertices based on distance to bones"""
        vertex_weights = ddict(lambda: ddict(float))
        
        # Distance between vertices and bones
        for vert in vertices:
            for bone in bones:
                distance = (vert.co - (bone.head_local + bone.tail_local)/2).length
                if distance != 0:
                    vertex_weights[vert.index][bone.name] = 1 / (distance ** power)
        
        # Normalize weights
        for weights in vertex_weights.values():
            total_weight = sum(weights.values())
            for bone_name in weights.keys():
                weights[bone_name] /= total_weight
        
        # Remove weights below the threshold and re-normalize
        for weights in vertex_weights.values():
            for bone_name in list(weights.keys()):
                if weights[bone_name] < threshold:
                    del weights[bone_name]
            
            # Re-normalize after thresholding
            total_weight = sum(weights.values())
            if total_weight == 0:  # Set a default weight if total is zero
                for bone in bones:
                    weights[bone.name] = 1.0 / len(bones)
                total_weight = sum(weights.values())
            
            for bone_name in weights.keys():
                weights[bone_name] /= total_weight
                
        return vertex_weights
    
class GenerateCoords:
    reverse_bone_chain:bool
    
    @staticmethod
    def init_generate(armature_name:str, bone_name:str):
        obj = bpy.context.object
        mode = bpy.context.object.mode
        if not obj:
            return popup_window(text="No active object")
        if not obj.visible_get():
            return popup_window(text="Object is hidden")
        if mode == 'EDIT_MESH':
            if not bpy.context.object.data.total_vert_sel:
                return popup_window(text="No vertices selected")
        arm = ArmatureFuncs.create(obj, armature_name, bone_name)
        return obj, mode, arm
    
    @staticmethod
    def initiate_bmesh(obj:bpy.types.Object, mode:str, merge_dist:float = 0.0001) -> bmesh.types.BMesh:
        # if debug: print(f"Initiating bmesh from {obj.name}")
        bpy.ops.object.mode_set(mode='OBJECT')
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        return bm
    
    @classmethod
    def from_mesh(cls, obj:bpy.types.Object, mode:str, resample:int, modulo:int, max_len:int, vert_sel:str) -> list[Vector]:
        # if debug: print(f"Generating coords from {obj.name}")
        bm = cls.initiate_bmesh(obj, mode)
        verts = [v for v in bm.verts]
        selected = {v for v in verts if v.select}
        assert verts, "No vertices"
        if mode == 'OBJECT': 
            selected = sel_vertex(verts, vert_sel)
        if not selected: return

        coords, island_indices = Algo.gen_coords(obj, selected, modulo, max_len)
        if resample: coords = Algo.resample_coords(coords, resample)
        coords.reverse()
        grp_islands = {
            "island_0":{
                "bones":    {j:{"co":co, "name":None} for j, co in enumerate(coords)},
                "indices":  island_indices}}
        bm.free()
        return grp_islands
    
    @classmethod
    def from_islands(cls, obj:bpy.types.Object, mode:str, resample:int, modulo:int, max_len:int, max_isl:int, vert_sel:str) -> list[list[Vector]]:
        # if debug: print(f"Generating coords from {obj.name}")
        if mode == 'EDIT': bpy.ops.mesh.select_linked()
        bm = cls.initiate_bmesh(obj, mode)
        islands_verts = Algo.mesh_islands(bm, mode, max_isl, max_len)
        grp_islands = {}
        for i, verts in enumerate(islands_verts):
            if not verts: continue
            selected = sel_vertex(verts, vert_sel)
            coords, island_indices = Algo.gen_coords(obj, selected, modulo, max_len)
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
        # if debug: print(f"Generating coords from {obj.name}")
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
        # if debug: print("Create armature")
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
        # if debug: print("Armature get coords")
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
        # if debug: print("Armature generate bones")
        
        # Set armature as active object and check if it is editable
        bpy.context.view_layer.objects.active = self.armature
        assert bpy.context.active_object, f"Expected armature '{self.armature_name}' to be editable"
        
        # Select armature and clear parent if it has one & report it
        bpy.ops.object.select_all(action='DESELECT')
        self.armature.select_set(True)
        if self.armature.parent: 
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            # report({'INFO'}, "Armature parent cleared")
            
        # Apply armature transforms if it has any & report it
        if self.armature.matrix_world != Matrix.Identity(4):
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            # report({'INFO'}, "Armature transforms applied")
        
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
        # if debug: print("Armature clear bones")
        
        if not edit_bones: edit_bones = self.armature.data.edit_bones
        bpy.context.view_layer.objects.active = self.armature
        assert bpy.context.active_object, "Expected active object to be editable. Is existing armature hidden?"
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in edit_bones:
            edit_bones.remove(bone)
        bpy.ops.object.mode_set(mode='OBJECT')

    @staticmethod
    def auto_weight(islands_grp:dict, arm_obj:object, mesh_obj:object, mix_mode:str = 'REPLACE') -> None:
        """
        islands_grp:      {f"Island_{i}":{"bones":{j:{"co":Vector((0,0,0)),"name":"bonename"}, ...},"indices":[]}, ...}\n
        temp_meshes:      {f"{i}":object, ...}\n
        temp_armatures:   {f"{i}":object, ...}"""
        # if debug: print("Armatures auto weight")
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
            
            # report({'INFO'}, "Applying auto weights")
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
            # if not output: 
                # report({'INFO'}, "Auto weights applied")
            return output
       
        islands_data = sort_islands(islands_grp)
        out = weights_islands()
        
        # Output errors
        if out: 
            popup_window([str(item) for item in out])
            print("Error output:", out)
        set_active_selected(arm_obj)

    @staticmethod
    def set_armature_name(adv_settings:bool, armature_name:str):
        def auto_name(obj):
            if obj.parent and obj.parent.type == 'ARMATURE':
                return obj.parent.name
            else:
                return bpy.context.object.name + '_rig'
        
        def auto_name_advanced(gen_type, existing):
            x = {
                'Auto': auto_name,
                'Custom': armature_name,
                'Existing': existing.name
            }
            return x[gen_type]
        
        obj = bpy.context.object
        
        armature_name = auto_name_advanced() if adv_settings else auto_name(obj)
        
        objects = bpy.data.objects
        if objects[armature_name].visible_get() == False if armature_name in objects else False:
            popup_window(f"Armature '{armature_name}' is hidden")
            return None
        else:
            return armature_name

#----------------------------------------------------------------

# check_vars = lambda *args: any(not lst for lst in [*args])

def get_armature_name():
    def armature_name_auto():
        if valid_armatures:
            return valid_armatures[0].name
        else:
            return bpy.context.object.name + '_rig'
        
    def armature_name_custom():
        if 'sna_armature_name' in bpy.context.scene and bpy.context.scene.sna_armature_name:
            return bpy.context.scene.sna_armature_name
        else:
            return armature_name_auto()
    
    def armature_name_existing():
        if 'sna_existing_armature' in bpy.context.scene and bpy.context.scene.sna_existing_armature:
            return bpy.context.scene.sna_existing_armature.name
        else:
            return armature_name_auto()
    
    valid_armatures = EditFuncs.connected_armatures()
    arm_name_type = bpy.context.scene.sna_armature_gen_type if 'sna_armature_gen_type' in bpy.context.scene else 'Auto'
    get_name_type = {
        'Auto': armature_name_auto,
        'Custom': armature_name_custom,
        'Existing': armature_name_existing
    }
    get_name_type[arm_name_type]
    armature_name = get_name_type[arm_name_type]()
    return armature_name

class CT_GenerateBonesMesh(bpy.types.Operator):
    bl_idname = "ct.generate_bones_mesh"
    bl_label = "Generate Bones"
    bl_description = "Generate bones from mesh selection"
    bl_options = {"REGISTER", "UNDO"}
    
    reverse: bpy.props.BoolProperty(name='reverse', description='', default=False)
    resample: bpy.props.IntProperty(name='resample', description='', default=0, subtype='NONE', min=0)
    resample_on: bpy.props.BoolProperty(name='resample_on', description='', default=False)
    
    apply_weights: bpy.props.BoolProperty(name='apply_weights', description='', default=False)
    modulo: bpy.props.IntProperty(name='modulo', description='Skip Bones', default=0, subtype='NONE', min=0)
    max_loops: bpy.props.IntProperty(name='max_loops', description='Max bones per island. (Increase when needed)', default=100, subtype='NONE', min=1, max=1000)
    vert_sel: bpy.props.EnumProperty(name='vert_sel', description='', items=[('Face area', 'Face area', '', 0, 0), ('Linked edges', 'Linked edges', '', 0, 1)])

    mix_mode: bpy.props.EnumProperty(name='mix_mode', description='', items=[('REPLACE', 'REPLACE', '', 0, 0), ('ADD', 'ADD', '', 0, 1)])
    advanced_settings: bpy.props.BoolProperty(name='advanced_settings', description='', default=False)
    
    @classmethod
    def poll(cls, context):
        if bpy.app.version < (3, 3, 0):
            cls.poll_message_set(f'Unsupported in Blender {bpy.app.version_string}')
            return False
        return True

    def execute(self, context):
        armature_name = get_armature_name()
        bone_name = bpy.context.scene.sna_bone_name if 'sna_bone_name' in bpy.context.scene else 'bone'
        self.resample = self.resample if self.resample_on else 0
        
        obj, obj_mode, arm = GenerateCoords.init_generate(armature_name, bone_name)
        islands_dict = GenerateCoords.from_mesh(obj, obj_mode, self.resample, self.modulo, self.max_loops, self.vert_sel)
        arm.generate_bones(islands_dict, self.reverse)
        if self.apply_weights: 
            ArmatureFuncs.auto_weight(islands_dict, arm.armature, obj, self.mix_mode)
       
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column(heading='', align=True)
        col.scale_x = 1.20
        col.scale_y = 1.20
        row_reverse = col.row(heading='', align=True)
        row_reverse.prop(self, 'reverse', text='', icon_value=(36 if self.reverse else 38), emboss=True)
        box_rev = row_reverse.box()
        box_rev.scale_x = 1.0
        box_rev.scale_y = 0.63
        box_rev.label(text='Reverse', icon_value=715)
        row_resample = col.row(heading='', align=True)
        row_resample.prop(self, 'resample_on', text='', icon_value=(39 if self.resample_on else 38), emboss=True)
        box_3C89A = row_resample.box()
        box_3C89A.scale_x = 1.0
        box_3C89A.scale_y = 0.63
        box_3C89A.label(text='Resample', icon_value=16)
        row_8E3B7 = row_resample.row(heading='', align=True)
        row_8E3B7.enabled = self.resample_on
        row_8E3B7.prop(self, 'resample', text='', icon_value=0, emboss=True)

        # Mesh
        col_8DE44 = col.column(heading='', align=True)
        row_66FE3 = col_8DE44.row(heading='', align=True)
        row_66FE3.prop(self, 'apply_weights', text='', icon_value=(36 if self.apply_weights else 38), emboss=True)
        row_C7688 = row_66FE3.row(heading='', align=True)
        box_79147 = row_C7688.box()
        box_79147.scale_x = 1.0
        box_79147.scale_y = 0.63
        box_79147.label(text='Apply weights', icon_value=475)
        row_C7688.prop(self, 'mix_mode', text='', icon_value=0, emboss=True)
        
        if 'ctools_armature' in bpy.context.preferences.addons:
            advanced_settings = bpy.context.preferences.addons['ctools_armature'].preferences.sna_advanced_settings
        elif bpy.context.scene.sna_addon_prefs_temp.sna_advanced_settings:
            advanced_settings = bpy.context.scene.sna_addon_prefs_temp.sna_advanced_settings
        else:
            advanced_settings = False
        
        if advanced_settings:
            col_E2B81 = col_8DE44.column(heading='', align=True)
            col_E2B81.alignment = 'Expand'.upper()
            col_E2B81.separator(factor=1.0)
            
            # Islands
            if False:
                row_27CC4 = col_E2B81.row(heading='', align=True)
                box_9F073 = row_27CC4.box()
                box_9F073.scale_x = 1.0
                box_9F073.scale_y = 0.63
                box_9F073.label(text='Start vertex algorithm', icon_value=0)
                row_27CC4.prop(self, 'sna_mesh_vert_sel', text='', icon_value=0, emboss=True)
                
            row_A61CC = col_E2B81.row(heading='', align=True)
            row_A61CC.scale_x = 1.0
            row_A61CC.scale_y = 1.0
            box_36178 = row_A61CC.box()
            box_36178.scale_x = 1.0
            box_36178.scale_y = 0.63
            box_36178.label(text='Modulo', icon_value=0)
            row_A61CC.prop(self, 'modulo', text='', icon_value=0, emboss=True)
            row_F4484 = col_E2B81.row(heading='', align=True)
            box_C0076 = row_F4484.box()
            box_C0076.scale_x = 1.0
            box_C0076.scale_y = 0.63
            box_C0076.label(text='Max length', icon_value=0)
            row_F4484.prop(self, 'max_loops', text='', icon_value=0, emboss=True)

    def invoke(self, context, event):
        obj = bpy.context.object
        if not obj:
            self.report({'ERROR'}, message='No active object')
            return {"CANCELLED"}
        if not obj.visible_get():
            self.report({'ERROR'}, message='Object is hidden')
            return {"CANCELLED"}
        
        if obj.type == 'MESH' and bpy.context.mode == 'EDIT_MESH':
            if not obj.data.total_vert_sel:
                self.report({'ERROR'}, message='No vertices selected')
                return {"CANCELLED"}
        context.window_manager.invoke_props_popup(self, event)
        return self.execute(context)
    
class CT_GenerateBonesCurves(bpy.types.Operator):
    bl_idname = "ct.generate_bones_curves"
    bl_label = "Generate Bones"
    bl_description = "Generate bones from curves"
    bl_options = {"REGISTER", "UNDO"}
    reverse: bpy.props.BoolProperty(name='reverse', description='', default=False)
    resample: bpy.props.IntProperty(name='resample', description='', default=0, subtype='NONE', min=0)
    resample_on: bpy.props.BoolProperty(name='resample_on', description='', default=False)

    @classmethod
    def poll(cls, context):
        if bpy.app.version < (3, 3, 0):
            cls.poll_message_set(f'Unsupported in Blender {bpy.app.version_string}')
            return False
        return True

    def execute(self, context):
        armature_name = get_armature_name()
        bone_name = bpy.context.scene.sna_bone_name if 'sna_bone_name' in bpy.context.scene else 'bone'
        self.resample = self.resample if self.resample_on else 0
        
        obj, _, arm = GenerateCoords.init_generate(armature_name, bone_name)
        islands_dict = GenerateCoords.from_curves(obj, self.resample)
        arm.generate_bones(islands_dict, self.reverse)
        
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        col_2D2FB = layout.column(align=True)
        col_2D2FB.scale_x = 1.20
        col_2D2FB.scale_y = 1.20
        
        row_D6B39 = col_2D2FB.row(align=True)
        row_D6B39.prop(self, 'reverse', text='', icon_value=36 if self.reverse else 38)
        box_FE569 = row_D6B39.box()
        box_FE569.scale_x = 1.0
        box_FE569.scale_y = 0.63
        box_FE569.label(text='Reverse', icon_value=715)
        
        row_20AB8 = col_2D2FB.row(align=True)
        row_20AB8.prop(self, 'resample_on', text='', icon_value=39 if self.resample_on else 38)
        box_9D78D = row_20AB8.box()
        box_9D78D.scale_x = 1.0
        box_9D78D.scale_y = 0.63
        box_9D78D.label(text='Resample', icon_value=16)
        row_16A4B = row_20AB8.row(align=True)
        row_16A4B.enabled = self.resample_on
        row_16A4B.prop(self, 'resample', text='')

    def invoke(self, context, event):
        obj = bpy.context.object
        if not obj:
            self.report({'ERROR'}, message='No active object')
            return {"CANCELLED"}
        if not obj.visible_get():
            self.report({'ERROR'}, message='Object is hidden')
            return {"CANCELLED"}
        if obj.type not in ['CURVE', 'CURVES']:
            self.report({'ERROR'}, message=f"Expected object type to be 'CURVE' or 'CURVES', not '{obj.type}'")
        if bpy.context.mode not in ['OBJECT', 'EDIT_CURVE', 'EDIT_CURVES', 'SCULPT_CURVES']:
            self.report({'ERROR'}, message=f"Expected object mode to be in: ['OBJECT', 'EDIT_CURVE', 'EDIT_CURVES', 'SCULPT_CURVES'] not '{bpy.context.mode}'")
        context.window_manager.invoke_props_popup(self, event)
        return self.execute(context)

class CT_GenerateBonesIslands(bpy.types.Operator):
    bl_idname = "ct.generate_bones_islands"
    bl_label = "Generate Bones"
    bl_description = "Generate bones from mesh islands"
    bl_options = {"REGISTER", "UNDO"}
    
    reverse: bpy.props.BoolProperty(name='reverse', description='', default=False)
    resample: bpy.props.IntProperty(name='resample', description='', default=0, subtype='NONE', min=0)
    resample_on: bpy.props.BoolProperty(name='resample_on', description='', default=False)
    
    apply_weights: bpy.props.BoolProperty(name='apply_weights', description='', default=False)
    modulo: bpy.props.IntProperty(name='modulo', description='Skip Bones', default=0, subtype='NONE', min=0)
    max_loops: bpy.props.IntProperty(name='max_loops', description='Max bones per island. (Increase when needed)', default=100, subtype='NONE', min=1, max=1000)
    vert_sel: bpy.props.EnumProperty(name='vert_sel', description='', items=[('Face area', 'Face area', '', 0, 0), ('Linked edges', 'Linked edges', '', 0, 1)])

    mix_mode: bpy.props.EnumProperty(name='mix_mode', description='', items=[('REPLACE', 'REPLACE', '', 0, 0), ('ADD', 'ADD', '', 0, 1)])
    advanced_settings: bpy.props.BoolProperty(name='advanced_settings', description='', default=False)
    
    @classmethod
    def poll(cls, context):
        if bpy.app.version < (3, 3, 0):
            cls.poll_message_set(f'Unsupported in Blender {bpy.app.version_string}')
            return False
        return True

    def execute(self, context):
        armature_name = get_armature_name()
        bone_name = bpy.context.scene.sna_bone_name if 'sna_bone_name' in bpy.context.scene else 'bone'
        self.resample = self.resample if self.resample_on else 0
        
        obj, obj_mode, arm = GenerateCoords.init_generate(armature_name, bone_name)
        islands_dict = GenerateCoords.from_islands(obj, obj_mode, self.resample, self.modulo, self.max_loops, max_isl=500, vert_sel=self.vert_sel)
        arm.generate_bones(islands_dict, self.reverse)
        if self.apply_weights: 
            ArmatureFuncs.auto_weight(islands_dict, arm.armature, obj, self.mix_mode)
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column(heading='', align=True)
        col.scale_x = 1.20
        col.scale_y = 1.20
        row_reverse = col.row(heading='', align=True)
        row_reverse.prop(self, 'reverse', text='', icon_value=(36 if self.reverse else 38), emboss=True)
        box_rev = row_reverse.box()
        box_rev.scale_x = 1.0
        box_rev.scale_y = 0.63
        box_rev.label(text='Reverse', icon_value=715)
        row_resample = col.row(heading='', align=True)
        row_resample.prop(self, 'resample_on', text='', icon_value=(39 if self.resample_on else 38), emboss=True)
        box_3C89A = row_resample.box()
        box_3C89A.scale_x = 1.0
        box_3C89A.scale_y = 0.63
        box_3C89A.label(text='Resample', icon_value=16)
        row_8E3B7 = row_resample.row(heading='', align=True)
        row_8E3B7.enabled = self.resample_on
        row_8E3B7.prop(self, 'resample', text='', icon_value=0, emboss=True)

        # Mesh
        col_8DE44 = col.column(heading='', align=True)
        row_66FE3 = col_8DE44.row(heading='', align=True)
        row_66FE3.prop(self, 'apply_weights', text='', icon_value=(36 if self.apply_weights else 38), emboss=True)
        row_C7688 = row_66FE3.row(heading='', align=True)
        box_79147 = row_C7688.box()
        box_79147.scale_x = 1.0
        box_79147.scale_y = 0.63
        box_79147.label(text='Apply weights', icon_value=475)
        row_C7688.prop(self, 'mix_mode', text='', icon_value=0, emboss=True)
        
        if 'ctools_armature' in bpy.context.preferences.addons:
            advanced_settings = bpy.context.preferences.addons['ctools_armature'].preferences.sna_advanced_settings
        elif bpy.context.scene.sna_addon_prefs_temp.sna_advanced_settings:
            advanced_settings = bpy.context.scene.sna_addon_prefs_temp.sna_advanced_settings
        else:
            advanced_settings = False
        
        if advanced_settings:
            col_E2B81 = col_8DE44.column(heading='', align=True)
            col_E2B81.alignment = 'Expand'.upper()
            col_E2B81.separator(factor=1.0)
            
            # Islands
            if True:
                row_27CC4 = col_E2B81.row(heading='', align=True)
                box_9F073 = row_27CC4.box()
                box_9F073.scale_x = 1.0
                box_9F073.scale_y = 0.63
                box_9F073.label(text='Start vertex algorithm', icon_value=0)
                row_27CC4.prop(self, 'vert_sel', text='', icon_value=0, emboss=True)
                
            row_A61CC = col_E2B81.row(heading='', align=True)
            row_A61CC.scale_x = 1.0
            row_A61CC.scale_y = 1.0
            box_36178 = row_A61CC.box()
            box_36178.scale_x = 1.0
            box_36178.scale_y = 0.63
            box_36178.label(text='Modulo', icon_value=0)
            row_A61CC.prop(self, 'modulo', text='', icon_value=0, emboss=True)
            row_F4484 = col_E2B81.row(heading='', align=True)
            box_C0076 = row_F4484.box()
            box_C0076.scale_x = 1.0
            box_C0076.scale_y = 0.63
            box_C0076.label(text='Max length', icon_value=0)
            row_F4484.prop(self, 'max_loops', text='', icon_value=0, emboss=True)

    def invoke(self, context, event):
        obj = bpy.context.object
        if not obj:
            self.report({'ERROR'}, message='No active object')
            return {"CANCELLED"}
        if not obj.visible_get():
            self.report({'ERROR'}, message='Object is hidden')
            return {"CANCELLED"}
        
        if obj.type == 'MESH' and bpy.context.mode == 'EDIT_MESH':
            if not obj.data.total_vert_sel:
                self.report({'ERROR'}, message='No vertices selected')
                return {"CANCELLED"}
        context.window_manager.invoke_props_popup(self, event)
        return self.execute(context)


def register():
    bpy.utils.register_class(CT_GenerateBonesCurves)
    bpy.utils.register_class(CT_GenerateBonesMesh)
    bpy.utils.register_class(CT_GenerateBonesIslands)

def unregister():
    bpy.utils.unregister_class(CT_GenerateBonesCurves)
    bpy.utils.unregister_class(CT_GenerateBonesMesh)
    bpy.utils.unregister_class(CT_GenerateBonesIslands)

