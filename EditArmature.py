import bpy
from bpy.types import Context, Event
import numpy as np
from mathutils import Vector
from collections import defaultdict
import importlib, os, sys
import bmesh

class ddict(defaultdict):
    __repr__ = dict.__repr__

#TODO - merge chains (average pos of "loop_bones", all weights of loop_bones to all verts affected by "loop_bones")
#todo - Resample chain 1.(separate mesh and reapply weights) or 2.(set new weights with math)
#todo     - order bones by root to tip and check if they are connected (stop if there are gaps)
#todo         1. Get meshes with weights, separate and run "auto weights" in GenerateArmature
#todo         2. Make weight paint algorithm

#NOTE Naming conventions:
#note EditFuncs - common operations 
#note editbones - functions for editing bones and transferring weights
#note     selected - run function on selected bones
#note     chain - run function on entire chain of selected bones (ex: select first bones of chains in the outliner)
#note     auto - complex operation

#SECTION ------------ Helpers ------------
class PoseModeWrapper:
    """
    Wrapper for running functions in pose mode if in edit mode, then switching back to edit mode
    - Undo steps are not stored properly in edit mode
    """
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        in_edit_mode = bpy.context.mode == 'EDIT_ARMATURE'
        if in_edit_mode:
            bpy.ops.object.mode_set(mode='POSE')
            print("Edit to pose")
        self.func(*args, **kwargs)
        if in_edit_mode:
            print("Pose to edit")
            bpy.ops.object.mode_set(mode='EDIT')

def import_module_from_file(filepath:str, module_name:str) -> object | None:
    try:
        file_dir = os.path.dirname(filepath)
        if os.path.exists(file_dir) and file_dir not in sys.path:
            sys.path.append(file_dir)
        if module_name in sys.modules:
            print("-- Existing")
            return sys.modules[module_name]
        print("-- Importing")
        return importlib.import_module(module_name)
    except ImportError as e:
        print("Import error -> platform:", sys.platform)
        return print(e)

def pprint_dict(d:ddict | dict, indent=0):
    """Pretty print a dictionary"""
    output = '{\n'
    for k, v in d.items():
        output += '  ' * (indent + 1)
        if isinstance(v, ddict | dict):
            output += f"'{k}': {pprint_dict(v, indent + 1)},\n"
        else:
            output += f"'{k}': {repr(v)},\n"
    output += '  ' * indent + '}'
    return output

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

def popup_window(title:str = "Error", text:str|list = "Error message", icon:str = 'ERROR'):
    """Minimal popup error - less intrusive than assert, easier to read than print"""
    def popup(self, context):
        # Each element in text is a line in the popup window
        lines = text if type(text) == list else text.split("\n") 
        for line in lines:
            row = self.layout.row()
            row.label(text=line)
    bpy.context.window_manager.popup_menu(popup, title=title, icon=icon) 

def interactive_popup(title:str = "Title", content:dict = {"Title":{"type":"label"}}, icon:str = 'NONE'):
    """Content structure:\n
    {
        \n\t"Label text":{
            \n\t\t"type":"label",
            \n\t\t"icon":"NONE"
        },
        \n\t"Button text":{
            \n\t\t"type":"button",
            \n\t\t"op":"object.select_all",
            \n\t\t"prop":{
                \n\t\t\t"prop_name": value
                \n\t\t},
            \n\t\t"icon":"NONE"
        },
        \n\t"Property text":{
            \n\t\t"type":"prop",
            \n\t\t"prop":"object.name",
            \n\t\t"icon":"NONE"
        }
    }"""
    
    def popup(self, context):
        # Each element in text is a line in the popup window
        self:bpy.types.Operator
        # <class '__main__.SNA_OT_Testop_E2553'>
        item:dict
        for text, item in content.items():
            item.setdefault("icon", "NONE")
            col = self.layout.column()
            if item["type"] == "label":
                col.label(text=text, icon=item["icon"])
            if item["type"] == "button":
                op = col.operator(item["op"], text=text, icon=item["icon"])
                for prop, value in item["prop"].items():
                    setattr(op, prop, value)
    bpy.context.window_manager.popup_menu(popup, title=title, icon=icon) 

def is_collection_excluded(obj:bpy.types.Object, unhide:bool=False) -> bool:
    """
    Unexclude & unhide collection if unhide is True\n
    Returns False if object is in any visible collection, else returns True
    """
    collections = [col for col in obj.users_collection if col != bpy.context.scene.collection]
    viewl_col = bpy.context.view_layer.layer_collection.children
    
    if obj.visible_get():
        return False
    if not collections: # Abort if object is somehow not visible when only in scene collection
        return True
    
    # Force show collection
    if unhide:
        collections[0].hide_viewport = False
        viewl_col[collections[0].name].hide_viewport = False
        viewl_col[collections[0].name].exclude = False
    
    # Check if object is in excluded collection
    visible_collection = [not any([
        col.hide_viewport, 
        viewl_col[col.name].hide_viewport, 
        viewl_col[col.name].exclude
        ]) for col in collections]
    
    # If object is in any visible collection, return False
    if any(visible_collection):
        return False
    else:
        return True

def clamp(value:float, a:float, b:float) -> float:
    return min(b, max(a, value))

#SECTION ------------ Common operations ------------
class EditFuncs:
    
    @staticmethod
    def init_bones(min_bones:int):
        """Returns: [armatures, selected bones, active bone]"""
        # Undo steps not stored properly in edit mode
        # ! Should be set in pose mode before arriving here
        assert bpy.context.mode in {'POSE', 'PAINT_WEIGHT'}, "Expected pose or weight paint mode"
        # if bpy.context.mode == 'EDIT_ARMATURE':
        #     bpy.ops.object.mode_set(mode='POSE')
        mode = bpy.context.mode
        
        obj = bpy.context.object
        valid_armatures = lambda mesh_obj: [mod.object for mod in mesh_obj.modifiers if mod.type == 'ARMATURE' and mod.object] # All armatures connected to mesh object
        arm_condition = lambda arm_obj: [ob for ob in bpy.data.objects if [mod for mod in ob.modifiers if mod.type == 'ARMATURE' and mod.object == arm_obj]] # All mesh objects connected to armature object
        selected_condition = lambda valid_objs: [ob for ob in bpy.context.selected_objects if ob != obj and ob in valid_objs] # All selected objects that are in 'valid objects'
        
        
        if mode == 'PAINT_WEIGHT':
            pose_bone = bpy.context.active_pose_bone
            armatures = EditFuncs.valid_armatures(obj)
            active_bone = pose_bone.id_data.data.bones[pose_bone.name] if pose_bone else None
        elif mode == 'POSE':
            armatures = bpy.context.objects_in_mode
            active_bone = bpy.context.active_bone
        else:
            return popup_window(title="Error", text="Must be in pose or weight paint mode", icon='ERROR')
        
        if not armatures:
            return popup_window(title="Error", text="No valid armatures found", icon='ERROR')
        
        sel_bones = [bone.id_data.data.bones[bone.name] for bone in bpy.context.selected_pose_bones]
        if len(sel_bones) < min_bones: 
            return popup_window(text=f"Select {min_bones} bones or more")
        
        return armatures, sel_bones, active_bone
    
    @staticmethod
    def set_active_bone(bone:object) -> None:
        """Set active bone and vertex group"""
        bone.select = True
        bone.id_data.bones.active = bone
        if bpy.context.mode == 'PAINT_WEIGHT':
            obj = bpy.context.object
            vgrps = obj.vertex_groups
            vgrps.active_index = vgrps[bone.name].index
    
    @PoseModeWrapper
    @staticmethod
    def cleanup_bones(bones:list):
        """Remove bones"""
        def remove_bones():
            for bone in bones:
                armature = bone.id_data.edit_bones
                edit_bone = armature[bone.name]
                armature.remove(edit_bone)
                
        if bpy.context.mode == 'PAINT_WEIGHT':
            obj = bpy.context.object
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = bpy.context.object.parent
            bpy.ops.object.mode_set(mode='EDIT')
            remove_bones()
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        else:
            bpy.ops.object.mode_set(mode='EDIT')
            remove_bones()
            bpy.ops.object.mode_set(mode='POSE')
    
    @PoseModeWrapper
    @staticmethod
    def cleanup_bones_dissolve(bones_dict:dict) -> None:
        """Dissolve bones"""
        def dissolve_bones():
            for target_bone, children in bones_dict.items():
                if len(children) < 1: continue
                
                # Edit bones for editing armature (Edit mode is required)
                edit_bones:bpy.types.ArmatureEditBones = target_bone.id_data.edit_bones
                parent = edit_bones[target_bone.name]
                children_recursive = parent.children_recursive
                
                # Disconnect unselected children, if target bone has more than 1 child
                if len(target_bone.children) > 1:
                    for target_child in target_bone.children:
                        if target_child in children: continue
                        child_bone = edit_bones[target_child.name]
                        edit_bones[target_child.name].use_connect = False
                
                for i, child in enumerate(children):
                    child_bone = edit_bones[child.name]
                    # Store tail location of last child
                    if i == len(children) - 1:
                        location = child_bone.tail
                    # Store connected bones
                    connect = [bone for bone in child_bone.children if bone.use_connect]
                    # Remove child bone
                    children_recursive.remove(child_bone)
                    edit_bones.remove(child_bone)
                    for bone in connect:
                        bone.use_connect = True
                
                # Set target bone tail location
                parent.tail = location
        
        if bpy.context.mode == 'PAINT_WEIGHT':
            obj = bpy.context.object
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = bpy.context.object.parent
            bpy.ops.object.mode_set(mode='EDIT')
            dissolve_bones()
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        else:
            bpy.ops.object.mode_set(mode='EDIT')
            dissolve_bones()
            bpy.ops.object.mode_set(mode='POSE')
    
    @staticmethod
    def cleanup_vertex_groups(sel_bones:list, objects:list) -> None:
        """Remove vertex groups"""
        for obj in objects:
            for bone in sel_bones:
                if bone.name not in obj.vertex_groups: continue
                vgrp = obj.vertex_groups[bone.name]
                obj.vertex_groups.remove(vgrp)
    
    @staticmethod
    def objects_from_bones(armatures:list, bones:list) -> list:
        """Returns a list of objects connected to selected bones"""
        assert len(bones), "No bones"
        objects = set()
        # Check every object in scene
        for obj in bpy.context.scene.objects:
            # Skip objects that are not meshes or have no modifiers
            if obj.type != 'MESH': continue
            if not obj.modifiers: continue
            # If "obj" has an "armature modifier" that is linked to any in "armatures" -> add to "objects"
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE' and mod.object in armatures:
                    objects.add(obj)
        return objects
    
    @staticmethod
    def weights_from_objects(objects:bpy.types.Object) -> dict:
        """Weights of object returned as a dict\n
        Returns: {obj: {grp_index: {vertex_pointer:grp_weight, ...}, ...}, ...}"""
        obj_weights = {}
        for obj in objects:
            verts = obj.data.vertices
            groups = obj.vertex_groups

            grps_dict = {grp.index: {} for grp in groups}
            for vertex in verts:
                group_weights = [float]*len(vertex.groups) #grp_weight
                group_indices = [int]*len(vertex.groups) #vertex_pointer
                vertex.groups.foreach_get("group", group_indices)
                vertex.groups.foreach_get("weight", group_weights)
                
                for i, grp_index in enumerate(group_indices):
                    # group_index -> {vertex_pointer: group weight, ...}
                    grps_dict[grp_index].setdefault(vertex, group_weights[i])
                    
            obj_weights.setdefault(obj, grps_dict)
        return obj_weights
    
    @staticmethod
    def filter_from_bones(obj_weights:dict, bones:list) -> dict:
        """Filters out weights not connected to selected bones and replaces group index with pointer"""
        out_weights = {}
        for obj, weights in obj_weights.items():
            out_weights.setdefault(obj, {})
            for bone in bones:
                if bone.name not in obj.vertex_groups: 
                    continue
                vgrp = obj.vertex_groups[bone.name] 
                weight = weights[vgrp.index]
                out_weights[obj].setdefault(vgrp, weight)
        
        return out_weights
    
    @staticmethod
    def bones_chain(bones:list) -> set:
        """Return list including parents and children of bone, invalid when forks are detected ###stop at forks"""
        # For every bone, check if parent or children > 1
        bones = [(bone, *bone.children_recursive, *bone.parent_recursive) for bone in bones]
        bones = {bone for bones in bones for bone in bones} # Flatten and remove duplicates
        return bones

    @staticmethod
    def bones_linked(bones:list) -> set:
        """Gets parents and children of bone, Stops at bones without "Connected" parent"""
        # Built in linked
        def auto():
            assert bpy.context.mode == 'EDIT_ARMATURE', "Expected edit mode"
            if bpy.context.mode == 'EDIT_ARMATURE':
                bpy.ops.armature.select_linked(all_forks=False)
                return bpy.context.use_selected_bones
            if bpy.context.mode == 'POSE':
                bpy.ops.pose.select_linked()
                return bpy.context.selected_pose_bones
            
        # Manual linked
        def manual():
            linked = []
            #TODO - bone.id_data / bone.id_data.data -> use_connect
            for bone in bones:
                parent = [bone, *bone.parent_recursive]
                children = bone.children_recursive
                # Parent (stop at connected)
                for i, bone in enumerate(parent):
                    if bone.use_connect:
                        continue
                    parent = parent[:i+1]
                    break
                # Children (stop at connected)
                for i, bone in enumerate(children):
                    if bone.use_connect:
                        continue
                    children.remove(bone)
                    [children.remove(x) for x in bone.children_recursive]
                    
                parent.reverse()
                linked = [*parent, *children]
                linked = [item for index, item in enumerate(linked) if item not in linked[:index]] # Remove duplicates
            return linked
        
        linked = manual()
        
        # Check forks
        forks = [bone for bone in linked if len(bone.children) > 1]
        # print(f"Forks: {len([child.use_connect for forks in forks for child in forks.children])}")
        for fork in forks:
            for child in fork.children:
                if child.use_connect:
                    return
            if len([child.use_connect for child in fork.children]):
                return
        
        return linked
    
    @staticmethod
    def organize_bones(bones:list) -> dict:
        """Return bones that have a valid parent as keys and their children as values"""
        def find_key(dict, value):
            for key, values in dict.items():
                if value in values:
                    return key
            return None
        
        valid_bones = {}
        for bone in bones:
            # If the bone.parent has a valid parent, add the bone to the parent's list of children
            valid_parent = find_key(valid_bones, bone.parent)
            if valid_parent:
                valid_bones.setdefault(valid_parent, []).append(bone)
                continue
            
            # Add bone to valid bones
            valid_bones.setdefault(bone.parent, []).append(bone)
            
            # Check if bone is in keys
            if bone in valid_bones:
                # Move bone values to bone.parent values
                valid_bones[bone.parent] += [*valid_bones.pop(bone)]
                
        return valid_bones
    
    @staticmethod
    def transfer_weights(obj_weights:dict, target_group:object, filter_bones:list) -> None:
        """Transfer vertex groups of corresponding object to target bone's vertex group"""
        obj_weights = EditFuncs.filter_from_bones(obj_weights, filter_bones)

        for obj, weights in obj_weights.items():                # Object: {Vertex group: {Group index: {vertex: weight, ...}, ...}, ...}
            vgroups = obj.vertex_groups
            for vertex_group, vgrp_data in weights.items():     # Vertex group: {Group index: {vertex: weight, ...}, ...}
                if vertex_group.name == target_group.name: continue
                for vertex, weight in vgrp_data.items():        # Group index: {vertex: weight, ...}
                    # Remove vertex groups
                    vertex_group.remove([vertex.index])
                    
                    # Create vertex group if not exists
                    if target_group.name not in vgroups: 
                        vgroups.new(name=target_group.name)
                    
                    # Add weight to active bone
                    vgroups[target_group.name].add([vertex.index], weight, 'ADD')

    @staticmethod
    def nearest_bone_weights(obj_weights:dict, bones:list) -> None:
        """Returns weights of closest bones to verts"""
        newdict = {}    # {"obj": {"vertex": {"vertex_group": "weight"}}}
        for obj, grp_weights in obj_weights.items():
            newdict.setdefault(obj, {})
            for vertex_group, weights in grp_weights.items():
                for vertex, weight in weights.items():
                    newdict[obj].setdefault(vertex, {})
                    newdict[obj][vertex].setdefault(vertex_group, weight)
        
        def calculate_percentage(numbers):
            total_count = len(numbers)
            count_dict = {}
            for number in numbers:
                if number in count_dict:
                    count_dict[number] += 1
                else:
                    count_dict[number] = 1
            percentage_dict = {number: (count / total_count) * 100 // 1 for number, count in count_dict.items()}
            percentage = lambda x: f"{int(x)}%"
            sorted_dict = {k: percentage(v) for k, v in sorted(percentage_dict.items(), key=lambda item: item[1], reverse=True)}
            return sorted_dict
        
        influence_count = [len(weights) for verts in newdict.values() for weights in verts.values()]
        percentages = calculate_percentage(influence_count)
        recursive_dict_print(percentages)

    @staticmethod
    def cleanup_armature_modifiers(obj=None, armature=None) -> None:
        """Cleanup armature modifiers and create a new one if doesn't exists"""
        obj = obj if obj else bpy.context.object
        armature_modifers = [mod for mod in obj.modifiers if mod.type == 'ARMATURE']
        for mod in armature_modifers:
            if mod.type != 'ARMATURE': continue
            if not mod.object: 
                bpy.ops.object.modifier_remove(modifier=mod.name)
        valid_mods = [mod for mod in armature_modifers if mod.object]
        # Use armature parameter if given
        if armature and len(valid_mods) < 1:
            obj.modifiers.new(name='Armature', type='ARMATURE').object = armature

    @staticmethod
    def connected_armatures(obj:bpy.types.Object=None) -> list:
        """All armatures connected to active object"""
        obj = bpy.context.object if not obj else obj
        if obj.type != 'MESH': return
        return [mod.object for mod in obj.modifiers if mod.type == 'ARMATURE' and mod.object]
    
    @staticmethod
    def connected_meshes(obj:bpy.types.Object=None) -> list:
        """All mesh objects connected to active object"""
        obj = bpy.context.object if not obj else obj
        if obj.type != 'ARMATURE': return
        return [ob for ob in bpy.data.objects if [mod for mod in ob.modifiers if mod.type == 'ARMATURE' and mod.object == obj]]

    @staticmethod
    def valid_armatures(mesh_obj:bpy.types.Object) -> list:
        """All armatures connected to mesh object"""
        assert mesh_obj.type == 'MESH', "Expected mesh object"
        return [mod.object for mod in mesh_obj.modifiers if mod.type == 'ARMATURE' and mod.object]

    @staticmethod
    def valid_meshes(arm_obj:bpy.types.Object) -> list:
        """All mesh objects connected to armature object"""
        assert arm_obj.type == 'ARMATURE', "Expected armature object"
        return [ob for ob in bpy.data.objects if [mod for mod in ob.modifiers if mod.type == 'ARMATURE' and mod.object == arm_obj]]

    @staticmethod
    def selected_in_objects(objects:list, obj:bpy.types.Object=None) -> list:
        """All selected objects that are in 'objects'\n
        obj: bpy.context.object"""
        obj = obj if obj else bpy.context.object
        return [ob for ob in bpy.context.selected_objects if ob != obj and ob in objects]


#SECTION ------------ Edit Functions ------------
@PoseModeWrapper
def editbones_selected_to_active() -> None:
    """ Transfer weights from selected bones to active bone
        Delete selected bones =! active bone"""
    init_vars = EditFuncs.init_bones(2)
    if not init_vars: return
    armatures, sel_bones, active_bone = init_vars
    if not active_bone: 
        return popup_window(title="Error", text="No active bone", icon='ERROR')
    
    objects = EditFuncs.objects_from_bones(armatures, sel_bones)
    obj_weights = EditFuncs.weights_from_objects(objects)
    EditFuncs.transfer_weights(obj_weights, active_bone, sel_bones)
    
    # Cleanup bones and vertex groups
    if active_bone in sel_bones:
        sel_bones.remove(active_bone)
    EditFuncs.set_active_bone(active_bone)
    EditFuncs.cleanup_vertex_groups(sel_bones, objects)
    EditFuncs.cleanup_bones(sel_bones)
    return

@PoseModeWrapper
def editbones_create_root_bone() -> None:
    """
    Create new bone at the centre of selected bones (pointing up, scale relative to child/parent bones?)
    Store parents of selected bones as "parent_bones"
    Option: Transfer weights from selected bones to root bone, then remove + cleanup groups
    Parent all bones in "parent_bones" to root bone
    """
    # def average_coords():
    #     """ Returns average of list of vectors """
    #     head_coords = [bone.head_local for bone in sel_bones]
    #     tail_coords = [bone.tail_local for bone in sel_bones]
    #     bone_length = np.average([bone.length for bone in sel_bones])
    #     head = Vector(np.average(head_coords, 0))
    #     tail = head + Vector((0, 0, bone_length))
        
    #     active_bone = bpy.context.active_bone
    #     active_bone.head = head
    #     active_bone.tail = tail
    #     name = active_bone.name.split('.')[0]
    #     active_bone.name = active_bone.name + '_root'
    # Object mode - Use all root bones in armature
    # Edit mode - Selected bones -> linked -> root bones
    # Transfer to first child bone
    # Object mode / no bones selected - Use all bones with no parent
    
    init_vars = EditFuncs.init_bones(2)
    if not init_vars: return
    armatures, sel_bones, active_bone = init_vars
    
    objects = EditFuncs.objects_from_bones(armatures, sel_bones)
    obj_weights = EditFuncs.weights_from_objects(objects)
    return

@PoseModeWrapper
def editbones_selected_remove() -> None:
    """
    Transfer weights to parent bone
    If no parent, transfer to child bone
    """
    
    #TODO - No parent -> error
    init_vars = EditFuncs.init_bones(1)
    if not init_vars: return
    armatures, sel_bones, active_bone = init_vars
    
    objects = EditFuncs.objects_from_bones(armatures, sel_bones)
    obj_weights = EditFuncs.weights_from_objects(objects)
    
    sel_bones = [bone for bone in sel_bones if bone.parent]
    if not sel_bones: 
        return popup_window(title="Error", text="No valid bones", icon='ERROR')
    
    bones_dict = EditFuncs.organize_bones(sel_bones)
    for target_bone, children in bones_dict.items():
        EditFuncs.transfer_weights(obj_weights, target_bone, children)
    
    # Set active bone
    activebone = next(iter(bones_dict))
    EditFuncs.set_active_bone(activebone)
    
    # Clean up bones and vertex groups
    cleanup_bones = [item for sublist in bones_dict.values() for item in sublist]
    EditFuncs.cleanup_vertex_groups(cleanup_bones, objects)
    EditFuncs.cleanup_bones(cleanup_bones)
    return

@PoseModeWrapper
def editbones_selected_dissolve() -> None:
    #Note Check if bone.parent - > return if not
    #Note Cannot run if bone.parent.children > 1 (Can be fixed by setting connected to false to children that are not in selected)
    #Note
    #Note Store bone head and connected
    #Note Delete selected bones
    #Note For every child of bone, set head to parent head loc, set connected to True
    
    #TODO - Doesn't run when no children
    #TODO - When more than one child
    #todo   -> If   -> more than one selected in children -> Error
    #todo   -> else -> "use_connect = False" for unselected children -> set unselected parent to tail of dissolved bone
        
    init_vars = EditFuncs.init_bones(1)
    if not init_vars: return
    armatures, sel_bones, active_bone = init_vars
    
    objects = EditFuncs.objects_from_bones(armatures, sel_bones)
    obj_weights = EditFuncs.weights_from_objects(objects)
    
    # Selected bones that have a parent
    sel_bones = [bone for bone in sel_bones if bone.parent]
    if not sel_bones: 
        return popup_window(title="Error", text="No valid bones selected", icon='ERROR')
    
    bones_dict = EditFuncs.organize_bones(sel_bones)
    for target_bone, children in bones_dict.items():
        children = [child for child in children if child.use_connect]
        bones_dict[target_bone] = children
        if not children: 
            print("No children")
            continue
        
        # If target bone has more than one child, check if selected children share the same parent
        if len(target_bone.children) > 1:
            # Selected children that are direct children of target bone
            selected_children = [c for c in target_bone.children if c in children]
            # If selected children share target bone as parent, remove from operation
            if len(selected_children) > 1:
                bones_dict[target_bone].clear()
                print("Invalid action")
                continue
        
        EditFuncs.transfer_weights(obj_weights, target_bone, children)
    
    # Set active bone
    bones_dict = {key:val for key, val in bones_dict.items() if val}
    if not bones_dict:
        return popup_window(title="Error", text="No valid bones selected for dissolve operator", icon='ERROR')
    activebone = next(iter(bones_dict))
    EditFuncs.set_active_bone(activebone)
    
    # Clean up bones and vertex groups
    EditFuncs.cleanup_vertex_groups(sel_bones, objects)
    EditFuncs.cleanup_bones_dissolve(bones_dict)
    return

# TODO
# def editbones_chain_resample() -> None:
#     def resample_coords(coords:list[Vector], res:int) -> list[Vector]:
#         """Resample point coordinates"""
#         length_list = [math_dist(coords[i], coords[i+1]) for i in range(len(coords)-1)]
#         length:int = np.sum(length_list)
#         cum_sum = np.insert(np.cumsum(length_list), 0, 0)
#         new_coords = []
#         for i in range(res):
#             target = length * (i / (res - 1))
#             i1:int = np.where(cum_sum <= target)[0][-1]
#             if i1 >= len(length_list):
#                 factor = 1
#                 i1 = i1 - 1
#             else:
#                 factor = (target - cum_sum[i1]) / length_list[i1]
#             point = coords[i1].lerp(coords[i1+1], factor)
#             new_coords.append(point)
#         return new_coords
    
#     def resample_coords_simple(coords, res):
#         new_coords = []
#         num_points = len(coords) - 1
#         segment_length = 1 / (res-1)

#         for i in range(res-1):
#             t = i * segment_length
#             index = int(t * num_points)
#             frac = (t * num_points) % 1

#             point_a = coords[index]
#             point_b = coords[index + 1]

#             interpolated_point = (1 - frac) * point_a + frac * point_b
#             new_coords.append(interpolated_point)

#         new_coords.append(coords[-1])  # Include the last coordinate
#         return new_coords
    
#     def resample_bones(bones, new_coords):
#         bpy.ops.object.mode_set(mode='EDIT')
#         armature = bones[0].id_data
#         new_bones = []
#         for i, bone in enumerate(bones):
#             armature.edit_bones.remove(armature.edit_bones[bone.name])
#         for i, co in enumerate(new_coords[1:]):
#             new_bone = armature.edit_bones.new("Bone{}".format(i))
#             new_bone.head = new_coords[i]
#             new_bone.tail = co
#             new_bone.parent = new_bones[i-1] if i else None
#             new_bones.append(new_bone)
#         return new_bones
    
#     def new_weights(weights, bones, new_bones):
#         # Get current distribution factor of bones (ex. {0.15, 0.7, 0.15} vs {0.6, 0.4})
#         # For every vertex, get the (2) closest bone(s)
#         #   Get the distance from the vertex to the bone
#         #   Compare the length of the resampled bones to the distance
#         ...
    
#     bpy.ops.object.mode_set(mode='POSE')
    
#     init_vars = EditFuncs.init_bones(1)
#     if not init_vars: return
#     armatures, sel_bones, active_bone = init_vars
    
#     objects = EditFuncs.objects_from_bones(armatures, sel_bones)
#     bones = EditFuncs.bones_linked(sel_bones)
#     weights = EditFuncs.weights_from_objects(objects)
    
#     EditFuncs.nearest_bone_weights(weights, bones)
#     return
#     mode = bpy.context.mode
#     coords_mode = {
#         'EDIT_ARMATURE': lambda x: [*[bone.head for bone in x], x[-1].tail],
#         'POSE': lambda x: [*[bone.head_local for bone in x], x[-1].tail_local],
#     }
#     coords = coords_mode[mode](bones)
#     coords = resample_coords(coords, 8)
#     bones = resample_bones(bones, coords)

#TODO
# def editbones_chain_merge() -> None:
#     # Starting at set index of chains, merge chains by distance
#     ...

#TODO
# def editbones_chain_reverse() -> None:
#     # Reverse selected chains
#     ...

@PoseModeWrapper
def editbones_chain_trim(start:float=0.0, end:float=1.0) -> None:
    """Trim selected chains from start and end"""
    # Clamp start and end
    min_offset = 0.1 # A minimum offset to prevent errors from negative values and all bones from being removed
    end = max(start + min_offset, end)
    end = min(end, 1.0)

    start = min(start, end - min_offset)
    start = max(start, 0.0)
    
    init_vars = EditFuncs.init_bones(1)
    if not init_vars: return
    armatures, sel_bones, active_bone = init_vars
    
    objects = EditFuncs.objects_from_bones(armatures, sel_bones)
    linked_bones = EditFuncs.bones_linked(sel_bones)
    
    if not linked_bones: 
        return popup_window(title="Error", text="Invalid bones", icon='ERROR')

    def assign_factors(items:list) -> list[float]:
        num_items = len(items)
        factors = []

        for i in range(num_items):
            factor = i / (num_items - 1)  # Calculate the factor based on the item's position
            factors.append(factor)
        return factors

    factors = assign_factors(linked_bones)

    start_bones = []
    end_bones = []
    start_bone = linked_bones[0]
    end_bone = linked_bones[-1]
    
    # Head (Start)
    for i, bone in enumerate(linked_bones):
        fac = factors[i] * 100 // 1 / 100
        if fac >= start: break
        start_bones.append(bone)
        start_bone = bone.children[0]
    
    # Tail (End)
    for i, bone in enumerate(linked_bones[::-1]):
        fac = factors[::-1][i] * 100 // 1 / 100
        if fac <= end: break
        end_bones.append(bone)
        end_bone = bone.parent

    objects = EditFuncs.objects_from_bones(armatures, [start_bone, *start_bones, end_bone, *end_bones])
    weights = EditFuncs.weights_from_objects(objects)
    
    EditFuncs.transfer_weights(weights, start_bone, start_bones)
    EditFuncs.transfer_weights(weights, end_bone, end_bones)
    
    cleanup_bones = start_bones + end_bones
    EditFuncs.cleanup_bones(cleanup_bones)
    EditFuncs.cleanup_vertex_groups(cleanup_bones, objects)
    
    EditFuncs.set_active_bone(start_bone)
    return

#def editbones_auto_optimize() -> None:
#    # Automatic version of chain_merge
#    return

#TODO
#def auto_finalize():
#    # Parent all root bones to a chosen "Root Bone"
#    # Fill empty groups
#   
#    # Edit mode
#    # Select all root bones
#    # Set root bone as active
#    # Parent to root bone
#   
#    # Weight paint mode
#    # Select all
#    # Un-select root bone
#    # Lock selected
#   
#    # Clear selected
#    # Select root bone
#    return




#SECTION ------------ ArmatureMode ------------
class ArmatureMode:
    @staticmethod
    def set_mode_auto(mode:str, arm_obj:bpy.types.Object, mesh_obj:bpy.types.Object = None, extend:bool = False):
        match mode:
            case 'OBJECT':
                ArmatureMode.object(arm_obj, mesh_obj, extend)
            case 'EDIT':
                ArmatureMode.edit(arm_obj, mesh_obj, extend)
            case 'POSE':
                ArmatureMode.pose(arm_obj, mesh_obj, extend)
            case 'PAINT_WEIGHT':
                ArmatureMode.weight_paint(arm_obj, mesh_obj, extend, select=False)
            case 'SELECT_WEIGHT':
                ArmatureMode.weight_paint(arm_obj, mesh_obj, extend, select=True)
            case 'MESH_OBJECT':
                ArmatureMode.mesh_object(arm_obj, mesh_obj, extend)

    @staticmethod
    def object(obj:bpy.types.Object, _, extend:bool = False) -> None:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_pattern(pattern=obj.name, extend=extend)
        bpy.context.view_layer.objects.active = obj
        
    @staticmethod
    def edit(obj:bpy.types.Object, _, extend:bool = False) -> None:
        assert obj.type == 'ARMATURE', "Expected armature object"
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_pattern(pattern=obj.name, extend=extend)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        
    @staticmethod
    def pose(obj:bpy.types.Object, _, extend:bool = False) -> None:
        assert obj.type == 'ARMATURE', "Expected armature object"
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_pattern(pattern=obj.name, extend=extend)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='POSE')
    
    @staticmethod
    def weight_paint(arm_obj:bpy.types.Object, mesh_obj:bpy.types.Object, extend:bool = False, select:bool = False) -> None:
        assert mesh_obj.type == 'MESH', "Expected mesh object"
        assert arm_obj.type == 'ARMATURE', "Expected armature object"
        
        # Set object mode
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
       
        # Select armature and mesh, set mesh as active
        bpy.ops.object.select_pattern(pattern=arm_obj.name, extend=extend)
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        
        # Mesh obj sometimes enters weight paint mode when set as active?
        bpy.ops.object.mode_set(mode='OBJECT') 
        
        # Set mode
        bpy.ops.paint.weight_paint_toggle()
        bpy.ops.wm.tool_set_by_id(name="builtin.move" if select else "builtin_brush.Draw")
        
    @staticmethod
    def mesh_object(_, mesh_obj:bpy.types.Object, extend:bool = False) -> None:
        assert mesh_obj.type == 'MESH', "Expected mesh object"
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_pattern(pattern=mesh_obj.name, extend=extend)
        bpy.context.view_layer.objects.active = mesh_obj
        # Mesh obj sometimes enters weight paint mode when set as active?
        bpy.ops.object.mode_set(mode='OBJECT')

import timeit
class WithTimer:
    def __init__(self, name:str = "Timer"):
        self.start = None
        self.time = None
        self.name = name
        
    def __enter__(self):
        self.start = timeit.default_timer()
        return self
    
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.time = timeit.default_timer() - self.start
        print(f"\n{self.name}:\n\t{self.time}")
        self.start = None


#SECTION ------------ Generate Vertex Weight ------------

def generate_vertex_weights_init(filepath:str, module_name:str) -> list:
    selected_bones_only = False
    selected_verts_only = False
    
    # ['EDIT_ARMATURE', 'POSE', 'PAINT_WEIGHT', 'EDIT_MESH']
    # if mode not in ['OBJECT', 'EDIT_MESH', 'PAINT_WEIGHT']:
    #     return popup_window(text="Invalid mode")
    # return popup_window(text="Not enough arguments")
    
    C = bpy.context
    obj = C.object
    arm_obj = C.pose_object
    
    if not obj or obj.type != 'MESH': 
        return popup_window(text="Invalid object. Select a mesh object")
    if bpy.context.mode != 'PAINT_WEIGHT': # Paint weight mode only
        return popup_window(text="Only weight paint mode is currently supported")
    if not arm_obj:
        return popup_window(text="No armature selected")
    
    
    file_dir = os.path.dirname(filepath)
    # import_module = module_name not in sys.modules
   
    if os.path.exists(file_dir):
        try:
            if file_dir not in sys.path:
                sys.path.append(file_dir)
            importlib.import_module(module_name)
            print(f"{module_name} imported")
            importlib.import_module("ctools_init_weights")
            print(f"ctools_init_weights imported")
        except ImportError as e: 
            print(f"!! Error: Import_lib failed: !!\n", e)
            pass
    
    valid_armatures = EditFuncs.valid_armatures(obj)
    selected_armatures = EditFuncs.selected_in_objects(valid_armatures)
    valid_armatures = selected_armatures if selected_armatures else valid_armatures
    
    if not valid_armatures:
        return popup_window(text="Mesh has no connected armature")
    
    #TODO - Temp (Only uses first valid armature. Should prompt user to select one if more than one is valid)
    # arm_obj = valid_armatures[0]
    
    assert arm_obj in valid_armatures, "I don't know how you managed to do this, so please tell me how you did it if you want me to fix it"
    
    # Selected bones
    sel_bones = [bone.id_data.data.bones[bone.name] for bone in bpy.context.selected_pose_bones]
    
    # Set first bone as active if none are selected
    if not sel_bones:
        arm_obj.data.bones[0].select = True
        arm_obj.data.bones.active = arm_obj.data.bones[0]
    
    bones = arm_obj.data.bones
    bones = sel_bones if selected_bones_only else bones
    posebones = [arm_obj.pose.bones[bone.name] for bone in bones]
    verts = obj.data.vertices
    verts = [vert for vert in verts if vert.select] if selected_verts_only else verts[:]
    
    if not bones: return popup_window(text="No bones selected")
    if not verts: return popup_window(text="No vertices selected")
    
    
    ctools_weights = sys.modules.get(module_name)
    import weight_algorithms.src
    bonesholder = ctools_weights.create_bones_array(posebones)
    vertsholder = ctools_weights.create_verts_array(verts)
    
    EditFuncs.cleanup_armature_modifiers(armature=arm_obj)
    return bonesholder, vertsholder

def generate_vertex_weights(args:list, power:float=3.0, threshold:float=0.01) -> None:
    #TODO - Option: Preserve "MeshIsland" --> "VertexGroup" Associations
    
    #TODO - Options in edit_mesh: (Default = Selected vertices)
    #TODO   - All verts | Selected verts | Linked mesh islands        -> use associated bones
    #TODO   - Only use the closest bone and it's linked chain
    
    #TODO - Options in [edit_bones, pose_bones, paint_weight]: (Default = All bones)
    #TODO   - All bones | Selected bones | Linked bone chains         -> use associated mesh island 
    
    #todo - check for locked bones
    #todo - 
    # Inputs:
    #   List of vertices to be affected
    #       List of vertex groups used by vertices
    #   List of bones to be affected
    
    
    C = bpy.context
    obj = C.object
    # Check args
    assert args, "Not enough arguments"
    bonesholder, vertsholder = args
    assert bonesholder, "No bones selected"
    assert vertsholder, "No vertices selected"
    
    power = clamp(power, 1.0, 10.0)
    threshold = clamp(threshold, 0.0, 1.0)
    
    ctools_weights = sys.modules.get("ctools_weights")
    if ctools_weights:
        # with WithTimer("New - Calculate vertex weights"):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        ctools_weights.distance_bones_verts(bm, vertsholder, bonesholder, power, threshold)
        bm.free()
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    else:
        return popup_window(text="New library not found")
        # print("New library not found, using old method")
        # # with WithTimer("Old Calculate vertex weights (slow)"):
        # weights = Algo.calculate_vertex_weights(bones, verts, power, threshold)
    
        # vgroups = obj.vertex_groups
        # vgroups.clear()
        # for vert_idx, bone_weights in weights.items():
        #     for bone_idx, weight in bone_weights.items():
        #         bone_name = bones[bone_idx].name
        #         if bone_name not in vgroups:
        #             vgroups.new(name=bone_name)
        #         vgroups[bone_name].add([vert_idx], weight, 'ADD')
    
    EditFuncs.set_active_bone(C.active_pose_bone.bone)

def resample_bone_weights():
    C = bpy.context
    selected_bones = C.selected_pose_bones
    vgroups = C.object.vertex_groups
    sel_groups = [vgroups[group.name].index for group in selected_bones]
    sel_verts = [v for v in C.object.data.vertices if v.select]
    
    weights = {}
    # {grp_idx: {vert_idx: weight}}
    for v in sel_verts:
        for grp in v.groups:
            if grp.group not in sel_groups: continue
            weights.setdefault(grp.group, {})
            weights[grp.group][v.index] = grp.weight
    print("----")
    sum_weights = {}
    # {vert_idx: sum_weights}
    for grp, weights in weights.items():
        for vert, weight in weights.items():
            sum_weights.setdefault(vert, 0)
            sum_weights[vert] += weight
    
    print(sum_weights)


#SECTION ------------ Operators ------------
class CT_OT_Init_SetPaintMode(bpy.types.Operator):
    bl_idname = "ct.init_set_paint_mode"
    bl_label = "Init Set Paint Mode"
    bl_description = "Init popup for each valid object or run directly if only one valid object is found"
    bl_options = {"REGISTER", "UNDO"}
    
    unhide: bpy.props.BoolProperty(name='Unhide', default=True, options={'HIDDEN'}) # type: ignore
    extend: bpy.props.BoolProperty(name='Extend', default=False, options={'HIDDEN'}) # type: ignore
    set_mode: bpy.props.EnumProperty(name='Object Mode', options={'HIDDEN'}, items=[
        ('OBJECT', 'OBJECT', '', 0, 0), 
        ('EDIT', 'EDIT', '', 0, 1), 
        ('POSE', 'POSE', '', 0, 2), 
        ('PAINT_WEIGHT', 'PAINT_WEIGHT', '', 0, 3),
        ('SELECT_WEIGHT', 'SELECT_WEIGHT', '', 0, 4),
        ('MESH_OBJECT', 'MESH_OBJECT', '', 0, 5)
        ]) # type: ignore
    
    def execute(self, context):
        C = bpy.context
        obj = C.object
        arm_obj = None
        mesh_obj = None
        
        if obj.type not in ['MESH', 'ARMATURE']: self.report({'ERROR'}, f"Invalid object type '{obj.type}'"); return {"CANCELLED"}
        if self.is_hidden(obj, self.unhide): return {"CANCELLED"}
        
        if self.unhide:
            self.unhide_object(obj)
            if not obj.visible_get():
                return {"CANCELLED"}
        
        if obj.type == 'ARMATURE':
            arm_obj = C.object
            
            if self.set_mode in ['OBJECT', 'EDIT', 'POSE']:
                ArmatureMode.set_mode_auto(self.set_mode, obj, None, self.extend); return {"FINISHED"}
            
            elif self.set_mode in ['PAINT_WEIGHT', 'SELECT_WEIGHT', 'MESH_OBJECT']:
                # Get all meshes connected to armature
                valid_meshes = EditFuncs.valid_meshes(arm_obj)
                if not valid_meshes:
                    self.report({'ERROR'}, f"No valid meshes connected to '{obj.name}'"); print("test"); return {"CANCELLED"}
                    popup_window(text=f"No valid meshes connected to '{obj.name}'"); return {"CANCELLED"}
                    
                
                # Check if any valid meshes are selected and use them if they are
                valid_selected = EditFuncs.selected_in_objects(valid_meshes)
                valid_meshes = valid_selected if valid_selected else valid_meshes
                
                # Prompt user to select a mesh if more than one is valid
                if len(valid_meshes) > 1:
                    self.select_menu(valid_meshes); return {"FINISHED"}
                else:
                    mesh_obj = valid_meshes[0]
                    if self.is_hidden(mesh_obj, self.unhide): return # Cancel if mesh is hidden
                    print(self.set_mode, mesh_obj.name)
                    ArmatureMode.set_mode_auto(self.set_mode, arm_obj, mesh_obj, self.extend); return {"FINISHED"}
        
        if obj.type == 'MESH':
            arm_obj = None
            if self.set_mode != 'MESH_OBJECT':
                # Get all armatures connected to mesh
                valid_armatures = EditFuncs.valid_armatures(obj)
                if not valid_armatures:
                    self.report({'ERROR'}, f"No valid armatures connected to '{obj.name}'"); print("test"); return {"CANCELLED"}
                    #return popup_window(text=f"No valid armatures connected to '{obj.name}'")
                
                # Check if any valid armatures are selected and use them if they are
                valid_selected = EditFuncs.selected_in_objects(valid_armatures)
                valid_armatures = valid_selected if valid_selected else valid_armatures
                
                # Prompt user to select an armature if more than one is valid
                if len(valid_armatures) > 1: # Exits if true
                    self.select_menu(valid_armatures); return {"FINISHED"}
                else:
                    arm_obj = valid_armatures[0]
                    if self.is_hidden(arm_obj, self.unhide): # Exits if true
                        return {"CANCELLED"}
            
            ArmatureMode.set_mode_auto(self.set_mode, arm_obj, obj, self.extend)
    
        return {"FINISHED"}
    
    """["OBJECT", "EDIT", "POSE", "PAINT_WEIGHT, SELECT_WEIGHT, MESH_OBJECT"]"""
    def select_menu(self, objects:set[bpy.types.Object]) -> None:
        """If more than one valid object is detected, prompt user to select one"""
        title = f"Select which {str(objects[0].type).lower()} to use:"
        content = {
            ob.name: {
                "type": "button", 
                "op": "ct.set_paint_mode", 
                "prop": {
                    "pattern": ob.name, 
                    "extend": self.extend,
                    "unhide": self.unhide,
                    "set_mode": self.set_mode}, 
                "icon": f"{objects[0].type}_DATA"
                } for ob in objects}
        interactive_popup(title=title, content=content, icon='VPAINT_HLT')
    
    def unhide_object(self, ob:bpy.types.Object, popup:bool = True) -> None:
        """Returns True if success, False if failed"""
        if ob.visible_get():
            return True
        # Unhide active object
        ob.hide_set(False)
        ob.hide_viewport = False
        if is_collection_excluded(ob, unhide=True):
            return self.report({'ERROR'}, f"Unable to unhide '{ob.name}'") if popup else False
            #return popup_window(text=f"Unable to unhide '{ob.name}'") if popup else False
        #~~ # Ensure active is selected after unhide
        #~~ ob.select_set(True)
    
    def is_hidden(self, ob:bpy.types.Object, unhide:bool=True, popup:bool=True) -> bool:
        """Check if object is hidden and raise errors if needed\n
        Returns: True if object is hidden or unhide failed
        Returns: False if object is not hidden or unhide succeeded
        """
        if unhide:
            self.unhide_object(ob, popup)
            if not ob.visible_get():
                return True
        
        if ob.visible_get():
            return False
        else:
            self.report({'ERROR'}, f"Unable to enter '{self.set_mode}' mode while '{ob.name}' is hidden") if popup else True
            #popup_window(text=f"Unable to enter '{self.set_mode}' mode while '{ob.name}' is hidden") if popup else True
            return True

class CT_OT_SetPaintMode(bpy.types.Operator):
    bl_idname = "ct.set_paint_mode"
    bl_label = "Set Paint Mode"
    bl_description = "Quickly set active/selected object and changes mode"
    bl_options = {"REGISTER", "UNDO"}

    pattern: bpy.props.StringProperty(default="*", options={'HIDDEN'}) # type: ignore
    extend: bpy.props.BoolProperty(default=False, options={'HIDDEN'}) # type: ignore
    unhide: bpy.props.BoolProperty(default=False, options={'HIDDEN'}) # type: ignore
    set_mode: bpy.props.EnumProperty(name='Object Mode', options={'HIDDEN'}, items=[
        ('OBJECT', 'OBJECT', '', 0, 0), 
        ('EDIT', 'EDIT', '', 0, 1), 
        ('POSE', 'POSE', '', 0, 2), 
        ('PAINT_WEIGHT', 'PAINT_WEIGHT', '', 0, 3),
        ('SELECT_WEIGHT', 'SELECT_WEIGHT', '', 0, 4),
        ('MESH_OBJECT', 'MESH_OBJECT', '', 0, 5)
        ]) # type: ignore

    def execute(self, context):
        obj = bpy.context.object
        target = bpy.data.objects[self.pattern]
        
        # Unhide target
        if self.unhide and not target.visible_get():
            target.hide_set(False)
            target.hide_viewport = False
        
        # Check if object is hidden
        if is_collection_excluded(target, self.unhide): 
            popup_window(text=f"{target.type} '{target.name}' is in a hidden collection")
            return {"CANCELLED"}
        if not target.visible_get():
            popup_window(text=f"Object '{target.name}' is hidden")
            return {"CANCELLED"}
        
        arm_obj = target if target.type == 'ARMATURE' else obj
        mesh_obj = target if target.type == 'MESH' else obj
        
        ArmatureMode.set_mode_auto(self.set_mode, arm_obj, mesh_obj, self.extend)
        return {"FINISHED"}

# TODO ----
class CT_OT_RenameBonesRecursive(bpy.types.Operator):
    bl_idname = "ct.rename_bones_recursive"
    bl_label = "Rename Bones Recursive"
    bl_description = "Rename children of active bone recursively with a base name"
    bl_options = {"REGISTER", "UNDO"}
    
    root_bone: bpy.props.StringProperty(default="", options={'HIDDEN'}) # type: ignore
    new_name: bpy.props.StringProperty(default="Bone", options={'HIDDEN'}) # type: ignore
    root_prefix: bpy.props.StringProperty(default="Root", options={'HIDDEN'}) # type: ignore
    suffix: bpy.props.StringProperty(default="", options={'HIDDEN'}) # type: ignore
    
    def execute(self, context):
        armature:bpy.types.Armature = bpy.context.object.data if bpy.context.object.type == 'ARMATURE' else None
        
        # Check if armature pointer is valid
        
        assert armature, "No armature found"
        assert armature.bones, "No bones found"
        
        root_bone = None
        match bpy.context.mode:
            case 'EDIT_ARMATURE':
                root_bone = bpy.context.active_bone
            case 'POSE':
                root_bone = bpy.context.active_bone
            case 'OBJECT':
                root_bone = armature.bones.get(self.root_bone, armature.bones[0])
            case _:
                assert False, "Invalid mode"
        
        assert root_bone, "No bone selected"

        #root_bone.name = '_temp_name'
        for bone in [root_bone, *root_bone.children_recursive]:
            bone.name = '_temp_name'
        self.rename_children_recursive(root_bone, base_name=self.new_name, root_prefix=self.root_prefix, suffix=self.suffix)
        return {"FINISHED"}
    
    def rename_children_recursive(self, bone, base_name="Bone", child_index="", recursive_index=0, start_branch_at_zero=True, root_prefix="Root", suffix=""):
        def create_name(_base_name, _child_index, _recursive_index, _suffix):
            """
            Root
                bone1_000 | ("bone", "1", "0")
                    bone1.1_001 | ("bone", "1.1", "1")
                    bone1.1_002 | ("bone", "1.1", "2")
                    
                    bone1.2_001 | ("bone", "1.2", "1")
                    bone1.2_002 | ("bone", "1.2", "2")
            
                bone2_000 | ("bone", "2", "0")
                    bone2_001 | ("bone", "2", "1") 
                    bone2_002 | ("bone", "2", "2")
            """
            #return f"{base_name}{child_index}_{recursive_index:03d}"
            return f"{_base_name}{'_' if _child_index else ''}{_child_index}.{_recursive_index:03d}{_suffix}"
        if not recursive_index:
            bone.name = base_name + "_" + root_prefix if root_prefix else f"{base_name}.{recursive_index:03d}{suffix}"
            if not root_prefix: recursive_index += 1
        
        branch = len(bone.children) > 1
        new_child_index = child_index
        # Recursive rename for children
        for i, child in enumerate(bone.children):
            if branch:
                if not start_branch_at_zero: i += 1
                new_child_index = f"{child_index}_{i}" if child_index else f"{i}"
            
            child.name = create_name(base_name, new_child_index, recursive_index, suffix)
            self.rename_children_recursive(child, base_name=base_name, child_index=new_child_index, recursive_index=recursive_index+1, suffix=suffix)


edit_operators = [
    CT_OT_SetPaintMode,
    CT_OT_Init_SetPaintMode,
    CT_OT_RenameBonesRecursive,
]

def register():
    for op in edit_operators:
        bpy.utils.register_class(op)

def unregister():
    for op in edit_operators:
        bpy.utils.unregister_class(op)