import bpy

#TODO -----------------  Weight Paint -----------------
#TODO - Dissolve bone / delete -> transfer to parent
#todo     - transfer to parent if there is one, else transfer to child
#todo     - Check if more than one child
#todo         1. Transfer to child with longest chain
#todo         2. Spread weights to all children

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


def planning():
    ...
    # Get active object
    # Assert
    # In edit mode
    # Assert edit mode or pose mode
    # Get armature of active object

    # Operator1: Dissolve -> transfer weights
    # Get selected bones

    #   1. Get all objects with armature modifier
    #   2. Get all objects with vertex groups of selected bones
    # Store info in dict
    # {bone1: [obj1, obj2, obj3], bone2: [obj1, obj2, obj3]]}

    # obj1: {vertex_group:{vertindex: weight, ...}, ...}
    # vertex_group: {obj1: {vertindex: weight, ...}, ...}

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

# def set_mode_paint(meshobj:bpy.types.Object):
#     assert len(bpy.context.selected_objects) > 1, "Expected 2 or more selected objects"
#     assert meshobj.parent, "Mesh object has no parent armature"
#     assert meshobj.parent.type != 'ARMATURE', "Mesh object parent is not an armature"
#     assert meshobj.modifiers, "Mesh object has no modifiers"
#     armature_modifiers = {m.object: m for m in meshobj.modifiers if m.type == 'ARMATURE'}
#     assert armature_modifiers, "Mesh object has no armature modifiers"
    
#     if meshobj.parent not in armature_modifiers:
#         [*armature_modifiers.values()][0].object = meshobj.parent
    
#     bpy.ops.object.mode_set(mode='OBJECT')
#     meshobj.parent.select_set(True)
#     bpy.context.view_layer.objects.active = meshobj
#     bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

# def set_mode_edit():
#     mode = bpy.context.mode
#     if mode == 'POSE':
#         bpy.ops.object.mode_set(mode='EDIT')
#     if mode == 'PAINT_WEIGHT':
#         bpy.ops.object.mode_set(mode='OBJECT')
#         bpy.context.view_layer.objects.active = bpy.context.object.parent
#         bpy.ops.object.mode_set(mode='EDIT')
    
# def set_mode_pose():
#     mode = bpy.context.mode
#     if mode == 'EDIT_ARMATURE':
#         bpy.ops.object.mode_set(mode='POSE')
#     if mode == 'PAINT_WEIGHT':
#         bpy.ops.object.mode_set(mode='OBJECT')
#         bpy.context.view_layer.objects.active = bpy.context.object.parent
#         bpy.ops.object.mode_set(mode='POSE')

#SECTION ------------ Common operations ------------
class EditFuncs:
    
    @staticmethod
    def init_bones(min_bones:int):
        # Undo steps not stored properly in edit mode
        if bpy.context.mode == 'EDIT_ARMATURE':
            bpy.ops.object.mode_set(mode='POSE')
        mode = bpy.context.mode
        
        if mode == 'PAINT_WEIGHT':
            armatures = [bpy.context.object.parent]
            active_bone = bpy.context.active_pose_bone.id_data.data.bones[bpy.context.active_pose_bone.name]
        elif mode == 'POSE':
            armatures = bpy.context.objects_in_mode
            active_bone = bpy.context.active_bone
        else:
            return popup_window(title="Error", text="Must be in pose or weight paint mode", icon='ERROR')
        
        if not armatures:
            return popup_window(title="Error", text="No valid armatures found", icon='ERROR')
        
        sel_bones = [pose_bone.id_data.data.bones[pose_bone.name] for pose_bone in bpy.context.selected_pose_bones]
        if len(sel_bones) < min_bones: 
            return popup_window(title="Error", text=f"Select {min_bones} bones or more", icon='ERROR')
        
        # Report.debug(dedent(f"""\
        #     Armatures: {armatures}
        #     Bones: {sel_bones}
        #     Active bone: {active_bone}
        #     Mode: {bpy.context.mode}
        #     """
        #     ).strip("\n"))
        return armatures, sel_bones, active_bone
    
    @staticmethod
    def set_active_bone(bone:object) -> None:
        """Set active bone in edit mode"""
        bone.select = True
        bone.id_data.bones.active = bone
    
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
    
    @staticmethod
    def cleanup_vertex_groups(bones:list, objects:list):
        for obj in objects:
            for bone in bones:
                if bone.name not in obj.vertex_groups: continue
                obj.vertex_groups.remove(obj.vertex_groups[bone.name])
    
    @staticmethod
    def objects_from_bones(armatures:list, bones:list) -> list:
        """Returns a list of objects connected to selected bones"""
        assert len(bones), "No bones"
        objects = set()
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH': continue
            if not obj.modifiers: continue
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE' and mod.object in armatures:
                    objects.add(obj)
            # armature_modifiers = {mod for mod in obj.modifiers if mod.type == 'ARMATURE' and mod.object in armatures}
            # objects += armature_modifiers
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
                group_weights = [float]*len(vertex.groups)
                group_indices = [int]*len(vertex.groups)
                vertex.groups.foreach_get("group", group_indices)
                vertex.groups.foreach_get("weight", group_weights)
                
                for i, grp_index in enumerate(group_indices):
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
                return bpy.context.selected_bones
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
            in_values = find_key(valid_bones, bone.parent)
            if in_values:
                valid_bones.setdefault(in_values, []).append(bone)
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
        

#SECTION ------------ Functions ------------
def editbones_selected_to_active() -> None:
    """ Transfer weights from selected bones to active bone \n
        Delete selected bones =! active bone"""
    mode = bpy.context.mode
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
    EditFuncs.cleanup_bones(sel_bones)
    EditFuncs.cleanup_vertex_groups(sel_bones, objects)
    
    EditFuncs.set_active_bone(active_bone)
    
    if mode == 'EDIT_ARMATURE': 
        bpy.ops.object.mode_set(mode='EDIT')
    return
    
# def editbones_selected_to_centre() -> None:
#     """ Merge selected bones to center \n"""
#     # Object mode - Use all root bones in armature
#     # Edit mode - Selected bones -> linked -> root bones
#     # Transfer to first child bone
    
#     mode = bpy.context.mode
#     init_vars = EditFuncs.init_bones(2)
#     if not init_vars: return
#     armatures, sel_bones, active_bone = init_vars
    
#     objects = EditFuncs.objects_from_bones(armatures, sel_bones)
#     obj_weights = EditFuncs.weights_from_objects(objects)
    
#     def set_connected():
#         assert bpy.context.mode == 'EDIT_ARMATURE', "Must be in edit armature mode"
#         bones = bpy.context.active_bone.children
#         for bone in bones:
#             bone.use_connect = False
    
#     def average_coords():
#         """ Returns average of list of vectors """
#         head_coords = [bone.head_local for bone in sel_bones]
#         tail_coords = [bone.tail_local for bone in sel_bones]
#         active_bone = bpy.context.active_bone
#         active_bone.head = Vector(np.average(head_coords, 0))
#         active_bone.tail = Vector(np.average(tail_coords, 0))
#         active_bone.name = active_bone.name + '_root'
    
#     if bpy.context.mode == 'PAINT_WEIGHT':
#         obj = bpy.context.object
#         bpy.ops.object.mode_set(mode='OBJECT')
#         bpy.context.view_layer.objects.active = bpy.context.object.parent
#         bpy.ops.object.mode_set(mode='EDIT')
#         set_connected()
#         average_coords()
#         bpy.ops.object.mode_set(mode='OBJECT')
#         bpy.context.view_layer.objects.active = obj
#         bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
#     else:
#         assert bpy.context.object.type == 'ARMATURE', "Must be in armature edit mode"
#         bpy.ops.object.mode_set(mode='EDIT')
#         set_connected()
#         average_coords()
#         bpy.ops.object.mode_set(mode='POSE')
    
#     EditFuncs.transfer_weights(obj_weights, active_bone, sel_bones)
    
#     # Cleanup bones and vertex groups
#     EditFuncs.cleanup_bones(sel_bones)
#     EditFuncs.cleanup_vertex_groups(sel_bones, objects)
    
#     if mode == 'EDIT_ARMATURE':
#         bpy.ops.object.mode_set(mode='EDIT')

def editbones_selected_remove() -> None:
    """
    Transfer weights to parent bone \n
    If no parent, transfer to child bone \n
    """
    
    #TODO - No parent -> error
    mode = bpy.context.mode
    if mode == 'EDIT_ARMATURE':
        bpy.ops.object.mode_set(mode='POSE')
    
    init_vars = EditFuncs.init_bones(1)
    if not init_vars: return
    armatures, sel_bones, active_bone = init_vars
    
    objects = EditFuncs.objects_from_bones(armatures, sel_bones)
    obj_weights = EditFuncs.weights_from_objects(objects)
    
    sel_bones = [bone for bone in sel_bones if bone.parent]
    if not sel_bones: 
        return popup_window(title="Error", text="No valid bones", icon='ERROR')
    
    bones = EditFuncs.organize_bones(sel_bones)
    for target_bone, children in bones.items():
        EditFuncs.transfer_weights(obj_weights, target_bone, children)
    
    # Clean up values
    cleanup_bones = [item for sublist in bones.values() for item in sublist]
    EditFuncs.cleanup_bones(cleanup_bones)
    EditFuncs.cleanup_vertex_groups(cleanup_bones, objects)
    
    # Select first key
    bone = [*bones.keys()][0]
    EditFuncs.set_active_bone(bone)
    
    if mode == 'EDIT_ARMATURE':
        bpy.ops.object.mode_set(mode='EDIT')
    return

# def editbones_selected_dissolve() -> None:
#     # Same as remove, but dissolve bone
#     #* Cannot run if bone has more than 1 child
#     #* Cannot run if parent has more than 1 child
#     ...

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

# def editbones_chain_merge() -> None:
#     # Starting at set index of chains, merge chains by distance
#     ...

# def editbones_chain_reverse() -> None:
#     # Reverse selected chains
#     ...

def editbones_chain_trim(start:float=0.0, end:float=1.0) -> None:
    """Trim selected chains from start and end"""
    # Clamp start and end
    min_offset = 0.1 # A minimum offset to prevent errors from negative values and all bones from being removed
    end = max(start + min_offset, end)
    end = min(end, 1.0)

    start = min(start, end - min_offset)
    start = max(start, 0.0)
    
    mode = bpy.context.mode
    if mode == 'EDIT_ARMATURE':
        bpy.ops.object.mode_set(mode='POSE') #* Undo doesn't work properly in edit mode
    
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
    if mode == 'EDIT_ARMATURE':
        bpy.ops.object.mode_set(mode='EDIT')
    return

# def editbones_auto_rename() -> None:
#     # Edit mode only runs on selected (bones? / bonechains?)
#     ...

# def editbones_auto_optimize() -> None:
#     # Automatic version of chain_merge
#     ...

def editbones_set_mode(mode:str):
    """["OBJECT", "EDIT", "POSE", "PAINT_WEIGHT"]"""
    obj = bpy.context.object
    #FIXME - Force visibility of armature -> error if not visible after
    error_dict = {
        obj.type not in ['MESH', 'ARMATURE']: "Invalid object type",
        obj.type == 'MESH' and not obj.parent: "Mesh object has no parent armature",
        obj.type == 'MESH' and obj.parent and obj.parent.type != 'ARMATURE': "Mesh object parent is not an armature"
    }
    for error in error_dict:
        if error:
            return popup_window(title="Error", text=error_dict[error], icon='ERROR')
    
    if mode == 'OBJECT':
        if obj.type == 'MESH':
            obj = obj.parent
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_pattern(pattern=obj.name, extend=False)
        bpy.context.view_layer.objects.active = obj
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        return
    
    if mode in ['EDIT', 'POSE']:
        if obj.type == 'MESH':
            if bpy.context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            armature = obj.parent
            armature.select_set(True)
            bpy.context.view_layer.objects.active = armature
            
        bpy.ops.object.mode_set(mode=mode)
        selected = bpy.context.selected_objects
        in_mode = bpy.context.objects_in_mode
        for ob in selected:
            # Skip objects that are in the mode
            if ob in in_mode:
                continue
            # De-select objects that are not in the mode
            ob.select_set(False)
        return
    
    
    if mode == 'PAINT_WEIGHT':
        if obj.type == 'MESH':
            if bpy.context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            armature = obj.parent
            armature.select_set(True)
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    
        if obj.type == 'ARMATURE':
            objects = bpy.data.objects
            
            # Valid meshes that are connected to the armature through a modifier or parent
            meshes = {ob.name for ob in objects if ob in obj.children and ob.type == 'MESH'}
            meshes |= {ob.name for ob in objects if ob.type == 'MESH' and ob.modifiers and obj in [mod.object for mod in ob.modifiers if mod.type == 'ARMATURE']}
            
            # If valid armature is selected, use it
            selected = bpy.context.selected_objects
            valid_selected = [sel.name for sel in selected if sel.name in meshes]
            if len(valid_selected) == 1:
                bpy.context.view_layer.objects.active = objects[valid_selected[0]]
                bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
                return
            
            #TODO - make this a setting
            # elif len(valid_selected) > 1:
            #     meshes = valid_selected
            
            if len(meshes) < 1: 
                return popup_window(title="Error", text=f"No valid meshes found for {obj.name}", icon='ERROR')
            
            # If no valid armature is selected, prompt user to select one
            title = f"Select which mesh to use:"
            content = {
                ob: {
                    "type": "button", 
                    "op": "sna.set_paint_mode", 
                    "prop": {"pattern": ob, "extend": True}, 
                    "icon": "MESH_DATA"
                    } for ob in meshes}
            interactive_popup(title=title, content=content, icon='VPAINT_HLT')
        return
    return


#SECTION ------------ Operators ------------
class SetPaintMode(bpy.types.Operator):
    bl_idname = "sna.set_paint_mode"
    bl_label = "Set Paint Mode"
    bl_description = "Description"
    bl_options = {"REGISTER"}

    pattern: bpy.props.StringProperty(default="*")
    extend: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        bpy.ops.object.select_pattern(pattern=self.pattern, extend=self.extend)
        bpy.context.view_layer.objects.active = bpy.data.objects[self.pattern]
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        return {"FINISHED"}


def register():
    bpy.utils.register_class(SetPaintMode)

def unregister():
    bpy.utils.unregister_class(SetPaintMode)

#STUB ------------------------------------------------- For testing as script

# if __name__ == '__main__':
#     import timeit
#     starttime = timeit.default_timer()
#     def report_print(REPORT_OUT:dict, debug:bool=False) -> None:
#         """Report if injected into a class, else print"""
#         self        = locals()['self'] if 'self' in locals() else None
#         report_out  = self.report if self else print
        
#         if REPORT_OUT is None: return
#         for key, string_list in REPORT_OUT.items():
#             if key in ("INFO", "WARNING", "ERROR"):
#                 for string in string_list:
#                     report_out({key}, string)
            
#             elif key == "DEBUG" and debug == True: 
#                 if REPORT_OUT["DEBUG"]: 
#                     print("\n ------- Debug: -------")
#                     for debug in REPORT_OUT["DEBUG"]:
#                         print(debug)
#                     print(" ------------------------ \n")
            
#             if key not in ("INFO", "WARNING", "ERROR", "DEBUG"):
#                 print("!! Invalid report key: {key} !!")
            
#     # editbones_chain_trim()
#     editbones_selected_remove()
#     report_print(REPORT_OUT, False)
#     print(f"Time: {timeit.default_timer() - starttime}")
    
#     print("\n------------------- __main__ -------------------")
    