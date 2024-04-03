import bpy, bmesh
from mathutils import Vector

import time
class Timer:
    @staticmethod
    def deco(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            print(f"Time elapsed: {time.time() - start}")
        return wrapper

@Timer.deco
def remove_unused_vertex_groups():
    obj = bpy.context.object
    if obj.type != 'MESH':
        return
    
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    dvert_lay = bm.verts.layers.deform.active
    
    vgroups = obj.vertex_groups
    vgroups_indices = {vgroup.index for vgroup in vgroups}
    
    for vert in bm.verts:
        dvert = vert[dvert_lay]
        for grp_index in dvert.keys():
            vgroups_indices.discard(grp_index)
    # Reverse order before deleting
    for i in list(vgroups_indices)[::-1]:
        vgroup = vgroups[i]
        vgroups.remove(vgroup)
        
    bm.to_mesh(obj.data)
    bm.free()
    
    
    
def reverse_bones():
    C = bpy.context
    for editbone in C.selected_bones:
        editbone.use_connect = False
        for child in editbone.children:
            child.use_connect = False
    
    editbones = C.selected_bones
    coords = [[bone.head.copy(), bone.tail.copy()] for bone in editbones]
    parents = [bone.parent for bone in editbones]
    children = [bone.children for bone in editbones]
    
    all_editbones = C.edit_object.data.edit_bones[:]
    
    print({all_editbones.index(b) for b in editbones[0].children_recursive})
    return
    for bone in editbones:
        bone.parent = None
        for child in bone.children:
            child.parent = None
    for i, editbone in enumerate(C.selected_bones[::-1]):
        print("---")
        editbone.tail = coords[i][0]
        editbone.head = coords[i][1]
        # editbone.parent = parents[i]
        # for child in children[i]:
        #     child.parent = editbone


# remove_unused_vertex_groups()
# reverse_bones()


bone_i = 0
class Bone: ...
class Bone:
    def __init__(self, name):
        global bone_i
        self.name = name
        self.index = bone_i
        bone_i += 1
        
        self._parent = None
        self._children = []
    
    def __repr__(self):
        # return f"<Bone: {self.name}>"
        # return f"<'{self.name}'>"
        # return self.name
        return f"{self.parent.name}->{self.name}"
    
    @property
    def parent(self) -> Bone:
        return self._parent
    
    @parent.setter
    def parent(self, parent:Bone) -> None:
        if parent and parent not in self.children_recursive:
            self._parent = parent
            if self not in parent.children:
                parent.children.append(self)
        elif not parent:
            if self._parent:
                self._parent.children.remove(self)
            
    @property
    def children(self) -> list[Bone]:
        return self._children
    
    @property
    def children_recursive(self) -> list[Bone]:
        children = []
        for child in self.children:
            children.append(child)
            children.extend(child.children_recursive)
        return children
    
    @property
    def parent_recursive(self) -> list[Bone]:
        parents = []
        parent = self.parent
        while parent:
            parents.insert(0, parent)
            parent = parent.parent
        return parents


from random import shuffle
def create_bones(names:list[str], randomize:bool=False) -> list[Bone]:
    bones = [Bone(name) for name in names]
    if randomize:
        shuffle(bones)
    for i, bone in enumerate(bones):
        if i > 0:
            bone.parent = bones[i-1]
    return bones 

bnames = [f"bone{i}" for i in range(0, 4, 1)]
bones = create_bones(bnames, 0)
bnames = [f"bone{i}" for i in range(11, 14, 1)]
bones2 = create_bones(bnames, 0)
bnames = [f"bone{i}" for i in range(17, 20, 1)]
bones3 = create_bones(bnames, 0)
bones2[0].parent = bones[-1]
bones3[0].parent = bones[-1]
# print(bones[-1].children)
# for b in bones[0].children_recursive:
#     print(b)

def recursive_indices(bones_list:list[Bone]) -> set[int]:
    recursive_bones = lambda j: (*bones_list[j].parent_recursive, bones_list[j], *bones_list[j].children_recursive)
    # recursive_bones = lambda j: (bones_list[j], *bones_list[j].children_recursive)
    bones_dict = {}
    for j in range(len(bones_list)):
        # temp = {bones_list.index(b): b for b in recursive_bones(j) if b in bones_list}
        # temp = {bones.index(b): b for b in recursive_bones(j) if b in bones_list}
        temp = {i: b for i, b in enumerate(recursive_bones(j)) if b in bones_list}
        print(temp)
        # bones_dict.setdefault(j, temp)
        bones_dict.update(temp)
    sorted_bones = sorted(bones_dict.items(), key=lambda x: x[0])
    return sorted_bones


cropped_bones = bones[0].children_recursive
# print(cropped_bones)
shuffle(cropped_bones)
# print(cropped_bones)
a = recursive_indices(cropped_bones)
for x in a:
    print(x[0], x[1])
# b = [i[1] for i in a]
# print(b)



