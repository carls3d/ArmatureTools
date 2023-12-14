# distutils: language = c++
# cython: language_level=3
# cython: cdivision = True
# cython: boundscheck = False

import bpy
import bmesh

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libc.math cimport sqrt, powf
from utils.weights_types cimport Vec3, Vert, Bone, Weight


cdef class VertsHolder:
    cdef int size
    cdef vector[Vert] vertarray

    def __cinit__(self, int size):
        self.size = size
        self.vertarray.resize(size)
    
    def __dealloc__(self):
        self.vertarray.clear()
        
cdef class BonesHolder:
    cdef int size
    cdef vector[Bone] bonearray

    def __cinit__(self, int size):
        self.size = size
        self.bonearray.resize(size)

    def __dealloc__(self):
        self.bonearray.clear()


cdef inline float vec_len(Vec3 vec):
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z)

cdef float length(Vec3 a, Vec3 b):
    return vec_len(Vec3(a.x - b.x, a.y - b.y, a.z - b.z))


cdef void calculate_weights(vector[Vert] &verts, vector[Bone] &bones, float power, float threshold):
    cdef int vert_len = verts.size()
    cdef int bone_len = bones.size()
    if vert_len < 1 or bone_len < 1: return

    cdef int i, j
    cdef float dist, total_weight
    cdef Vert* vert
    cdef Weight* weight
    for i in range(vert_len):
        vert = &verts[i]
        vert.weights.resize(bone_len)
        
        total_weight = 0
        for j in range(bone_len):
            dist = length(vert.co, bones[j].pos)
            dist = 1 / powf(dist, power)
            total_weight += dist
            weight = &vert.weights[j]
            weight.value = dist
            weight.group = bones[j].group_index

        # Normalize weights
        for j in range(bone_len):
            weight = &vert.weights[j]
            weight.value /= total_weight
        
        # Remove weights below threshold
        total_weight = 1.0
        for j in range(bone_len):
            weight = &vert.weights[j]
            if weight.value < threshold:
                total_weight -= weight.value
                weight.value = 0

        # Re-Normalize weights
        for j in range(bone_len):
            weight = &vert.weights[j]
            weight.value /= total_weight

cdef void set_weights(bm:bmesh.types.BMesh, verts:vector[Vert]):
    dvert_lay = bm.verts.layers.deform.active
    cdef Vert* vert
    cdef int i
    cdef int n = verts.size()
    for i in range(n):
        vert = &verts[i]
        dvert = bm.verts[vert.index][dvert_lay]
        dvert.clear()
        for weight in vert.weights:
            if weight.value == 0: continue
            dvert[weight.group] = weight.value
    bm.to_mesh(bpy.context.object.data)

def create_verts_array(verts:list[bpy.types.MeshVertex]):
    matr = bpy.context.object.matrix_world
    coords = [matr @ v.co for v in verts]
    cdef int verts_size = len(verts)
    holder = VertsHolder(verts_size)
    cdef int i
    for i in range(verts_size):
        # coords = bpy.context.object.matrix_world @ verts[i].co
        holder.vertarray[i].index = verts[i].index
        holder.vertarray[i].co.x = coords[i].x
        holder.vertarray[i].co.y = coords[i].y
        holder.vertarray[i].co.z = coords[i].z
    return holder

def create_bones_array(posebones:list[bpy.types.Bone]):
    vgroups = bpy.context.object.vertex_groups
    posebones = [b for b in posebones if b.name in vgroups]
    matr = bpy.context.pose_object.matrix_world
    coords = [matr @ pb.center for pb in posebones]
    cdef int bones_size = len(posebones)
    holder = BonesHolder(bones_size)
    cdef int i
    for i in range(bones_size):
        # coords = matr @ posebones[i].center
        holder.bonearray[i].group_index = vgroups[posebones[i].name].index
        holder.bonearray[i].pos.x = coords[i].x
        holder.bonearray[i].pos.y = coords[i].y
        holder.bonearray[i].pos.z = coords[i].z
    return holder

cpdef distance_bones_verts(
        bm:bmesh.types.BMesh,
        vertsholder:VertsHolder,
        bonesholder:BonesHolder,
        power:float=2.0, 
        threshold:float=0.01):
    # posebones -> bones to operate on
    # bm_verts -> bm_verts to operate on
    
    # cdef vector[Vert] verts = vertsholder.vertarray
    # cdef vector[Bone] bones = bonesholder.bonearray

    calculate_weights(vertsholder.vertarray, bonesholder.bonearray, power, threshold)
    set_weights(bm, vertsholder.vertarray)