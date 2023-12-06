# distutils: language = c++
# cython: language_level=3
# cython: cdivision = True
# cython: boundscheck = False

import bpy
import bmesh
# import numpy as np
# from mathutils import Vector

from libcpp.vector cimport vector as cpplist
from libcpp.string cimport string as cppstring
from libcpp.algorithm cimport sort, clamp, for_each
from libcpp.numeric cimport accumulate
from libcpp.unordered_map cimport unordered_map as cppdict
from libcpp.unordered_set cimport unordered_set as cppset
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, powf, fmaxf, fminf


ctypedef struct Vec3:
    float x, y, z

ctypedef struct VGroup:
    int index
    cppstring name

ctypedef struct Weight:
    int group
    float value

ctypedef struct Vert:
    int index
    Vec3 co
    cpplist[Weight] weights

ctypedef struct Bone:
    cppstring name
    int group_index
    Vec3 head, tail, center


cdef inline float vec_len(Vec3 vec):
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)

cdef float length(Vec3 a, Vec3 b):
    return vec_len(Vec3(a.x - b.x, a.y - b.y, a.z - b.z))


cdef inline Vec3 vec_center(Vec3 a, Vec3 b):
    return Vec3((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)

    

cdef verts_from_bpy(py_verts:list[bpy.types.MeshVertex]):
    cdef cpplist[Vert] verts
    cdef int i
    cdef int n = len(py_verts)
    for i in range(n):
        verts.push_back(Vert(py_verts[i].index, Vec3(py_verts[i].co.x, py_verts[i].co.y, py_verts[i].co.z)))
    return verts

cdef bones_from_bpy(bpy_bones:list[bpy.types.Bone], obj:bpy.types.Object):
    vgroups = obj.vertex_groups
    cdef cpplist[Bone] bones
    for b in bpy_bones:
        if b.name not in vgroups: continue
        head = Vec3(b.head_local.x, b.head_local.y, b.head_local.z)
        tail = Vec3(b.tail_local.x, b.tail_local.y, b.tail_local.z)
        center = vec_center(head, tail)
        bones.push_back(Bone(
            b.name.encode(), 
            vgroups[b.name].index,
            head, 
            tail,
            center))
    return bones


cdef void calculate_weights(cpplist[Vert] &verts, cpplist[Bone] &bones, float power, float threshold):
    cdef int vert_len = verts.size()
    cdef int bone_len = bones.size()
    if vert_len < 1 or bone_len < 2: return

    cdef int i, j
    cdef float dist, total_weight
    cdef Vert* vert
    cdef Weight* weight
    for i in range(vert_len):
        vert = &verts[i]
        vert.weights.resize(bone_len)
        
        total_weight = 0
        for j in range(bone_len):
            dist = length(vert.co, bones[j].center)
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

cdef void set_weights(bm:bmesh.types.BMesh, verts:cpplist[Vert]):
    bm.verts.ensure_lookup_table()
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

cpdef distance_bones_verts(
        obj:bpy.types.Object, 
        bm:bmesh.types.BMesh,
        vertices:list[bpy.types.MeshVertex], 
        bonelist:list[bpy.types.Bone], 
        power:float=2.0, 
        threshold:float=0.01):
    # if obj.type != 'MESH': return
    # Bonelist -> bones to operate on
    # Vertices -> vertices to operate on

    cdef cpplist[Vert] verts = verts_from_bpy(vertices)
    cdef cpplist[Bone] bones = bones_from_bpy(bonelist, obj)

    calculate_weights(verts, bones, power, threshold)
    set_weights(bm, verts)

    

    