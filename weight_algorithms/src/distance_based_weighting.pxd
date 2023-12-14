# distutils: language = c++
# cython: language_level=3
# cython: cdivision = True
# cython: boundscheck = False

import bpy
import bmesh

from libcpp.vector cimport vector

ctypedef struct Vec3:
    float x, y, z

ctypedef struct Weight:
    int group
    float value

ctypedef struct Vert:
    int index
    Vec3 co
    vector[Weight] weights

ctypedef struct Bone:
    int group_index
    Vec3 pos


cdef class VertsHolder:
    cdef int size
    cdef vector[Vert] vertarray

cdef class BonesHolder:
    cdef int size
    cdef vector[Bone] bonearray

cdef inline float vec_len(Vec3 vec): ...
cdef float length(Vec3 a, Vec3 b): ...

cdef void calculate_weights(vector[Vert] &verts, vector[Bone] &bones, float power, float threshold): ...
cdef void set_weights(bm:bmesh.types.BMesh, verts:vector[Vert]): ...

def create_verts_array(verts:list[bpy.types.MeshVertex]) -> VertsHolder: ...
def create_bones_array(posebones:list[bpy.types.Bone]) -> BonesHolder: ...

cpdef distance_bones_verts(
        bm:bmesh.types.BMesh,
        vertsholder:VertsHolder,
        bonesholder:BonesHolder,
        power:float=2.0, 
        threshold:float=0.01): ...