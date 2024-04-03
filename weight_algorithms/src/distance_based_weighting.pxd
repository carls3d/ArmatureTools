# distutils: language = c++
# cython: language_level=3
# cython: cdivision = True
# cython: boundscheck = False

from libcpp.vector cimport vector
from utils.weights_types cimport Vec3, Vert, Bone, Weight
import bmesh
import bpy

cdef class VertsHolder:
    cdef public int size
    cdef public vector[Vert] vertarray

cdef class BonesHolder:
    cdef public int size
    cdef public vector[Bone] bonearray

cdef inline float vec_len(Vec3 vec)
cdef float length(Vec3 a, Vec3 b)
cdef void calculate_weights(vector[Vert] &verts, vector[Bone] &bones, float power, float threshold)
cdef void set_weights(bm:bmesh.types.BMesh, verts:vector[Vert], replace:bool=True)

cpdef create_verts_array(verts:list[bpy.types.MeshVertex])
cpdef create_bones_array(posebones:list[bpy.types.Bone])
cpdef distance_bones_verts(bm:bmesh.types.BMesh, vertsholder:VertsHolder, bonesholder:BonesHolder, power:float=2.0, threshold:float=0.01)