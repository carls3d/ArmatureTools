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
