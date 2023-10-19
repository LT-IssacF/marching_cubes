#ifndef TYPES_CUH
#define TYPES_CUH

// unsigned char _tmp_[4] = {0x01, 0x00, 0x80, 0x7f};
// float NaN = *(float*)_tmp_;

struct Point {
    float x, y, z, value;
    __device__ Point() : x(1.17549e-38f), y(1.17549e-38f), z(1.17549e-38f), value(1.17549e-38f) {}
    __device__ Point(float a, float b, float c) : x(a), y(b), z(c) {}
    __device__ Point(float a, float b, float c, float _value) : x(a), y(b), z(c), value(_value) {}
    __device__ Point(const Point &rhs) : x(rhs.x), y(rhs.y), z(rhs.z), value(rhs.value) {}
    __device__ bool operator<(const Point &rhs) const {
        if (x != rhs.x) {
            return x < rhs.x;
        } else if (y != rhs.y) {
            return y < rhs.y;
        } return z < rhs.z;
    }
    __device__ Point operator+(const Point &rhs) const {
        return Point(x + rhs.x, y + rhs.y, z + rhs.z);
    }
    __device__ Point operator-(const Point &rhs) const {
        return Point(x - rhs.x, y - rhs.y, z - rhs.z);
    }
    __device__ Point operator*(const float &f) const {
        return Point(x * f, y * f, z * f);
    }
};

struct _Point {
    float x, y, z;
    __device__ _Point() : x(1.17549e-38f), y(1.17549e-38f), z(1.17549e-38f) {}
    __device__ _Point(float a, float b, float c) : x(a), y(b), z(c) {}
    __device__ _Point(const _Point &rhs) : x(rhs.x), y(rhs.y), z(rhs.z) {}
    __device__ _Point(const Point &rhs) : x(rhs.x), y(rhs.y), z(rhs.z) {}
};

struct Voxel { // just a tiny cube with 8 vertices and 12 edges
    Point vertices[8];
    _Point triangle[15];
    _Point intersections[12];
    int index = -1;
};

#endif