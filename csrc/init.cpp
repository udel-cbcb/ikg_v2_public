#include <pybind11/pybind11.h>
#include "build_index.h"

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(ikg_native, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
    m.def("to_indexed_triples",&to_indexed_triples,"Convert triples with string names to indexed values");
}