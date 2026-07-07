#pragma once

#include <Python.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
// The import from Python will load the .so consisting of this file
// in this extension, so that the TORCH_LIBRARY static initializers are run.
// We use extern "C" to ensure C linkage for the Python module init function.
// Setting module size to -1 (instead of 0) ensures module state is kept in
// global variables and the module is properly initialized each time it's imported.
#define REGISTER_EXTENSION(NAME)                                               \
  extern "C" {                                                                 \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, -1, nullptr}; \
    return PyModule_Create(&module);                                           \
  }                                                                            \
  }
