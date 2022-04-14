#include </usr/include/python3.7/Python.h>
#include </usr/include/python3.7/numpy/arrayobject.h>
#include </usr/include/python3.7/numpy/arrayscalars.h>
#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "modmoe.h" //Functions for Maximum of Erosions and Minimum of Dilations
#include "lodloe.h" //Functions for Linear combination of dilations and erosions
#include "morphpool.h" //Functions for Morphological poolig

static PyObject* morph_mod_5x5(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    dilation_simd_5x5(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_mod_7x7(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    dilation_simd_7x7(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_moe_5x5(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    erosion_simd_5x5(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_moe_7x7(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    erosion_simd_7x7(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_ave(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims;
    PyArrayObject *input1_array, *input2_array, *output_array;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input1_array, &input2_array, &output_array
    );
    if(!err) return NULL;

    image_dims = PyArray_DIMS(input1_array);

		ave(input1_array->data, input2_array->data, output_array->data, image_dims[0], image_dims[1]);

    Py_RETURN_NONE;
}

static PyObject* morph_lap_7x7(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    laplacian_simd_7x7(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_lap_5x5(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    laplacian_simd_5x5(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_lod_5x5(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *weights, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOOO",
        &input_array, &output_array, &se, &weights
    );
    if(!err) return NULL;

    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    lod_simd_5x5(input_array->data, output_array->data, se->data, weights->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_loe_5x5(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *weights, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOOO",
        &input_array, &output_array, &se, &weights
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    loe_simd_5x5(input_array->data, output_array->data, se->data, weights->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_linear_lap_5x5(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *weights, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOOO",
        &input_array, &output_array, &se, &weights
    );
    if(!err) return NULL;

    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    linear_laplacian_simd_5x5(input_array->data, output_array->data, se->data, weights->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* morph_linear_lap_7x7(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *weights, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOOO",
        &input_array, &output_array, &se, &weights
    );
    if(!err) return NULL;

    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    linear_laplacian_simd_7x7(input_array->data, output_array->data, se->data, weights->data, image_dims[0], image_dims[1], se_dims[2]);

    Py_RETURN_NONE;
}

static PyObject* m_pool_7x7(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    morph_pool_7x7(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1]);

    Py_RETURN_NONE;
}

static PyObject* m_pool_3x3(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    morph_pool_3x3(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1]);

    Py_RETURN_NONE;
}

static PyObject* m_pool_5x5(PyObject *self, PyObject *args)
{
    int err, type;
    //npy_intp ndim;
    npy_intp *image_dims, *se_dims;
    PyArrayObject *input_array, *output_array, *se;

    // parse arguments 画像3枚引数
    err = PyArg_ParseTuple(
        args, "OOO",
        &input_array, &output_array, &se
    );
    if(!err) return NULL;

    //
    image_dims = PyArray_DIMS(input_array);
    se_dims    = PyArray_DIMS(se);

    morph_pool_5x5(input_array->data, output_array->data, se->data, image_dims[0], image_dims[1]);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {
      "mod_5x5", morph_mod_5x5, METH_VARARGS, ""
    },
    {
      "mod_7x7", morph_mod_7x7, METH_VARARGS, ""
    },
    {
      "moe_5x5", morph_moe_5x5, METH_VARARGS, ""
    },
    {
      "moe_7x7", morph_moe_7x7, METH_VARARGS, ""
    },
    {
		  "ave", morph_ave, METH_VARARGS, ""
    },
    {
      "modmoe_7x7", morph_lap_7x7, METH_VARARGS, ""
    },
    {
      "modmoe_5x5", morph_lap_5x5, METH_VARARGS, ""
    },
    {
      "lodloe_5x5", morph_linear_lap_5x5, METH_VARARGS, ""
    },
    {
      "lodloe_7x7", morph_linear_lap_7x7, METH_VARARGS, ""
    },
    {
      "lod_5x5", morph_lod_5x5, METH_VARARGS, ""
    },
    {
      "loe_5x5", morph_loe_5x5, METH_VARARGS, ""
    },
    {
      "pool_7x7", m_pool_7x7, METH_VARARGS, ""
    },
    {
      "pool_3x3", m_pool_3x3, METH_VARARGS, ""
    },
    {
      "pool_5x5", m_pool_5x5, METH_VARARGS, ""
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef morphmodule = {
	PyModuleDef_HEAD_INIT,
	"morph",
	NULL,
	-1,
	methods
};

PyMODINIT_FUNC PyInit_morph(void) {
	return PyModule_Create(&morphmodule);
}
