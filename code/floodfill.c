#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>

/* List of hard coded functions, these guys don't have fancy features like type correction and broadcasting */

/* constants used during filling */
npy_uint8 UNCLAIMED_CELL = 0;
npy_uint8 UNCLAIMED_BOUNDARY = 1;
npy_uint8 CLAIMED_CELL_OR_BOUNDARY = 2;

/* flags used */
npy_uint8 IS_BOUNDARY = 1;

struct FillTarget {
    npy_intp index;
    npy_uint8 flags; 
};

static inline long getIndex(long y, long x, npy_intp height, npy_intp width) {
    return y * width + x;
}
static inline long getY(long index, npy_intp height, npy_intp width) {
    return (index - (index % width))/width;  
}
static inline long getX(long index, npy_intp height, npy_intp width) {
    return index % width; 
}

static inline int isOutOfBounds(long y, long x, npy_intp height, npy_intp width) {
    return y < 0 || x < 0 || y >= height || x >= width; 
}

static PyObject *first_unclaimed(PyObject *self, PyObject * args) {
    long index;
    npy_intp found = -1;
    PyArrayObject* arr;
    PyArg_ParseTuple(args, "Ol", &arr, &index);
    if (PyErr_Occurred()) {
        goto ExceptEarly;
    }
    if (!PyArray_Check(arr) || PyArray_TYPE(arr) != NPY_UINT8 || !PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be c-contiguous uint8 numpy array");
        goto ExceptEarly;
    }
    
    npy_uint8* data = PyArray_DATA(arr);
    npy_intp length = PyArray_SIZE(arr);
    for(npy_intp i = index + 1; i < length; i++) {
        if(data[i] == UNCLAIMED_CELL || data[i] == UNCLAIMED_BOUNDARY) {
            found = i;
            break;
        }
    }
    return Py_BuildValue("i",found); 

ExceptMid:
    // array loaded successfully but arguments were invalid
ExceptEarly:    
    // error occured before memory was allocated or references were increased
    return NULL;    
}



static npy_intp argMax(npy_float* array, npy_intp len) {
    npy_intp argmax = 0;
    for(npy_intp j = 1; j < len; j++) {
        if(array[argmax] < array[j]) {
            argmax = j;
        }
    }
    return argmax; 
}
// compute norm of a vector
static inline npy_float norm(npy_float* arr, npy_intp len) {
    npy_float total = 0;
    for(npy_intp i =0; i < len; i++) {
        total += arr[i];
    }
    return total; 
}

// compute L1 distance between two vectors where arr1 is already L1 normalized
static inline npy_float computeDistance(npy_float* arr1, npy_float* arr2, npy_intp len) {
    npy_float total = 0;
    npy_float norm2 = norm(arr2, len);
    for(npy_intp i =0; i < len; i++) {
        total += fabs(arr1[i] - arr2[i]/norm2); 
    }
    return total;
}

struct CandidatePixel {
    npy_intp index;
    npy_float patience; 
};

/* params:
    mapping: H, W <int> array           // masks flattens to this, -1 for unassigned
    seg: H, W, C <float32> array        // raw segmentation
    masks: M + C, H, W <uint8> array    // existing masks plus space for C more individual maks
    stats: M + C, C <float32> array     // cumulative stats of masked pixels
    new_threshold: C <float32> array    // how far in norm a pixel has to be to create a new group
    join_threshold: float32             // how close in norm a pixel has to be to join a group
    impatience: float32                 // norm multiplier per pass > 1, smaller values make us wait longer before 
                                        // assigning a pixel to a group 
*/
static PyObject *fill_dust(PyObject *self, PyObject * args) {
    PyArrayObject* mapping;
    PyArrayObject* seg;
    PyArrayObject* masks;
    PyArrayObject* stats;
    PyArrayObject* new_threshold;
    npy_float32 join_threshold;
    npy_float32 impatience; 
    PyArg_ParseTuple(args, "OOOOOff", &mapping, &seg, &masks, &stats, &new_threshold, &join_threshold, &impatience);
    int cycles = 0;
    // perform basic type checking
    if (PyErr_Occurred()) {
        goto ExceptEarly;
    }
    if (!PyArray_Check(mapping) || PyArray_TYPE(mapping) != NPY_INTP || !PyArray_IS_C_CONTIGUOUS(mapping)) {
        PyErr_SetString(PyExc_TypeError, "First argument <mapping> must be c-contiguous int numpy array");
        goto ExceptEarly;
    }
    if (!PyArray_Check(seg) || PyArray_TYPE(seg) != NPY_FLOAT32 || !PyArray_IS_C_CONTIGUOUS(seg)) {
        PyErr_SetString(PyExc_TypeError, "Second argument <seg> must be c-contiguous float32 numpy array");
        goto ExceptEarly;
    }
    if (!PyArray_Check(masks) || PyArray_TYPE(masks) != NPY_UINT8 || !PyArray_IS_C_CONTIGUOUS(masks)) {
        PyErr_SetString(PyExc_TypeError, "Third argument <masks> must be c-contiguous uint8 numpy array");
        goto ExceptEarly;
    }
    if (!PyArray_Check(stats) || PyArray_TYPE(stats) != NPY_FLOAT32 || !PyArray_IS_C_CONTIGUOUS(stats)) {
        PyErr_SetString(PyExc_TypeError, "Fourth argument <stats> must be c-contiguous float32 numpy array");
        goto ExceptEarly;
    }
    if (!PyArray_Check(new_threshold) || PyArray_TYPE(new_threshold) != NPY_FLOAT32 || !PyArray_IS_C_CONTIGUOUS(new_threshold)) {
        PyErr_SetString(PyExc_TypeError, "Fourth argument <join_threshold> must be c-contiguous float32 numpy array");
        goto ExceptEarly;
    }
    // mapping
    int mapping_dim = PyArray_NDIM(mapping);
    npy_intp* mapping_shape = PyArray_SHAPE(mapping);
    if(mapping_dim != 2) {
        PyErr_SetString(PyExc_TypeError, "First argument <mapping> must be 2D");
        goto ExceptEarly;
    }
    npy_intp HEIGHT = mapping_shape[0];
    npy_intp WIDTH = mapping_shape[1];
    // seg
    int seg_dim = PyArray_NDIM(seg);
    npy_intp* seg_shape = PyArray_SHAPE(seg);
    if(seg_dim != 3) {
        PyErr_SetString(PyExc_TypeError, "Second argument <seg> must be H, W, C array");
        goto ExceptEarly;
    }
    if(seg_shape[0] != HEIGHT || seg_shape[1] != WIDTH) {
        PyErr_SetString(PyExc_TypeError, "Second argument <seg> shape [0, 1] must agree with <mapping> [0, 1]");
        goto ExceptEarly;
    }
    npy_intp LABELS = seg_shape[2]; 
    // masks
    int masks_dim = PyArray_NDIM(masks);
    npy_intp* masks_shape = PyArray_SHAPE(masks);
    if(masks_dim != 3) {
        PyErr_SetString(PyExc_TypeError, "Third argument <masks> must be M+C, H, W array");
        goto ExceptEarly;
    }
    if(masks_shape[1] != HEIGHT || masks_shape[2] != WIDTH) {
        PyErr_SetString(PyExc_TypeError, "Third argument <masks> shape [1, 2] must agree with <mapping> [0,1]");
        goto ExceptEarly;
    }
    if(masks_shape[0] <= LABELS) {
        PyErr_SetString(PyExc_TypeError, "Third argument <masks> should have at least one seed partition");
         goto ExceptEarly;
    }
    npy_intp MASKS = masks_shape[0] - LABELS;
    // stats
    int stats_dim = PyArray_NDIM(stats);
    npy_intp* stats_shape = PyArray_SHAPE(stats);
    if(stats_dim != 2) {
        PyErr_SetString(PyExc_TypeError, "Fourth argument <stats> must be M+C, C array");
        goto ExceptEarly;
    }
    if(stats_shape[0] != LABELS + MASKS || stats_shape[1] != LABELS) {
        PyErr_SetString(PyExc_TypeError, "Fourth argument <stats> shape [0] must agree with <masks> [0], shape[1] must agree with <seg> [2]");
        goto ExceptEarly;
    }
    // new_threshold
    int new_threshold_dim = PyArray_NDIM(new_threshold);
    npy_intp* new_threshold_shape = PyArray_SHAPE(new_threshold);
    if(new_threshold_dim != 1) {
        PyErr_SetString(PyExc_TypeError, "Fifth argument <new_threshold> must be C array");
        goto ExceptEarly;
    }
    if(new_threshold_shape[0] != LABELS ) {
        PyErr_SetString(PyExc_TypeError, "Fifth argument <new_threshold> shape [0] must agree with <seg> [2]");
        goto ExceptEarly;
    }
    // validate join_threshold
    if(join_threshold <= 0) {
        PyErr_SetString(PyExc_TypeError, "join_threshold must be positive definite");
        goto ExceptEarly;
    }
    // validate impatience 
    if(impatience <= 1) {
        PyErr_SetString(PyExc_TypeError, "impatience must be larger than 1");
        goto ExceptEarly;
    }
    // everything is valid, we can now get access to the data
    npy_intp* mapping_data = PyArray_DATA(mapping);
    npy_float* seg_data = PyArray_DATA(seg);
    npy_uint8* masks_data = PyArray_DATA(masks);
    npy_float* stats_data = PyArray_DATA(stats);
    npy_float* new_threshold_data = PyArray_DATA(new_threshold);
    // allocate a stack and swap buffer
    struct CandidatePixel* buffer = malloc(HEIGHT * WIDTH * 2 * sizeof(struct CandidatePixel));
    struct CandidatePixel* stack = buffer;
    // points at insertion position
    npy_intp stack_ptr = 0;
    struct CandidatePixel* swap_buffer = buffer + HEIGHT*WIDTH; 
    // points at insertion position
    npy_intp swap_buffer_ptr = 0;
    if(stack == 0) {
        PyErr_SetString(PyExc_TypeError, "Failed to allocate memory");
        goto ExceptEarly; 
    }
    // perform a single pass, promote pixels that exceed new_threshold to their own layer
    // put everything else unassigned onto the stack
    for(npy_intp index =0; index < HEIGHT * WIDTH; index++) {
        // skip mapped pixels
        if(mapping_data[index] >= 0) {
            continue;
        }
        // pointer to index data
        npy_float* seg_index_data = seg_data + index * LABELS; 
        // get the maximum
        npy_intp argmax = argMax(seg_index_data, LABELS);
        if(seg_index_data[argmax] >= new_threshold_data[argmax]) {
            // pixel exceeds threshold and can be populated in the promotion layer
            mapping_data[index] = MASKS + argmax; 
            masks_data[(MASKS + argmax) * HEIGHT * WIDTH + index] = 1;
            // we do not change pixel statistics in the promotion layer
        } else {
            // pixel to be processed in later passes
            stack[stack_ptr].index = index;
            stack[stack_ptr].patience = 1.0;
            stack_ptr++;
        }
    }
    // proceed until all pixels have been processed
    while(stack_ptr > 0) {
        cycles++;
        while(stack_ptr > 0) {
            stack_ptr--; 
            npy_intp index = stack[stack_ptr].index;
            npy_float patience = stack[stack_ptr].patience;
            // deviation from closest mapped cell
            npy_float distance = 10000; 
            npy_int closest_mask = -1;
            
            // get neighbour cells 
            npy_intp x = getX(index, HEIGHT, WIDTH);
            npy_intp y = getY(index, HEIGHT, WIDTH); 

            // copy paste the block 
            if(!isOutOfBounds(y + 1 ,x, HEIGHT, WIDTH)) {
                npy_intp targetIndex = getIndex(y+1, x, HEIGHT, WIDTH);
                npy_intp map_index = mapping_data[targetIndex];
                if(map_index >= 0) {
                    // this neighbour is mapped, compute the distance
                    npy_float d = computeDistance(seg_data + index * LABELS, stats_data + map_index * LABELS, LABELS);
                    if(d < distance) {
                        distance = d;
                        closest_mask = map_index; 
                    }
                }
            }
            if(!isOutOfBounds(y - 1 ,x, HEIGHT, WIDTH)) {
                npy_intp targetIndex = getIndex(y-1, x, HEIGHT, WIDTH);
                npy_intp map_index = mapping_data[targetIndex];
                if(map_index >= 0) {
                    // this neighbour is mapped, compute the distance
                    npy_float d = computeDistance(seg_data + index * LABELS, stats_data + map_index * LABELS, LABELS);
                    if(d < distance) {
                        distance = d;
                        closest_mask = map_index; 
                    }
                }
            }
            if(!isOutOfBounds(y,x + 1, HEIGHT, WIDTH)) {
                npy_intp targetIndex = getIndex(y, x + 1, HEIGHT, WIDTH);
                npy_intp map_index = mapping_data[targetIndex];
                if(map_index >= 0) {
                    // this neighbour is mapped, compute the distance
                    npy_float d = computeDistance(seg_data + index * LABELS, stats_data + map_index * LABELS, LABELS);
                    if(d < distance) {
                        distance = d;
                        closest_mask = map_index; 
                    }
                }
            }
            if(!isOutOfBounds(y, x -1, HEIGHT, WIDTH)) {
                npy_intp targetIndex = getIndex(y, x - 1, HEIGHT, WIDTH);
                npy_intp map_index = mapping_data[targetIndex];
                if(map_index >= 0) {
                    // this neighbour is mapped, compute the distance
                    npy_float d = computeDistance(seg_data + index * LABELS, stats_data + map_index * LABELS, LABELS);
                    if(d < distance) {
                        distance = d;
                        closest_mask = map_index; 
                    }
                }
            }
            if(closest_mask >= 0 && distance < join_threshold * patience) {
                // we can safely assign the pixel
                mapping_data[index] = closest_mask; 
                masks_data[closest_mask * HEIGHT * WIDTH + index] = 1;
                // accumulate stats for unpromoted masks
                if(closest_mask < MASKS) {
                    for(npy_intp i =0; i < LABELS; i++) {
                        stats_data[closest_mask * LABELS + i] += seg_data[index * LABELS + i]; 
                    }
                }
            } else {
                // put it into the swap buffer
                swap_buffer[swap_buffer_ptr].index = index;
                // increase patience if at least one mask is adjescent 
                swap_buffer[swap_buffer_ptr].patience = closest_mask >= 0 ? patience * impatience : patience; 
                swap_buffer_ptr++;  
            }
        }
        // swap the stack and swap_buffer, clear the swap_buffer position
        stack_ptr = swap_buffer_ptr;
        swap_buffer_ptr = 0;
        struct CandidatePixel* temp = stack;
        stack = swap_buffer;
        swap_buffer = temp; 
    }
    //printf("total cycles: %ld", cycles); 
    // free memory
    free(buffer);
    Py_RETURN_NONE;
ExceptEarly:
    return NULL; 
}




/* uint* 2d_array, long y, long x */
static PyObject *flood(PyObject *self, PyObject * args) {
    long y;
    long x;
    PyArrayObject* arr;
    PyArrayObject* mask; 
    PyArg_ParseTuple(args, "OOll", &arr, &mask, &y, &x);
    if (PyErr_Occurred()) {
        goto ExceptEarly;
    }
    if (!PyArray_Check(arr) || PyArray_TYPE(arr) != NPY_UINT8 || !PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be c-contiguous uint8 numpy array");
        goto ExceptEarly;
    }
    if (!PyArray_Check(mask) || PyArray_TYPE(mask) != NPY_UINT8 || !PyArray_IS_C_CONTIGUOUS(mask)) {
        PyErr_SetString(PyExc_TypeError, "Second argument must be c-contiguous uint8 numpy array");
        goto ExceptEarly;
    }
    npy_uint8* data = PyArray_DATA(arr);
    int length = PyArray_NDIM(arr);
    npy_intp* shape = PyArray_SHAPE(arr);
    npy_intp height = shape[0];
    npy_intp width = shape[1];
    if(length != 2) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a 2D array");
        goto ExceptMid;
    }  
    npy_uint8* mask_data = PyArray_DATA(mask);
    int mask_length = PyArray_NDIM(mask);
    npy_intp* mask_shape = PyArray_SHAPE(mask);
    if(mask_length != 2 || mask_shape[0] != height || mask_shape[1] != width) {
        PyErr_SetString(PyExc_TypeError, "First and Second argument must have identical dimensions");
        goto ExceptMid;
    }
    if(x < 0 || y < 0 || y >= height || x >= width) {
        PyErr_SetString(PyExc_TypeError, "Coordinates are out of bounds");
        goto ExceptMid;
    }
    // allocate an oversized array to hold all possible targets
    struct FillTarget* stack = malloc(height * width * 4 * sizeof(struct FillTarget));
    if(stack == 0) {
        PyErr_SetString(PyExc_TypeError, "Failed to allocate memory");
        goto ExceptMid;
    }
    // track filled cells
    long filled = 0;
    // initialize the stack pointer
    npy_intp stack_ptr = 0; 
    stack[stack_ptr].index = getIndex(y, x, height, width);
    stack[stack_ptr].flags = 0;
    while(stack_ptr >= 0) {
        // get current target from the stack
        struct FillTarget target = stack[stack_ptr];
        stack_ptr--;
        y = getY(target.index, height, width);
        x = getX(target.index, height, width);
        npy_uint8 val = data[target.index]; 
        // boundary cell cannot claim another boundary cell
        if((target.flags & IS_BOUNDARY) > 0 && val == UNCLAIMED_BOUNDARY) {
            continue;
        }
        // only unclaimed cells are valid
        if(val != UNCLAIMED_BOUNDARY && val != UNCLAIMED_CELL) {
            continue;
        }
        npy_uint8 flags = 0;
        if(val == UNCLAIMED_CELL) {
            data[target.index] = CLAIMED_CELL_OR_BOUNDARY;
            mask_data[target.index] = 1;
            filled++;
        } else {
            data[target.index] = CLAIMED_CELL_OR_BOUNDARY;
            mask_data[target.index] = 1;
            flags = IS_BOUNDARY;
            filled++;
        }
        // since we are using an index, we have to check bounds here, otherwise things will just wrap
        if(!isOutOfBounds(y+1, x, height, width)) {
            stack_ptr++;
            stack[stack_ptr].index = getIndex(y+1, x, height, width); 
            stack[stack_ptr].flags = flags;
        }
        if(!isOutOfBounds(y-1, x, height, width)) {
            stack_ptr++;
            stack[stack_ptr].index = getIndex(y-1, x, height, width); 
            stack[stack_ptr].flags = flags;
        }
        if(!isOutOfBounds(y, x + 1, height, width)) {
            stack_ptr++;
            stack[stack_ptr].index = getIndex(y, x + 1, height, width); 
            stack[stack_ptr].flags = flags;
        }
        if(!isOutOfBounds(y, x - 1, height, width)) {
            stack_ptr++;
            stack[stack_ptr].index = getIndex(y, x - 1, height, width); 
            stack[stack_ptr].flags = flags;
        }
    }
    free(stack);
    // make sure that return types are working
    return PyInt_FromLong(filled);

// handle exceptions here
// currently unreachable but just in case 
ExceptLate:
    free(stack);
ExceptMid:
    // array loaded successfully but arguments were invalid
ExceptEarly:    
    // error occured before memory was allocated or references were increased
    return NULL;    
}

static PyMethodDef FloodMethods[] = {
    {"flood", flood, METH_VARARGS, "fill region starting at given coordinate"},
    {"first_unclaimed", first_unclaimed, METH_VARARGS, "Get first unclaimed index in array"},
    {"fill_dust", fill_dust, METH_VARARGS, "Populate dust into masks"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "floodfill",
    "implements flood fill algorithm in C",
    -1,
    FloodMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_floodfill(void)
{
    PyObject *m, *logit, *d;
    import_array();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}