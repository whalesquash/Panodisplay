#cython: boundscheck=False, wraparound=False, nonecheck=False
#%%cython --annotate

#First comment are compiler directives.

#This cython module massively speeds up the loop in imageDeform.

from cython.parallel import prange

cpdef loop(short[:,::1] x,float[:,::1] xmap,float[:,::1] ymap, int w, int h, int wT):
    cdef:
        Py_ssize_t j
        Py_ssize_t i
        Py_ssize_t i2
        Py_ssize_t l
        short start
        short stop
        short tmp = 2*wT
    for j in range(h):
        for i in range(w): 
            if x[j,i] == 0: #limit condition
                i2 = i -1
                start = x[j,i2] #negative
                stop = tmp
                start = tmp + start
                for l in range(start,stop):
                    xmap[j, l] = i 
                    ymap[j, l] = j
            else:
                if i != 0:
                    i2 = i -1
                    start = x[j,i2]
                    stop = x[j,i]
                    if start < 0: #if start negative stop negative
                        start = tmp + start
                        stop =  tmp + stop
                    for l in range(start, stop):
                        xmap[j, l] = i
                        ymap[j, l] = j
