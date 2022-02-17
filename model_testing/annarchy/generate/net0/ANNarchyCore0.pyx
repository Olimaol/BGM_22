# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
from libcpp.string cimport string
from math import ceil
import numpy as np
import sys
cimport numpy as np
cimport cython

# Short names for unsigned integer types
ctypedef unsigned char _ann_uint8
ctypedef unsigned short _ann_uint16
ctypedef unsigned int _ann_uint32
ctypedef unsigned long _ann_uint64

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport LILConnectivity as LIL

cdef extern from "ANNarchy.h":

    # User-defined functions


    # User-defined constants


    # Data structures

    # Export Population 0 (cor_go)
    cdef struct PopStruct0 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 1 (cor_pause)
    cdef struct PopStruct1 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 2 (cor_stop)
    cdef struct PopStruct2 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 3 (str_d1)
    cdef struct PopStruct3 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 4 (str_d2)
    cdef struct PopStruct4 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 5 (str_fsi)
    cdef struct PopStruct5 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 6 (stn)
    cdef struct PopStruct6 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()




    # Monitors
    cdef cppclass Monitor:
        vector[int] ranks
        int period_
        int period_offset_
        long offset_


    # Population 0 (cor_go) : Monitor
    cdef cppclass PopRecorder0 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder0* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] p
        bool record_p

        vector[vector[double]] act
        bool record_act

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 1 (cor_pause) : Monitor
    cdef cppclass PopRecorder1 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder1* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] p
        bool record_p

        vector[vector[double]] act
        bool record_act

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 2 (cor_stop) : Monitor
    cdef cppclass PopRecorder2 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder2* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] p
        bool record_p

        vector[vector[double]] act
        bool record_act

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 3 (str_d1) : Monitor
    cdef cppclass PopRecorder3 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder3* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] g_ampa
        bool record_g_ampa

        vector[vector[double]] g_gaba
        bool record_g_gaba

        vector[vector[double]] I_ampa
        bool record_I_ampa

        vector[vector[double]] I_gaba
        bool record_I_gaba

        vector[vector[double]] I
        bool record_I

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] u
        bool record_u

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 4 (str_d2) : Monitor
    cdef cppclass PopRecorder4 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder4* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] g_ampa
        bool record_g_ampa

        vector[vector[double]] g_gaba
        bool record_g_gaba

        vector[vector[double]] I_ampa
        bool record_I_ampa

        vector[vector[double]] I_gaba
        bool record_I_gaba

        vector[vector[double]] I
        bool record_I

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] u
        bool record_u

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 5 (str_fsi) : Monitor
    cdef cppclass PopRecorder5 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder5* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] g_ampa
        bool record_g_ampa

        vector[vector[double]] g_gaba
        bool record_g_gaba

        vector[vector[double]] I_ampa
        bool record_I_ampa

        vector[vector[double]] I_gaba
        bool record_I_gaba

        vector[vector[double]] I
        bool record_I

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] U_v
        bool record_U_v

        vector[vector[double]] u
        bool record_u

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 6 (stn) : Monitor
    cdef cppclass PopRecorder6 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder6* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] g_ampa
        bool record_g_ampa

        vector[vector[double]] g_gaba
        bool record_g_gaba

        vector[vector[double]] I_ampa
        bool record_I_ampa

        vector[vector[double]] I_gaba
        bool record_I_gaba

        vector[vector[double]] I
        bool record_I

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] u
        bool record_u

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()


    # Instances

    PopStruct0 pop0
    PopStruct1 pop1
    PopStruct2 pop2
    PopStruct3 pop3
    PopStruct4 pop4
    PopStruct5 pop5
    PopStruct6 pop6


    # Methods
    void initialize(double)
    void init_rng_dist()
    void setSeed(long, int, bool)
    void run(int nbSteps) nogil
    int run_until(int steps, vector[int] populations, bool or_and)
    void step()

    # Time
    long getTime()
    void setTime(long)

    # dt
    double getDt()
    void setDt(double dt_)


    # Number of threads
    void setNumberThreads(int, vector[int])


# Population wrappers

# Wrapper for population 0 (cor_go)
@cython.auto_pickle(True)
cdef class pop0_wrapper :

    def __init__(self, size, max_delay):

        pop0.set_size(size)
        pop0.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop0.get_size()
    # Reset the population
    def reset(self):
        pop0.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop0.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop0.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop0.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop0.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop0.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop0.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop0.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop0.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop0.set_global_attribute_double(cpp_string, value)






    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop0.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop0.size_in_bytes()

    def clear(self):
        return pop0.clear()

# Wrapper for population 1 (cor_pause)
@cython.auto_pickle(True)
cdef class pop1_wrapper :

    def __init__(self, size, max_delay):

        pop1.set_size(size)
        pop1.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop1.get_size()
    # Reset the population
    def reset(self):
        pop1.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop1.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop1.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop1.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop1.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop1.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop1.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop1.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop1.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop1.set_global_attribute_double(cpp_string, value)






    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop1.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop1.size_in_bytes()

    def clear(self):
        return pop1.clear()

# Wrapper for population 2 (cor_stop)
@cython.auto_pickle(True)
cdef class pop2_wrapper :

    def __init__(self, size, max_delay):

        pop2.set_size(size)
        pop2.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop2.get_size()
    # Reset the population
    def reset(self):
        pop2.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop2.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop2.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop2.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop2.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop2.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop2.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop2.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop2.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop2.set_global_attribute_double(cpp_string, value)






    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop2.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop2.size_in_bytes()

    def clear(self):
        return pop2.clear()

# Wrapper for population 3 (str_d1)
@cython.auto_pickle(True)
cdef class pop3_wrapper :

    def __init__(self, size, max_delay):

        pop3.set_size(size)
        pop3.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop3.get_size()
    # Reset the population
    def reset(self):
        pop3.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop3.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop3.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop3.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop3.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop3.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop3.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop3.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop3.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop3.set_global_attribute_double(cpp_string, value)






    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop3.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop3.size_in_bytes()

    def clear(self):
        return pop3.clear()

# Wrapper for population 4 (str_d2)
@cython.auto_pickle(True)
cdef class pop4_wrapper :

    def __init__(self, size, max_delay):

        pop4.set_size(size)
        pop4.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop4.get_size()
    # Reset the population
    def reset(self):
        pop4.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop4.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop4.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop4.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop4.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop4.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop4.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop4.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop4.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop4.set_global_attribute_double(cpp_string, value)






    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop4.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop4.size_in_bytes()

    def clear(self):
        return pop4.clear()

# Wrapper for population 5 (str_fsi)
@cython.auto_pickle(True)
cdef class pop5_wrapper :

    def __init__(self, size, max_delay):

        pop5.set_size(size)
        pop5.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop5.get_size()
    # Reset the population
    def reset(self):
        pop5.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop5.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop5.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop5.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop5.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop5.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop5.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop5.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop5.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop5.set_global_attribute_double(cpp_string, value)






    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop5.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop5.size_in_bytes()

    def clear(self):
        return pop5.clear()

# Wrapper for population 6 (stn)
@cython.auto_pickle(True)
cdef class pop6_wrapper :

    def __init__(self, size, max_delay):

        pop6.set_size(size)
        pop6.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop6.get_size()
    # Reset the population
    def reset(self):
        pop6.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop6.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop6.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop6.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop6.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop6.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop6.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop6.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop6.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop6.set_global_attribute_double(cpp_string, value)






    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop6.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop6.size_in_bytes()

    def clear(self):
        return pop6.clear()


# Projection wrappers


# Monitor wrappers

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder0_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder0.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder0.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder0.get_instance(self.id)).clear()

    property p:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).p
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).p = val
    property record_p:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_p
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_p = val
    def clear_p(self):
        (PopRecorder0.get_instance(self.id)).p.clear()

    property act:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).act
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).act = val
    property record_act:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_act
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_act = val
    def clear_act(self):
        (PopRecorder0.get_instance(self.id)).act.clear()

    property r:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder0.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder0.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder1_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder1.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder1.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder1.get_instance(self.id)).clear()

    property p:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).p
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).p = val
    property record_p:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_p
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_p = val
    def clear_p(self):
        (PopRecorder1.get_instance(self.id)).p.clear()

    property act:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).act
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).act = val
    property record_act:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_act
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_act = val
    def clear_act(self):
        (PopRecorder1.get_instance(self.id)).act.clear()

    property r:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder1.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder1.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder2_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder2.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder2.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder2.get_instance(self.id)).clear()

    property p:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).p
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).p = val
    property record_p:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_p
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_p = val
    def clear_p(self):
        (PopRecorder2.get_instance(self.id)).p.clear()

    property act:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).act
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).act = val
    property record_act:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_act
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_act = val
    def clear_act(self):
        (PopRecorder2.get_instance(self.id)).act.clear()

    property r:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder2.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder2.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder3_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder3.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder3.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder3.get_instance(self.id)).clear()

    property g_ampa:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).g_ampa
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).g_ampa = val
    property record_g_ampa:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_g_ampa
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_g_ampa = val
    def clear_g_ampa(self):
        (PopRecorder3.get_instance(self.id)).g_ampa.clear()

    property g_gaba:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).g_gaba
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).g_gaba = val
    property record_g_gaba:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_g_gaba
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_g_gaba = val
    def clear_g_gaba(self):
        (PopRecorder3.get_instance(self.id)).g_gaba.clear()

    property I_ampa:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).I_ampa
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).I_ampa = val
    property record_I_ampa:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_I_ampa
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_I_ampa = val
    def clear_I_ampa(self):
        (PopRecorder3.get_instance(self.id)).I_ampa.clear()

    property I_gaba:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).I_gaba
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).I_gaba = val
    property record_I_gaba:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_I_gaba
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_I_gaba = val
    def clear_I_gaba(self):
        (PopRecorder3.get_instance(self.id)).I_gaba.clear()

    property I:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).I
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).I = val
    property record_I:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_I
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_I = val
    def clear_I(self):
        (PopRecorder3.get_instance(self.id)).I.clear()

    property v:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder3.get_instance(self.id)).v.clear()

    property u:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).u
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).u = val
    property record_u:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_u
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_u = val
    def clear_u(self):
        (PopRecorder3.get_instance(self.id)).u.clear()

    property r:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder3.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder3.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder4_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder4.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder4.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder4.get_instance(self.id)).clear()

    property g_ampa:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).g_ampa
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).g_ampa = val
    property record_g_ampa:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_g_ampa
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_g_ampa = val
    def clear_g_ampa(self):
        (PopRecorder4.get_instance(self.id)).g_ampa.clear()

    property g_gaba:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).g_gaba
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).g_gaba = val
    property record_g_gaba:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_g_gaba
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_g_gaba = val
    def clear_g_gaba(self):
        (PopRecorder4.get_instance(self.id)).g_gaba.clear()

    property I_ampa:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).I_ampa
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).I_ampa = val
    property record_I_ampa:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_I_ampa
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_I_ampa = val
    def clear_I_ampa(self):
        (PopRecorder4.get_instance(self.id)).I_ampa.clear()

    property I_gaba:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).I_gaba
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).I_gaba = val
    property record_I_gaba:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_I_gaba
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_I_gaba = val
    def clear_I_gaba(self):
        (PopRecorder4.get_instance(self.id)).I_gaba.clear()

    property I:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).I
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).I = val
    property record_I:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_I
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_I = val
    def clear_I(self):
        (PopRecorder4.get_instance(self.id)).I.clear()

    property v:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder4.get_instance(self.id)).v.clear()

    property u:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).u
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).u = val
    property record_u:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_u
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_u = val
    def clear_u(self):
        (PopRecorder4.get_instance(self.id)).u.clear()

    property r:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder4.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder4.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder5_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder5.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder5.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder5.get_instance(self.id)).clear()

    property g_ampa:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).g_ampa
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).g_ampa = val
    property record_g_ampa:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_g_ampa
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_g_ampa = val
    def clear_g_ampa(self):
        (PopRecorder5.get_instance(self.id)).g_ampa.clear()

    property g_gaba:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).g_gaba
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).g_gaba = val
    property record_g_gaba:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_g_gaba
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_g_gaba = val
    def clear_g_gaba(self):
        (PopRecorder5.get_instance(self.id)).g_gaba.clear()

    property I_ampa:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).I_ampa
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).I_ampa = val
    property record_I_ampa:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_I_ampa
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_I_ampa = val
    def clear_I_ampa(self):
        (PopRecorder5.get_instance(self.id)).I_ampa.clear()

    property I_gaba:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).I_gaba
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).I_gaba = val
    property record_I_gaba:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_I_gaba
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_I_gaba = val
    def clear_I_gaba(self):
        (PopRecorder5.get_instance(self.id)).I_gaba.clear()

    property I:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).I
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).I = val
    property record_I:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_I
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_I = val
    def clear_I(self):
        (PopRecorder5.get_instance(self.id)).I.clear()

    property v:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder5.get_instance(self.id)).v.clear()

    property U_v:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).U_v
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).U_v = val
    property record_U_v:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_U_v
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_U_v = val
    def clear_U_v(self):
        (PopRecorder5.get_instance(self.id)).U_v.clear()

    property u:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).u
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).u = val
    property record_u:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_u
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_u = val
    def clear_u(self):
        (PopRecorder5.get_instance(self.id)).u.clear()

    property r:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder5.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder5.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder6_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder6.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder6.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder6.get_instance(self.id)).clear()

    property g_ampa:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).g_ampa
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).g_ampa = val
    property record_g_ampa:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_g_ampa
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_g_ampa = val
    def clear_g_ampa(self):
        (PopRecorder6.get_instance(self.id)).g_ampa.clear()

    property g_gaba:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).g_gaba
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).g_gaba = val
    property record_g_gaba:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_g_gaba
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_g_gaba = val
    def clear_g_gaba(self):
        (PopRecorder6.get_instance(self.id)).g_gaba.clear()

    property I_ampa:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).I_ampa
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).I_ampa = val
    property record_I_ampa:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_I_ampa
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_I_ampa = val
    def clear_I_ampa(self):
        (PopRecorder6.get_instance(self.id)).I_ampa.clear()

    property I_gaba:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).I_gaba
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).I_gaba = val
    property record_I_gaba:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_I_gaba
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_I_gaba = val
    def clear_I_gaba(self):
        (PopRecorder6.get_instance(self.id)).I_gaba.clear()

    property I:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).I
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).I = val
    property record_I:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_I
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_I = val
    def clear_I(self):
        (PopRecorder6.get_instance(self.id)).I.clear()

    property v:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder6.get_instance(self.id)).v.clear()

    property u:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).u
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).u = val
    property record_u:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_u
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_u = val
    def clear_u(self):
        (PopRecorder6.get_instance(self.id)).u.clear()

    property r:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder6.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder6.get_instance(self.id)).clear_spike()


# User-defined functions


# User-defined constants


# Initialize the network
def pyx_create(double dt):
    initialize(dt)

def pyx_init_rng_dist():
    init_rng_dist()

# Simple progressbar on the command line
def progress(count, total, status=''):
    """
    Prints a progress bar on the command line.

    adapted from: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    Modification: The original code set the '\r' at the end, so the bar disappears when finished.
    I moved it to the front, so the last status remains.
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()

# Simulation for the given number of steps
def pyx_run(int nb_steps, progress_bar):
    cdef int nb, rest
    cdef int batch = 1000
    if nb_steps < batch:
        with nogil:
            run(nb_steps)
    else:
        nb = int(nb_steps/batch)
        rest = nb_steps % batch
        for i in range(nb):
            with nogil:
                run(batch)
            PyErr_CheckSignals()
            if nb > 1 and progress_bar:
                progress(i+1, nb, 'simulate()')
        if rest > 0:
            run(rest)

        if (progress_bar):
            print('\n')

# Simulation for the given number of steps except if a criterion is reached
def pyx_run_until(int nb_steps, list populations, bool mode):
    cdef int nb
    nb = run_until(nb_steps, populations, mode)
    return nb

# Simulate for one step
def pyx_step():
    step()

# Access time
def set_time(t):
    setTime(t)
def get_time():
    return getTime()

# Access dt
def set_dt(double dt):
    setDt(dt)
def get_dt():
    return getDt()


# Set number of threads
def set_number_threads(int n, core_list):
    setNumberThreads(n, core_list)


# Set seed
def set_seed(long seed, int num_sources, use_seed_seq):
    setSeed(seed, num_sources, use_seed_seq)
