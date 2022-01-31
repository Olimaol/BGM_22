/*
 *  ANNarchy-version: 4.7.0.1
 */
#pragma once
#include "ANNarchy.h"
#include <random>


extern double dt;
extern long int t;
extern std::vector<std::mt19937> rng;


///////////////////////////////////////////////////////////////
// Main Structure for the population of id 1 (STR_D1)
///////////////////////////////////////////////////////////////
struct PopStruct1{

    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission

    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int d) { max_delay  = d; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }



    // Structures for managing spikes
    std::vector<long int> last_spike;
    std::vector<int> spiked;

    // Neuron specific parameters and variables

    // Global parameter a
    double  a ;

    // Global parameter b
    double  b ;

    // Global parameter c
    double  c ;

    // Global parameter d
    double  d ;

    // Global parameter n0
    double  n0 ;

    // Global parameter n1
    double  n1 ;

    // Global parameter n2
    double  n2 ;

    // Global parameter I
    double  I ;

    // Global parameter tau_ampa
    double  tau_ampa ;

    // Global parameter tau_gaba
    double  tau_gaba ;

    // Global parameter E_ampa
    double  E_ampa ;

    // Global parameter E_gaba
    double  E_gaba ;

    // Global parameter Vr
    double  Vr ;

    // Global parameter C
    double  C ;

    // Local variable g_ampa
    std::vector< double > g_ampa;

    // Local variable g_gaba
    std::vector< double > g_gaba;

    // Local variable v
    std::vector< double > v;

    // Local variable u
    std::vector< double > u;

    // Local variable r
    std::vector< double > r;

    // Random numbers



    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;
    long int _mean_fr_window;
    double _mean_fr_rate;
    void compute_firing_rate(double window){
        if(window>0.0){
            _mean_fr_window = int(window/dt);
            _mean_fr_rate = 1000./window;
        }
    };


    // Access methods to the parameters and variables

    std::vector<double> get_local_attribute_all_double(std::string name) {

        // Local variable g_ampa
        if ( name.compare("g_ampa") == 0 ) {
            return g_ampa;
        }

        // Local variable g_gaba
        if ( name.compare("g_gaba") == 0 ) {
            return g_gaba;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v;
        }

        // Local variable u
        if ( name.compare("u") == 0 ) {
            return u;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r;
        }


        // should not happen
        std::cerr << "PopStruct1::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk) {
        assert( (rk < size) );

        // Local variable g_ampa
        if ( name.compare("g_ampa") == 0 ) {
            return g_ampa[rk];
        }

        // Local variable g_gaba
        if ( name.compare("g_gaba") == 0 ) {
            return g_gaba[rk];
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v[rk];
        }

        // Local variable u
        if ( name.compare("u") == 0 ) {
            return u[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r[rk];
        }


        // should not happen
        std::cerr << "PopStruct1::get_local_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_local_attribute_all_double(std::string name, std::vector<double> value) {
        assert( (value.size() == size) );

        // Local variable g_ampa
        if ( name.compare("g_ampa") == 0 ) {
            g_ampa = value;
            return;
        }

        // Local variable g_gaba
        if ( name.compare("g_gaba") == 0 ) {
            g_gaba = value;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v = value;
            return;
        }

        // Local variable u
        if ( name.compare("u") == 0 ) {
            u = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct1::set_local_attribute_all_double: " << name << " not found" << std::endl;
    }

    void set_local_attribute_double(std::string name, int rk, double value) {
        assert( (rk < size) );

        // Local variable g_ampa
        if ( name.compare("g_ampa") == 0 ) {
            g_ampa[rk] = value;
            return;
        }

        // Local variable g_gaba
        if ( name.compare("g_gaba") == 0 ) {
            g_gaba[rk] = value;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v[rk] = value;
            return;
        }

        // Local variable u
        if ( name.compare("u") == 0 ) {
            u[rk] = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct1::set_local_attribute_double: " << name << " not found" << std::endl;
    }

    double get_global_attribute_double(std::string name) {

        // Global parameter a
        if ( name.compare("a") == 0 ) {
            return a;
        }

        // Global parameter b
        if ( name.compare("b") == 0 ) {
            return b;
        }

        // Global parameter c
        if ( name.compare("c") == 0 ) {
            return c;
        }

        // Global parameter d
        if ( name.compare("d") == 0 ) {
            return d;
        }

        // Global parameter n0
        if ( name.compare("n0") == 0 ) {
            return n0;
        }

        // Global parameter n1
        if ( name.compare("n1") == 0 ) {
            return n1;
        }

        // Global parameter n2
        if ( name.compare("n2") == 0 ) {
            return n2;
        }

        // Global parameter I
        if ( name.compare("I") == 0 ) {
            return I;
        }

        // Global parameter tau_ampa
        if ( name.compare("tau_ampa") == 0 ) {
            return tau_ampa;
        }

        // Global parameter tau_gaba
        if ( name.compare("tau_gaba") == 0 ) {
            return tau_gaba;
        }

        // Global parameter E_ampa
        if ( name.compare("E_ampa") == 0 ) {
            return E_ampa;
        }

        // Global parameter E_gaba
        if ( name.compare("E_gaba") == 0 ) {
            return E_gaba;
        }

        // Global parameter Vr
        if ( name.compare("Vr") == 0 ) {
            return Vr;
        }

        // Global parameter C
        if ( name.compare("C") == 0 ) {
            return C;
        }


        // should not happen
        std::cerr << "PopStruct1::get_global_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_global_attribute_double(std::string name, double value)  {

        // Global parameter a
        if ( name.compare("a") == 0 ) {
            a = value;
            return;
        }

        // Global parameter b
        if ( name.compare("b") == 0 ) {
            b = value;
            return;
        }

        // Global parameter c
        if ( name.compare("c") == 0 ) {
            c = value;
            return;
        }

        // Global parameter d
        if ( name.compare("d") == 0 ) {
            d = value;
            return;
        }

        // Global parameter n0
        if ( name.compare("n0") == 0 ) {
            n0 = value;
            return;
        }

        // Global parameter n1
        if ( name.compare("n1") == 0 ) {
            n1 = value;
            return;
        }

        // Global parameter n2
        if ( name.compare("n2") == 0 ) {
            n2 = value;
            return;
        }

        // Global parameter I
        if ( name.compare("I") == 0 ) {
            I = value;
            return;
        }

        // Global parameter tau_ampa
        if ( name.compare("tau_ampa") == 0 ) {
            tau_ampa = value;
            return;
        }

        // Global parameter tau_gaba
        if ( name.compare("tau_gaba") == 0 ) {
            tau_gaba = value;
            return;
        }

        // Global parameter E_ampa
        if ( name.compare("E_ampa") == 0 ) {
            E_ampa = value;
            return;
        }

        // Global parameter E_gaba
        if ( name.compare("E_gaba") == 0 ) {
            E_gaba = value;
            return;
        }

        // Global parameter Vr
        if ( name.compare("Vr") == 0 ) {
            Vr = value;
            return;
        }

        // Global parameter C
        if ( name.compare("C") == 0 ) {
            C = value;
            return;
        }


        std::cerr << "PopStruct1::set_global_attribute_double: " << name << " not found" << std::endl;
    }



    // Method called to initialize the data structures
    void init_population() {
    #ifdef _DEBUG
        std::cout << "PopStruct1::init_population() - this = " << this << std::endl;
    #endif
        _active = true;

        // Global parameter a
        a = 0.0;

        // Global parameter b
        b = 0.0;

        // Global parameter c
        c = 0.0;

        // Global parameter d
        d = 0.0;

        // Global parameter n0
        n0 = 0.0;

        // Global parameter n1
        n1 = 0.0;

        // Global parameter n2
        n2 = 0.0;

        // Global parameter I
        I = 0.0;

        // Global parameter tau_ampa
        tau_ampa = 0.0;

        // Global parameter tau_gaba
        tau_gaba = 0.0;

        // Global parameter E_ampa
        E_ampa = 0.0;

        // Global parameter E_gaba
        E_gaba = 0.0;

        // Global parameter Vr
        Vr = 0.0;

        // Global parameter C
        C = 0.0;

        // Local variable g_ampa
        g_ampa = std::vector<double>(size, 0.0);

        // Local variable g_gaba
        g_gaba = std::vector<double>(size, 0.0);

        // Local variable v
        v = std::vector<double>(size, 0.0);

        // Local variable u
        u = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);


        // Spiking variables
        spiked = std::vector<int>();
        last_spike = std::vector<long int>(size, -10000L);



        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());
        _mean_fr_window = 0;
        _mean_fr_rate = 1.0;


    }

    // Method called to reset the population
    void reset() {

        spiked.clear();
        spiked.shrink_to_fit();
        last_spike.clear();
        last_spike = std::vector<long int>(size, -10000L);



    }

    // Init rng dist
    void init_rng_dist() {

    }

    // Method to draw new random numbers
    void update_rng() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct1::update_rng()" << std::endl;
#endif

    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {

    }

    // Main method to update neural variables
    void update() {

        if( _active ) {

            spiked.clear();

            // Updating local variables
            #pragma omp simd
            for(int i = 0; i < size; i++){

                // dg_ampa/dt = -g_ampa / tau_ampa
                double _g_ampa = -g_ampa[i]/tau_ampa;

                // dg_gaba/dt = -g_gaba / tau_gaba
                double _g_gaba = -g_gaba[i]/tau_gaba;

                // dv/dt      = n2 * v * v + n1 * v + n0 - u / C + I / C - g_ampa * (v - E_ampa) - g_gaba * (v - E_gaba)
                double _v = (C*(E_ampa*g_ampa[i] + E_gaba*g_gaba[i] - g_ampa[i]*v[i] - g_gaba[i]*v[i] + n0 + n1*v[i] + n2*pow(v[i], 2)) + I - u[i])/C;

                // du/dt      = a *(b*(v-Vr)-u)
                double _u = a*(-Vr*b + b*v[i] - u[i]);

                // dg_ampa/dt = -g_ampa / tau_ampa
                g_ampa[i] += dt*_g_ampa ;


                // dg_gaba/dt = -g_gaba / tau_gaba
                g_gaba[i] += dt*_g_gaba ;


                // dv/dt      = n2 * v * v + n1 * v + n0 - u / C + I / C - g_ampa * (v - E_ampa) - g_gaba * (v - E_gaba)
                v[i] += dt*_v ;


                // du/dt      = a *(b*(v-Vr)-u)
                u[i] += dt*_u ;


            }
        } // active

    }

    void spike_gather() {

        if( _active ) {
            for (int i = 0; i < size; i++) {


                // Spike emission
                if(v[i] >= 40){ // Condition is met
                    // Reset variables

                    v[i] = c;

                    u[i] = d + u[i];

                    // Store the spike
                    spiked.push_back(i);
                    last_spike[i] = t;

                    // Refractory period


                    // Update the mean firing rate
                    if(_mean_fr_window> 0)
                        _spike_history[i].push(t);

                }

                // Update the mean firing rate
                if(_mean_fr_window> 0){
                    while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                        _spike_history[i].pop(); // Suppress spikes outside the window
                    }
                    r[i] = _mean_fr_rate * double(_spike_history[i].size());
                }



            }
        } // active

    }



    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(double);	// a
        size_in_bytes += sizeof(double);	// b
        size_in_bytes += sizeof(double);	// c
        size_in_bytes += sizeof(double);	// d
        size_in_bytes += sizeof(double);	// n0
        size_in_bytes += sizeof(double);	// n1
        size_in_bytes += sizeof(double);	// n2
        size_in_bytes += sizeof(double);	// I
        size_in_bytes += sizeof(double);	// tau_ampa
        size_in_bytes += sizeof(double);	// tau_gaba
        size_in_bytes += sizeof(double);	// E_ampa
        size_in_bytes += sizeof(double);	// E_gaba
        size_in_bytes += sizeof(double);	// Vr
        size_in_bytes += sizeof(double);	// C
        // Variables
        size_in_bytes += sizeof(double) * g_ampa.capacity();	// g_ampa
        size_in_bytes += sizeof(double) * g_gaba.capacity();	// g_gaba
        size_in_bytes += sizeof(double) * v.capacity();	// v
        size_in_bytes += sizeof(double) * u.capacity();	// u
        size_in_bytes += sizeof(double) * r.capacity();	// r

        return size_in_bytes;
    }

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct1::clear() - this = " << this << std::endl;
#endif
        // Variables
        g_ampa.clear();
        g_ampa.shrink_to_fit();
        g_gaba.clear();
        g_gaba.shrink_to_fit();
        v.clear();
        v.shrink_to_fit();
        u.clear();
        u.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();

    }
};

