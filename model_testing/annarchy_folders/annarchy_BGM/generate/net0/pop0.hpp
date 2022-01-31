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
// Main Structure for the population of id 0 (Cortex-Go)
///////////////////////////////////////////////////////////////
struct PopStruct0{

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

    // Local parameter rates
    std::vector< double > rates;

    // Global parameter tauUP
    double  tauUP ;

    // Global parameter tauDOWN
    double  tauDOWN ;

    // Local variable p
    std::vector< double > p;

    // Local variable act
    std::vector< double > act;

    // Local variable r
    std::vector< double > r;

    // Random numbers
    std::vector<double> rand_0;
    std::uniform_real_distribution< double > dist_rand_0;



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

        // Local parameter rates
        if ( name.compare("rates") == 0 ) {
            return rates;
        }

        // Local variable p
        if ( name.compare("p") == 0 ) {
            return p;
        }

        // Local variable act
        if ( name.compare("act") == 0 ) {
            return act;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r;
        }


        // should not happen
        std::cerr << "PopStruct0::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk) {
        assert( (rk < size) );

        // Local parameter rates
        if ( name.compare("rates") == 0 ) {
            return rates[rk];
        }

        // Local variable p
        if ( name.compare("p") == 0 ) {
            return p[rk];
        }

        // Local variable act
        if ( name.compare("act") == 0 ) {
            return act[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r[rk];
        }


        // should not happen
        std::cerr << "PopStruct0::get_local_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_local_attribute_all_double(std::string name, std::vector<double> value) {
        assert( (value.size() == size) );

        // Local parameter rates
        if ( name.compare("rates") == 0 ) {
            rates = value;
            return;
        }

        // Local variable p
        if ( name.compare("p") == 0 ) {
            p = value;
            return;
        }

        // Local variable act
        if ( name.compare("act") == 0 ) {
            act = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct0::set_local_attribute_all_double: " << name << " not found" << std::endl;
    }

    void set_local_attribute_double(std::string name, int rk, double value) {
        assert( (rk < size) );

        // Local parameter rates
        if ( name.compare("rates") == 0 ) {
            rates[rk] = value;
            return;
        }

        // Local variable p
        if ( name.compare("p") == 0 ) {
            p[rk] = value;
            return;
        }

        // Local variable act
        if ( name.compare("act") == 0 ) {
            act[rk] = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct0::set_local_attribute_double: " << name << " not found" << std::endl;
    }

    double get_global_attribute_double(std::string name) {

        // Global parameter tauUP
        if ( name.compare("tauUP") == 0 ) {
            return tauUP;
        }

        // Global parameter tauDOWN
        if ( name.compare("tauDOWN") == 0 ) {
            return tauDOWN;
        }


        // should not happen
        std::cerr << "PopStruct0::get_global_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_global_attribute_double(std::string name, double value)  {

        // Global parameter tauUP
        if ( name.compare("tauUP") == 0 ) {
            tauUP = value;
            return;
        }

        // Global parameter tauDOWN
        if ( name.compare("tauDOWN") == 0 ) {
            tauDOWN = value;
            return;
        }


        std::cerr << "PopStruct0::set_global_attribute_double: " << name << " not found" << std::endl;
    }



    // Method called to initialize the data structures
    void init_population() {
    #ifdef _DEBUG
        std::cout << "PopStruct0::init_population() - this = " << this << std::endl;
    #endif
        _active = true;

        // Local parameter rates
        rates = std::vector<double>(size, 0.0);

        // Global parameter tauUP
        tauUP = 0.0;

        // Global parameter tauDOWN
        tauDOWN = 0.0;

        // Local variable p
        p = std::vector<double>(size, 0.0);

        // Local variable act
        act = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);

        rand_0 = std::vector<double>(size, 0.0);


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

        dist_rand_0 = std::uniform_real_distribution< double >(0.0, 1.0);

    }

    // Method to draw new random numbers
    void update_rng() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct0::update_rng()" << std::endl;
#endif

        if (_active){

            for(int i = 0; i < size; i++) {

                rand_0[i] = dist_rand_0(rng[0]);

            }
        }

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

                // p       = Uniform(0.0, 1.0) * 1000.0 / dt
                p[i] = 1000.0*rand_0[i]/dt;


                // dact/dt = if (rates - act) > 0: (rates - act) / tauUP else: (rates - act) / tauDOWN
                double _act = (-act[i] + rates[i] > 0 ? (-act[i] + rates[i])/tauUP : (-act[i] + rates[i])/tauDOWN);

                // dact/dt = if (rates - act) > 0: (rates - act) / tauUP else: (rates - act) / tauDOWN
                act[i] += dt*_act ;


            }
        } // active

    }

    void spike_gather() {

        if( _active ) {
            for (int i = 0; i < size; i++) {


                // Spike emission
                if(p[i] <= act[i]){ // Condition is met
                    // Reset variables

                    p[i] = 0.0;

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
        size_in_bytes += sizeof(double) * rates.capacity();	// rates
        size_in_bytes += sizeof(double);	// tauUP
        size_in_bytes += sizeof(double);	// tauDOWN
        // Variables
        size_in_bytes += sizeof(double) * p.capacity();	// p
        size_in_bytes += sizeof(double) * act.capacity();	// act
        size_in_bytes += sizeof(double) * r.capacity();	// r

        return size_in_bytes;
    }

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct0::clear() - this = " << this << std::endl;
#endif
        // Variables
        p.clear();
        p.shrink_to_fit();
        act.clear();
        act.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();

    }
};

