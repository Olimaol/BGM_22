#pragma once
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "sparse_matrix.hpp"

#include "pop0.hpp"
#include "pop1.hpp"



extern PopStruct0 pop0;
extern PopStruct1 pop1;

extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj0: Cortex-Go -> STR_D1 with target ampa
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0 : LILInvMatrix<int, int> {
    ProjStruct0() : LILInvMatrix<int, int>( 10, 10) {
    }


    void init_from_lil( std::vector<int> &row_indices,
                        std::vector< std::vector<int> > &column_indices,
                        std::vector< std::vector<double> > &values,
                        std::vector< std::vector<int> > &delays) {
        static_cast<LILInvMatrix<int, int>*>(this)->init_matrix_from_lil(row_indices, column_indices);

        w = values[0][0];


        init_attributes();
    #ifdef _DEBUG_CONN
        static_cast<LILInvMatrix<int, int>*>(this)->print_data_representation();
    #endif
    }





    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;





    // Local parameter max_trans
    std::vector< std::vector<double > > max_trans;

    // Local parameter mod_factor
    std::vector< std::vector<double > > mod_factor;

    // Global parameter w
    double  w ;




    // Method called to allocate/initialize the variables
    void init_attributes() {

        // Local parameter max_trans
        max_trans = init_matrix_variable<double>(static_cast<double>(0.0));

        // Local parameter mod_factor
        mod_factor = init_matrix_variable<double>(static_cast<double>(0.0));



    }

    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::init_projection() - this = " << this << std::endl;
    #endif

        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

        init_attributes();



    }

    // Spiking networks: reset the ring buffer when non-uniform
    void reset_ring_buffer() {

    }

    // Spiking networks: update maximum delay when non-uniform
    void update_max_delay(int d){

    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::compute_psp()" << std::endl;
    #endif
int nb_post; double sum;

        // Event-based summation
        if (_transmission && pop1._active){


            // Iterate over all incoming spikes (possibly delayed constantly)
            for(int _idx_j = 0; _idx_j < pop0.spiked.size(); _idx_j++){
                // Rank of the presynaptic neuron
                int rk_j = pop0.spiked[_idx_j];
                // Find the presynaptic neuron in the inverse connectivity matrix
                auto inv_post_ptr = inv_pre_rank.find(rk_j);
                if (inv_post_ptr == inv_pre_rank.end())
                    continue;
                // List of postsynaptic neurons receiving spikes from that neuron
                std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;
                // Number of post neurons
                int nb_post = inv_post.size();

                // Iterate over connected post neurons
                for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                    // Retrieve the correct indices
                    int i = inv_post[_idx_i].first;
                    int j = inv_post[_idx_i].second;

                    // Event-driven integration

                    // Update conductance

                    pop1.g_ampa[post_rank[i]] +=  mod_factor[i][j]*w;

                    if (pop1.g_ampa[post_rank[i]] > max_trans[i][j])
                        pop1.g_ampa[post_rank[i]] = max_trans[i][j];

                    // Synaptic plasticity: pre-events

                }
            }
        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::update_synapse()" << std::endl;
    #endif


    }

    // Post-synaptic events
    void post_event() {


    }

    // Variable/Parameter access methods

    std::vector<std::vector<double>> get_local_attribute_all_double(std::string name) {

        if ( name.compare("max_trans") == 0 ) {

            return get_matrix_variable_all<double>(max_trans);
        }

        if ( name.compare("mod_factor") == 0 ) {

            return get_matrix_variable_all<double>(mod_factor);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<double> get_local_attribute_row_double(std::string name, int rk_post) {

        if ( name.compare("max_trans") == 0 ) {

            return get_matrix_variable_row<double>(max_trans, rk_post);
        }

        if ( name.compare("mod_factor") == 0 ) {

            return get_matrix_variable_row<double>(mod_factor, rk_post);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_row_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk_post, int rk_pre) {

        if ( name.compare("max_trans") == 0 ) {

            return get_matrix_variable<double>(max_trans, rk_post, rk_pre);
        }

        if ( name.compare("mod_factor") == 0 ) {

            return get_matrix_variable<double>(mod_factor, rk_post, rk_pre);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all_double(std::string name, std::vector<std::vector<double>> value) {

        if ( name.compare("max_trans") == 0 ) {
            update_matrix_variable_all<double>(max_trans, value);

            return;
        }

        if ( name.compare("mod_factor") == 0 ) {
            update_matrix_variable_all<double>(mod_factor, value);

            return;
        }

    }

    void set_local_attribute_row_double(std::string name, int rk_post, std::vector<double> value) {

        if ( name.compare("max_trans") == 0 ) {
            update_matrix_variable_row<double>(max_trans, rk_post, value);

            return;
        }

        if ( name.compare("mod_factor") == 0 ) {
            update_matrix_variable_row<double>(mod_factor, rk_post, value);

            return;
        }

    }

    void set_local_attribute_double(std::string name, int rk_post, int rk_pre, double value) {

        if ( name.compare("max_trans") == 0 ) {
            update_matrix_variable<double>(max_trans, rk_post, rk_pre, value);

            return;
        }

        if ( name.compare("mod_factor") == 0 ) {
            update_matrix_variable<double>(mod_factor, rk_post, rk_pre, value);

            return;
        }

    }

    double get_global_attribute_double(std::string name) {

        if ( name.compare("w") == 0 ) {
            return w;
        }


        // should not happen
        std::cerr << "ProjStruct0::get_global_attribute_double: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_global_attribute_double(std::string name, double value) {

        if ( name.compare("w") == 0 ) {
            w = value;

            return;
        }

    }


    // Access additional


    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // connectivity
        size_in_bytes += static_cast<LILInvMatrix<int, int>*>(this)->size_in_bytes();

        // Local parameter max_trans
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * max_trans.capacity();
        for(auto it = max_trans.cbegin(); it != max_trans.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Local parameter mod_factor
        size_in_bytes += sizeof(std::vector<std::vector<double>>);
        size_in_bytes += sizeof(std::vector<double>) * mod_factor.capacity();
        for(auto it = mod_factor.cbegin(); it != mod_factor.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        // Global parameter w
        size_in_bytes += sizeof(double);

        return size_in_bytes;
    }

    // Structural plasticity



    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::clear() - this = " << this << std::endl;
    #endif

        // Connectivity
        static_cast<LILInvMatrix<int, int>*>(this)->clear();

        // max_trans
        for (auto it = max_trans.begin(); it != max_trans.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        max_trans.clear();
        max_trans.shrink_to_fit();

        // mod_factor
        for (auto it = mod_factor.begin(); it != mod_factor.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        mod_factor.clear();
        mod_factor.shrink_to_fit();

    }
};

