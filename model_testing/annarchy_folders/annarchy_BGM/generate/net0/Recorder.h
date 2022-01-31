#pragma once
extern long int t;

int addRecorder(class Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(class Monitor* recorder);

/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, int period_offset, long int offset) {
        this->ranks = ranks;
        this->period_ = period;
        this->period_offset_ = period_offset;
        this->offset_ = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    ~Monitor() = default;

    virtual void record() = 0;
    virtual void record_targets() = 0;
    virtual long int size_in_bytes() = 0;
    virtual void clear() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period_;
    int period_offset_;
    long int offset_;
};

class PopRecorder0 : public Monitor
{
protected:
    PopRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << this << ") instantiated." << std::endl;
    #endif

        this->p = std::vector< std::vector< double > >();
        this->record_p = false; 
        this->act = std::vector< std::vector< double > >();
        this->record_act = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop0.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; 

    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder0* get_instance(int id) {
        return static_cast<PopRecorder0*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder0::record()" << std::endl;
    #endif

        if(this->record_p && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->p.push_back(pop0.p);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.p[this->ranks[i]]);
                }
                this->p.push_back(tmp);
            }
        }
        if(this->record_act && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->act.push_back(pop0.act);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.act[this->ranks[i]]);
                }
                this->act.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop0.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop0.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop0.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop0.spiked[i])!=this->ranks.end() ){
                        this->spike[pop0.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable p
        size_in_bytes += sizeof(std::vector<double>) * p.capacity();
        for(auto it=p.begin(); it!= p.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable act
        size_in_bytes += sizeof(std::vector<double>) * act.capacity();
        for(auto it=act.begin(); it!= act.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // record spike events
        size_in_bytes += sizeof(spike);
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            size_in_bytes += sizeof(int); // key
            size_in_bytes += sizeof(long int) * (it->second).capacity(); // value
        }
                
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "Delete instance of PopRecorder0 ( " << this << " ) " << std::endl;
    #endif

        for(auto it = this->p.begin(); it != this->p.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->p.clear();
    
        for(auto it = this->act.begin(); it != this->act.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->act.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    

        removeRecorder(this);
    }



    // Local variable p
    std::vector< std::vector< double > > p ;
    bool record_p ; 
    // Local variable act
    std::vector< std::vector< double > > act ;
    bool record_act ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
    }

};

class PopRecorder1 : public Monitor
{
protected:
    PopRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_ampa = std::vector< std::vector< double > >();
        this->record_g_ampa = false; 
        this->g_gaba = std::vector< std::vector< double > >();
        this->record_g_gaba = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->u = std::vector< std::vector< double > >();
        this->record_u = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop1.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; 

    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder1(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder1* get_instance(int id) {
        return static_cast<PopRecorder1*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder1::record()" << std::endl;
    #endif

        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop1.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_u && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->u.push_back(pop1.u);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.u[this->ranks[i]]);
                }
                this->u.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop1.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop1.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop1.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop1.spiked[i])!=this->ranks.end() ){
                        this->spike[pop1.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_ampa.push_back(pop1.g_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.g_ampa[this->ranks[i]]);
                }
                this->g_ampa.push_back(tmp);
            }
        }
        if(this->record_g_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_gaba.push_back(pop1.g_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.g_gaba[this->ranks[i]]);
                }
                this->g_gaba.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable v
        size_in_bytes += sizeof(std::vector<double>) * v.capacity();
        for(auto it=v.begin(); it!= v.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable u
        size_in_bytes += sizeof(std::vector<double>) * u.capacity();
        for(auto it=u.begin(); it!= u.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // record spike events
        size_in_bytes += sizeof(spike);
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            size_in_bytes += sizeof(int); // key
            size_in_bytes += sizeof(long int) * (it->second).capacity(); // value
        }
                
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "Delete instance of PopRecorder1 ( " << this << " ) " << std::endl;
    #endif

        for(auto it = this->v.begin(); it != this->v.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->v.clear();
    
        for(auto it = this->u.begin(); it != this->u.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->u.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    

        removeRecorder(this);
    }



    // Local variable g_ampa
    std::vector< std::vector< double > > g_ampa ;
    bool record_g_ampa ; 
    // Local variable g_gaba
    std::vector< std::vector< double > > g_gaba ;
    bool record_g_gaba ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable u
    std::vector< std::vector< double > > u ;
    bool record_u ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
    }

};

class ProjRecorder0 : public Monitor
{
protected:
    ProjRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj0.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder0* get_instance(int id) {
        return static_cast<ProjRecorder0*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor0::clear(): not implemented for openMP paradigm." << std::endl;
    }


};

