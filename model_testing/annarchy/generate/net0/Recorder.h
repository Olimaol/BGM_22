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

        this->p = std::vector< std::vector< double > >();
        this->record_p = false; 
        this->act = std::vector< std::vector< double > >();
        this->record_act = false; 
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

        if(this->record_p && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->p.push_back(pop1.p);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.p[this->ranks[i]]);
                }
                this->p.push_back(tmp);
            }
        }
        if(this->record_act && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->act.push_back(pop1.act);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.act[this->ranks[i]]);
                }
                this->act.push_back(tmp);
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
        std::cout << "Delete instance of PopRecorder1 ( " << this << " ) " << std::endl;
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

class PopRecorder2 : public Monitor
{
protected:
    PopRecorder2(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder2 (" << this << ") instantiated." << std::endl;
    #endif

        this->p = std::vector< std::vector< double > >();
        this->record_p = false; 
        this->act = std::vector< std::vector< double > >();
        this->record_act = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop2.size; i++) {
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
        auto new_recorder = new PopRecorder2(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder2 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder2* get_instance(int id) {
        return static_cast<PopRecorder2*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder2::record()" << std::endl;
    #endif

        if(this->record_p && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->p.push_back(pop2.p);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.p[this->ranks[i]]);
                }
                this->p.push_back(tmp);
            }
        }
        if(this->record_act && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->act.push_back(pop2.act);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.act[this->ranks[i]]);
                }
                this->act.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop2.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop2.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop2.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop2.spiked[i])!=this->ranks.end() ){
                        this->spike[pop2.spiked[i]].push_back(t);
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
        std::cout << "Delete instance of PopRecorder2 ( " << this << " ) " << std::endl;
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

class PopRecorder3 : public Monitor
{
protected:
    PopRecorder3(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder3 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_ampa = std::vector< std::vector< double > >();
        this->record_g_ampa = false; 
        this->g_gaba = std::vector< std::vector< double > >();
        this->record_g_gaba = false; 
        this->I_ampa = std::vector< std::vector< double > >();
        this->record_I_ampa = false; 
        this->I_gaba = std::vector< std::vector< double > >();
        this->record_I_gaba = false; 
        this->I = std::vector< std::vector< double > >();
        this->record_I = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->u = std::vector< std::vector< double > >();
        this->record_u = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop3.size; i++) {
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
        auto new_recorder = new PopRecorder3(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder3 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder3* get_instance(int id) {
        return static_cast<PopRecorder3*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder3::record()" << std::endl;
    #endif

        if(this->record_I_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I_ampa.push_back(pop3.I_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.I_ampa[this->ranks[i]]);
                }
                this->I_ampa.push_back(tmp);
            }
        }
        if(this->record_I_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I_gaba.push_back(pop3.I_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.I_gaba[this->ranks[i]]);
                }
                this->I_gaba.push_back(tmp);
            }
        }
        if(this->record_I && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I.push_back(pop3.I);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.I[this->ranks[i]]);
                }
                this->I.push_back(tmp);
            }
        }
        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop3.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_u && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->u.push_back(pop3.u);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.u[this->ranks[i]]);
                }
                this->u.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop3.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop3.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop3.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop3.spiked[i])!=this->ranks.end() ){
                        this->spike[pop3.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_ampa.push_back(pop3.g_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.g_ampa[this->ranks[i]]);
                }
                this->g_ampa.push_back(tmp);
            }
        }
        if(this->record_g_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_gaba.push_back(pop3.g_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.g_gaba[this->ranks[i]]);
                }
                this->g_gaba.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable I_ampa
        size_in_bytes += sizeof(std::vector<double>) * I_ampa.capacity();
        for(auto it=I_ampa.begin(); it!= I_ampa.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable I_gaba
        size_in_bytes += sizeof(std::vector<double>) * I_gaba.capacity();
        for(auto it=I_gaba.begin(); it!= I_gaba.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable I
        size_in_bytes += sizeof(std::vector<double>) * I.capacity();
        for(auto it=I.begin(); it!= I.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
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
        std::cout << "Delete instance of PopRecorder3 ( " << this << " ) " << std::endl;
    #endif

        for(auto it = this->I_ampa.begin(); it != this->I_ampa.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I_ampa.clear();
    
        for(auto it = this->I_gaba.begin(); it != this->I_gaba.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I_gaba.clear();
    
        for(auto it = this->I.begin(); it != this->I.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I.clear();
    
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
    // Local variable I_ampa
    std::vector< std::vector< double > > I_ampa ;
    bool record_I_ampa ; 
    // Local variable I_gaba
    std::vector< std::vector< double > > I_gaba ;
    bool record_I_gaba ; 
    // Local variable I
    std::vector< std::vector< double > > I ;
    bool record_I ; 
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

class PopRecorder4 : public Monitor
{
protected:
    PopRecorder4(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder4 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_ampa = std::vector< std::vector< double > >();
        this->record_g_ampa = false; 
        this->g_gaba = std::vector< std::vector< double > >();
        this->record_g_gaba = false; 
        this->I_ampa = std::vector< std::vector< double > >();
        this->record_I_ampa = false; 
        this->I_gaba = std::vector< std::vector< double > >();
        this->record_I_gaba = false; 
        this->I = std::vector< std::vector< double > >();
        this->record_I = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->u = std::vector< std::vector< double > >();
        this->record_u = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop4.size; i++) {
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
        auto new_recorder = new PopRecorder4(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder4 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder4* get_instance(int id) {
        return static_cast<PopRecorder4*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder4::record()" << std::endl;
    #endif

        if(this->record_I_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I_ampa.push_back(pop4.I_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.I_ampa[this->ranks[i]]);
                }
                this->I_ampa.push_back(tmp);
            }
        }
        if(this->record_I_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I_gaba.push_back(pop4.I_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.I_gaba[this->ranks[i]]);
                }
                this->I_gaba.push_back(tmp);
            }
        }
        if(this->record_I && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I.push_back(pop4.I);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.I[this->ranks[i]]);
                }
                this->I.push_back(tmp);
            }
        }
        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop4.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_u && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->u.push_back(pop4.u);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.u[this->ranks[i]]);
                }
                this->u.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop4.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop4.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop4.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop4.spiked[i])!=this->ranks.end() ){
                        this->spike[pop4.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_ampa.push_back(pop4.g_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.g_ampa[this->ranks[i]]);
                }
                this->g_ampa.push_back(tmp);
            }
        }
        if(this->record_g_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_gaba.push_back(pop4.g_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.g_gaba[this->ranks[i]]);
                }
                this->g_gaba.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable I_ampa
        size_in_bytes += sizeof(std::vector<double>) * I_ampa.capacity();
        for(auto it=I_ampa.begin(); it!= I_ampa.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable I_gaba
        size_in_bytes += sizeof(std::vector<double>) * I_gaba.capacity();
        for(auto it=I_gaba.begin(); it!= I_gaba.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable I
        size_in_bytes += sizeof(std::vector<double>) * I.capacity();
        for(auto it=I.begin(); it!= I.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
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
        std::cout << "Delete instance of PopRecorder4 ( " << this << " ) " << std::endl;
    #endif

        for(auto it = this->I_ampa.begin(); it != this->I_ampa.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I_ampa.clear();
    
        for(auto it = this->I_gaba.begin(); it != this->I_gaba.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I_gaba.clear();
    
        for(auto it = this->I.begin(); it != this->I.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I.clear();
    
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
    // Local variable I_ampa
    std::vector< std::vector< double > > I_ampa ;
    bool record_I_ampa ; 
    // Local variable I_gaba
    std::vector< std::vector< double > > I_gaba ;
    bool record_I_gaba ; 
    // Local variable I
    std::vector< std::vector< double > > I ;
    bool record_I ; 
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

class PopRecorder5 : public Monitor
{
protected:
    PopRecorder5(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder5 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_ampa = std::vector< std::vector< double > >();
        this->record_g_ampa = false; 
        this->g_gaba = std::vector< std::vector< double > >();
        this->record_g_gaba = false; 
        this->I_ampa = std::vector< std::vector< double > >();
        this->record_I_ampa = false; 
        this->I_gaba = std::vector< std::vector< double > >();
        this->record_I_gaba = false; 
        this->I = std::vector< std::vector< double > >();
        this->record_I = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->U_v = std::vector< std::vector< double > >();
        this->record_U_v = false; 
        this->u = std::vector< std::vector< double > >();
        this->record_u = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop5.size; i++) {
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
        auto new_recorder = new PopRecorder5(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder5 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder5* get_instance(int id) {
        return static_cast<PopRecorder5*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder5::record()" << std::endl;
    #endif

        if(this->record_I_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I_ampa.push_back(pop5.I_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.I_ampa[this->ranks[i]]);
                }
                this->I_ampa.push_back(tmp);
            }
        }
        if(this->record_I_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I_gaba.push_back(pop5.I_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.I_gaba[this->ranks[i]]);
                }
                this->I_gaba.push_back(tmp);
            }
        }
        if(this->record_I && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I.push_back(pop5.I);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.I[this->ranks[i]]);
                }
                this->I.push_back(tmp);
            }
        }
        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop5.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_U_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->U_v.push_back(pop5.U_v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.U_v[this->ranks[i]]);
                }
                this->U_v.push_back(tmp);
            }
        }
        if(this->record_u && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->u.push_back(pop5.u);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.u[this->ranks[i]]);
                }
                this->u.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop5.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop5.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop5.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop5.spiked[i])!=this->ranks.end() ){
                        this->spike[pop5.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_ampa.push_back(pop5.g_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.g_ampa[this->ranks[i]]);
                }
                this->g_ampa.push_back(tmp);
            }
        }
        if(this->record_g_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_gaba.push_back(pop5.g_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.g_gaba[this->ranks[i]]);
                }
                this->g_gaba.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable I_ampa
        size_in_bytes += sizeof(std::vector<double>) * I_ampa.capacity();
        for(auto it=I_ampa.begin(); it!= I_ampa.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable I_gaba
        size_in_bytes += sizeof(std::vector<double>) * I_gaba.capacity();
        for(auto it=I_gaba.begin(); it!= I_gaba.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable I
        size_in_bytes += sizeof(std::vector<double>) * I.capacity();
        for(auto it=I.begin(); it!= I.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable v
        size_in_bytes += sizeof(std::vector<double>) * v.capacity();
        for(auto it=v.begin(); it!= v.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable U_v
        size_in_bytes += sizeof(std::vector<double>) * U_v.capacity();
        for(auto it=U_v.begin(); it!= U_v.end(); it++) {
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
        std::cout << "Delete instance of PopRecorder5 ( " << this << " ) " << std::endl;
    #endif

        for(auto it = this->I_ampa.begin(); it != this->I_ampa.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I_ampa.clear();
    
        for(auto it = this->I_gaba.begin(); it != this->I_gaba.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I_gaba.clear();
    
        for(auto it = this->I.begin(); it != this->I.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I.clear();
    
        for(auto it = this->v.begin(); it != this->v.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->v.clear();
    
        for(auto it = this->U_v.begin(); it != this->U_v.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->U_v.clear();
    
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
    // Local variable I_ampa
    std::vector< std::vector< double > > I_ampa ;
    bool record_I_ampa ; 
    // Local variable I_gaba
    std::vector< std::vector< double > > I_gaba ;
    bool record_I_gaba ; 
    // Local variable I
    std::vector< std::vector< double > > I ;
    bool record_I ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable U_v
    std::vector< std::vector< double > > U_v ;
    bool record_U_v ; 
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

class PopRecorder6 : public Monitor
{
protected:
    PopRecorder6(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder6 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_ampa = std::vector< std::vector< double > >();
        this->record_g_ampa = false; 
        this->g_gaba = std::vector< std::vector< double > >();
        this->record_g_gaba = false; 
        this->I_ampa = std::vector< std::vector< double > >();
        this->record_I_ampa = false; 
        this->I_gaba = std::vector< std::vector< double > >();
        this->record_I_gaba = false; 
        this->I = std::vector< std::vector< double > >();
        this->record_I = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->u = std::vector< std::vector< double > >();
        this->record_u = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop6.size; i++) {
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
        auto new_recorder = new PopRecorder6(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder6 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder6* get_instance(int id) {
        return static_cast<PopRecorder6*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder6::record()" << std::endl;
    #endif

        if(this->record_I_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I_ampa.push_back(pop6.I_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.I_ampa[this->ranks[i]]);
                }
                this->I_ampa.push_back(tmp);
            }
        }
        if(this->record_I_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I_gaba.push_back(pop6.I_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.I_gaba[this->ranks[i]]);
                }
                this->I_gaba.push_back(tmp);
            }
        }
        if(this->record_I && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I.push_back(pop6.I);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.I[this->ranks[i]]);
                }
                this->I.push_back(tmp);
            }
        }
        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop6.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_u && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->u.push_back(pop6.u);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.u[this->ranks[i]]);
                }
                this->u.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop6.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop6.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop6.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop6.spiked[i])!=this->ranks.end() ){
                        this->spike[pop6.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_ampa && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_ampa.push_back(pop6.g_ampa);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.g_ampa[this->ranks[i]]);
                }
                this->g_ampa.push_back(tmp);
            }
        }
        if(this->record_g_gaba && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_gaba.push_back(pop6.g_gaba);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.g_gaba[this->ranks[i]]);
                }
                this->g_gaba.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable I_ampa
        size_in_bytes += sizeof(std::vector<double>) * I_ampa.capacity();
        for(auto it=I_ampa.begin(); it!= I_ampa.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable I_gaba
        size_in_bytes += sizeof(std::vector<double>) * I_gaba.capacity();
        for(auto it=I_gaba.begin(); it!= I_gaba.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable I
        size_in_bytes += sizeof(std::vector<double>) * I.capacity();
        for(auto it=I.begin(); it!= I.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
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
        std::cout << "Delete instance of PopRecorder6 ( " << this << " ) " << std::endl;
    #endif

        for(auto it = this->I_ampa.begin(); it != this->I_ampa.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I_ampa.clear();
    
        for(auto it = this->I_gaba.begin(); it != this->I_gaba.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I_gaba.clear();
    
        for(auto it = this->I.begin(); it != this->I.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I.clear();
    
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
    // Local variable I_ampa
    std::vector< std::vector< double > > I_ampa ;
    bool record_I_ampa ; 
    // Local variable I_gaba
    std::vector< std::vector< double > > I_gaba ;
    bool record_I_gaba ; 
    // Local variable I
    std::vector< std::vector< double > > I ;
    bool record_I ; 
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

