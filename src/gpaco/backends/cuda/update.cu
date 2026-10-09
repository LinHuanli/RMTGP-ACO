// 固定 ACO 更新。无信息素 GP 树、局部搜索或在线 FP64 审计。
// AS 按蚂蚁编号累加，消除 atomicAdd 的不确定累加次序。
extern "C" __global__ void initialize(const float* tau0, const float* low,
    const float* high, const int* instances, int tasks, int n,
    float* tau, float* state, int* counts, uint64_t* diagnostics) {
    int task=blockIdx.x, tid=threadIdx.x;
    if(task>=tasks) return;
    for(int e=tid;e<n*n;e+=blockDim.x)
        tau[(size_t)task*n*n+e]=(e/n==e%n)?0.0f:tau0[instances[task]];
    if(tid<8) diagnostics[(size_t)task*8+tid]=0;
    if(tid==0) {
        state[task*4]=CUDART_INF_F; state[task*4+1]=CUDART_INF_F;
        state[task*4+2]=low[instances[task]]; state[task*4+3]=high[instances[task]];
        counts[task]=0; counts[tasks+task]=0;
    }
}

extern "C" __global__ void update(const uint16_t* nearest_all, const int* instances,
    int tasks, int n, int k, int ants, int iteration, float rho, int period,
    float p_best, int branch_period, float branch_lambda, float branch_threshold,
    int restart_stagnation, float* tau_all, const uint16_t* tours_all,
    const float* lengths_all, float* deposits_all, uint16_t* best_all,
    uint16_t* restart_all, float* states, int* counts, uint64_t* diagnostics) {
    int task=blockIdx.x, tid=threadIdx.x;
    if(task>=tasks) return;
    float* tau=tau_all+(size_t)task*n*n;
    float* deposits=deposits_all+(size_t)task*n*n;
    float* state=states+task*4;
    const uint16_t* tours=tours_all+(size_t)task*ants*(n+1);
    const float* lengths=lengths_all+(size_t)task*ants;
    uint16_t* best=best_all+(size_t)task*(n+1);
    uint16_t* restart=restart_all+(size_t)task*(n+1);
    __shared__ int winner, copy_best, copy_restart, reset;
    if(tid==0) {
        winner=0;
        for(int a=1;a<ants;++a) if(lengths[a]<lengths[winner]) winner=a;
        copy_best=lengths[winner]<state[0];
        copy_restart=lengths[winner]<state[1]; reset=0;
        if(copy_best) {
            state[0]=lengths[winner]; counts[task]=0;
            #if RMTGP_VARIANT==2
            state[3]=1.0f/(rho*state[0]);
            float px=expf(logf(p_best)/(float)n);
            state[2]=state[3]*(1.0f-px)/(px*(float)((k+1)/2));
            #endif
        } else ++counts[task];
        if(copy_restart) { state[1]=lengths[winner]; counts[tasks+task]=iteration; }
    }
    __syncthreads();
    for(int i=tid;i<=n;i+=blockDim.x) {
        if(copy_best) best[i]=tours[winner*(n+1)+i];
        if(copy_restart) restart[i]=tours[winner*(n+1)+i];
    }
    __syncthreads();
    #if RMTGP_VARIANT==1
    for(int i=tid;i<n;i+=blockDim.x) {
        int u=best[i],v=best[i+1];
        float value=(1.0f-rho)*tau[u*n+v]+rho*(1.0f/state[0]);
        tau[u*n+v]=value; tau[v*n+u]=value;
    }
    #else
    for(int e=tid;e<n*n;e+=blockDim.x) deposits[e]=0.0f;
    __syncthreads();
    #if RMTGP_VARIANT==0
    int sources=ants;
    #else
    int sources=1;
    #endif
    for(int a=0;a<sources;++a) {
        #if RMTGP_VARIANT==0
        const uint16_t* source=tours+a*(n+1); float length=lengths[a];
        #else
        const uint16_t* source=(iteration%period==0)?restart:tours+winner*(n+1);
        float length=(iteration%period==0)?state[1]:lengths[winner];
        #endif
        for(int i=tid;i<n;i+=blockDim.x) {
            int u=source[i],v=source[i+1]; float value=1.0f/length;
            deposits[u*n+v]+=value; deposits[v*n+u]+=value;
        }
        __syncthreads();
    }
    for(int e=tid;e<n*n;e+=blockDim.x) {
        if(e/n==e%n) { tau[e]=0.0f; continue; }
        float raw=(1.0f-rho)*tau[e]+deposits[e];
        #if RMTGP_VARIANT==2
        float value=fminf(fmaxf(raw,state[2]),state[3]);
        tau[e]=value;
        if(value!=raw) atomicAdd(diagnostics+(size_t)task*8+2,1ULL);
        #else
        tau[e]=raw;
        #endif
    }
    #endif
    __syncthreads();
    #if RMTGP_VARIANT==2
    if(tid==0 && iteration%branch_period==0 && iteration-counts[tasks+task]>restart_stagnation) {
        const uint16_t* nearest=nearest_all+(size_t)instances[task]*n*k;
        int branches=0;
        for(int u=0;u<n;++u) {
            float lo=CUDART_INF_F, hi=-CUDART_INF_F;
            for(int j=0;j<k;++j) { float v=tau[u*n+nearest[u*k+j]]; lo=fminf(lo,v); hi=fmaxf(hi,v); }
            float cutoff=lo+branch_lambda*(hi-lo);
            for(int j=0;j<k;++j) branches+=tau[u*n+nearest[u*k+j]]>cutoff;
        }
        if((float)branches/(2.0f*(float)n)<branch_threshold) {
            reset=1; state[1]=CUDART_INF_F; counts[tasks+task]=iteration;
            ++diagnostics[(size_t)task*8+3];
        }
    }
    __syncthreads();
    if(reset) for(int e=tid;e<n*n;e+=blockDim.x) tau[e]=(e/n==e%n)?0.0f:state[3];
    #endif
}
