// 候选局部搜索。一个逻辑组处理一条路径；L=1为标量，L=32为warp协作。
// 并行检查后取最小候选序号，不把 first-improvement 改成 best-improvement。
__device__ __forceinline__ bool ls_accept(float removed, float added) {
    float scale=removed+added;
    return added-removed < -(1.0e-7f+9.5367431640625e-7f*scale);
}
__device__ int ls_two(const uint16_t* t,const int* pos,const float* d,const uint16_t* nn,
                     int n,int stride,int limit,int city,int ordinal,int &i,int &j) {
    int candidate=nn[city*stride+ordinal%limit];
    i=pos[city]; j=pos[candidate];
    int u,v;
    if(ordinal<limit) { u=t[(i+1)%n]; v=t[(j+1)%n]; }
    else { i=(i+n-1)%n; j=(j+n-1)%n; u=t[i]; v=t[j]; }
    if(candidate==city || candidate==u || v==city || d[city*n+candidate]>=d[city*n+u]) return 0;
    float removed=d[city*n+u]+d[candidate*n+v];
    float added=d[city*n+candidate]+d[u*n+v];
    if(i>j) {int tmp=i;i=j;j=tmp;}
    return ls_accept(removed,added);
}
__device__ int ls_three(const uint16_t* t,const int* pos,const float* d,const uint16_t* nn,
                       int n,int stride,int limit,int city,int ordinal,
                       int &i,int &j,int &k,int &pattern) {
    pattern=ordinal%4;
    i=pos[city]; j=pos[nn[city*stride+ordinal/(4*limit)]];
    k=pos[nn[t[(i+1)%n]*stride+(ordinal/4)%limit]];
    if(i>j){int tmp=i;i=j;j=tmp;} if(j>k){int tmp=j;j=k;k=tmp;}
    if(i>j){int tmp=i;i=j;j=tmp;}
    if(j-i<2 || k-j<2 || (i==0 && k==n-1)) return 0;
    int a=t[i],b=t[i+1],c=t[j],dd=t[j+1],e=t[k],f=t[(k+1)%n];
    float removed=(d[a*n+b]+d[c*n+dd])+d[e*n+f];
    float x,y,z;
    if(pattern==0){x=d[a*n+c];y=d[b*n+e];z=d[dd*n+f];}
    else if(pattern==1){x=d[a*n+dd];y=d[e*n+b];z=d[c*n+f];}
    else if(pattern==2){x=d[a*n+e];y=d[dd*n+b];z=d[c*n+f];}
    else {x=d[a*n+dd];y=d[e*n+c];z=d[b*n+f];}
    return ls_accept(removed,(x+y)+z);
}
__device__ __forceinline__ int ls_source(int index,int i,int j,int k,int pattern) {
    if(index<=i || index>k) return index;
    if(pattern==-1) return i+1+j-index;
    int offset=index-i-1,s1=j-i,s2=k-j;
    if(pattern==0) return offset<s1 ? j-offset : k-(offset-s1);
    if(offset<s2) return pattern==2 ? k-offset : j+1+offset;
    offset-=s2; return pattern==3 ? j-offset : i+1+offset;
}
template<int L> __device__ __forceinline__ void ls_sync() {
    if constexpr(L==32) __syncwarp();
}
template<int L> __device__ __forceinline__ int ls_min(int value) {
    if constexpr(L==32) {
        for(int shift=16;shift;shift/=2) value=min(value,__shfl_down_sync(0xffffffff,value,shift));
        value=__shfl_sync(0xffffffff,value,0);
    }
    return value;
}
template<int L> __device__ void ls_apply(uint16_t* t,int* pos,uint8_t* dlb,uint16_t* tmp,
                                        int n,int lane,int i,int j,int k,int pattern) {
    if(lane==0) {
        dlb[t[i]]=dlb[t[(i+1)%n]]=0;
        dlb[t[j]]=dlb[t[(j+1)%n]]=0;
        dlb[t[k]]=dlb[t[(k+1)%n]]=0;
    }
    for(int index=i+1+lane;index<=k;index+=L) tmp[index]=t[ls_source(index,i,j,k,pattern)];
    ls_sync<L>();
    for(int index=i+1+lane;index<=k;index+=L) {t[index]=tmp[index];pos[tmp[index]]=index;}
    ls_sync<L>();
    if(lane==0) t[n]=t[0];
    ls_sync<L>();
}
// 排列不依赖GP程序。按(instance,ant)仅生成一次，各程序共享只读顺序。
extern "C" __global__ void make_orders(int* orders,const uint64_t* keys,int batch,int ants,int n,
                                      int iteration,uint64_t seed) {
    int logical=blockIdx.x*blockDim.x+threadIdx.x;
    if(logical>=batch*ants) return;
    int b=logical/ants,ant=logical%ants;
    int* order=orders+(size_t)logical*n;
    for(int x=0;x<n;++x)order[x]=x;
    for(int x=0;x<n-1;++x) {
        int other=x+min((int)(counter_uniform(seed,keys[b],iteration,ant,x,4)*(float)(n-x)),n-x-1);
        int value=order[x];order[x]=order[other];order[other]=value;
    }
}
extern "C" __global__ void improve_tours(
    const float* distance,const uint16_t* nearest,const int* task_instance,const uint64_t* keys,
    int tasks,int n,int stride,int limit,int ants,int iteration,uint64_t seed,int mode,
    uint16_t* tours,float* lengths,int* positions,int* orders,uint8_t* dlbs,uint16_t* scratch,
    uint64_t* statistics) {
    constexpr int L=LS_LANES;
    int thread=blockIdx.x*blockDim.x+threadIdx.x, logical=thread/L,lane=thread%L;
    if(logical>=tasks*ants) return;  // 边界warp整组退出，不能出现半warp。
    int task=logical/ants,ant=logical%ants,b=task_instance[task];
    const float* d=distance+(size_t)b*n*n;
    const uint16_t* nn=nearest+(size_t)b*n*stride;
    uint16_t* t=tours+(size_t)logical*(n+1);
    int* pos=positions+(size_t)logical*n; const int* order=orders+((size_t)b*ants+ant)*n;
    uint8_t* dlb=dlbs+(size_t)logical*n; uint16_t* tmp=scratch+(size_t)logical*n;
    uint64_t* stats=statistics+(size_t)logical*4;
    for(int x=lane;x<n;x+=L){pos[t[x]]=x;dlb[x]=0;}
    ls_sync<L>();
    while(true) {
        bool changed=true;
        while(changed) {
            changed=false; if(lane==0) stats[3]++;
            for(int x=0;x<n;++x) {
                int city=order[x]; if(dlb[city]) continue;
                int winner=2*limit;
                for(int base=0;base<2*limit;base+=L) {
                    int ordinal=base+lane,i,j;
                    int selected=ordinal<2*limit && ls_two(t,pos,d,nn,n,stride,limit,city,ordinal,i,j)
                                 ? ordinal : 2*limit;
                    winner=ls_min<L>(selected); if(winner<2*limit) break;
                }
                if(lane==0) stats[2]+=winner<2*limit ? winner+1 : 2*limit;
                if(winner<2*limit) {
                    int i,j; ls_two(t,pos,d,nn,n,stride,limit,city,winner,i,j);
                    // 所有lane先读取旧位置，再允许任何lane更改路径。
                    ls_sync<L>();
                    ls_apply<L>(t,pos,dlb,tmp,n,lane,i,j,j,-1);
                    changed=true; if(lane==0) stats[0]++;
                } else {if(lane==0) dlb[city]=1;ls_sync<L>();}
            }
        }
        if(mode==1) break;
        bool moved=false;
        for(int x=0;x<n;++x) {
            int city=order[x],count=4*limit*limit,winner=count;
            for(int base=0;base<count;base+=L) {
                int ordinal=base+lane,i,j,k,pattern;
                int selected=ordinal<count && ls_three(t,pos,d,nn,n,stride,limit,city,ordinal,i,j,k,pattern)
                             ? ordinal : count;
                winner=ls_min<L>(selected); if(winner<count) break;
            }
            if(lane==0) stats[2]+=winner<count ? winner+1 : count;
            if(winner<count) {
                int i,j,k,pattern;ls_three(t,pos,d,nn,n,stride,limit,city,winner,i,j,k,pattern);
                ls_sync<L>(); ls_apply<L>(t,pos,dlb,tmp,n,lane,i,j,k,pattern);
                if(lane==0) stats[1]++;
                moved=true; break;
            }
        }
        if(!moved) break;
        for(int x=lane;x<n;x+=L) dlb[x]=0;
        ls_sync<L>();
    }
    // 与构造路径相同的升序FP32累加，不改变更新阶段的长度合同。
    if(lane==0) {float length=0.0f;for(int x=0;x<n;++x)length+=d[t[x]*n+t[x+1]];lengths[logical]=length;}
}
