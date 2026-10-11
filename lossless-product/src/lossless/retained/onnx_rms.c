/* Experimental RMSNorm/residual fusion with ARM vectorization.
 * API declarations: Microsoft's MIT-licensed ONNX Runtime 1.20.1 header.
 * Domain lifetime is the worker process; this is not a deployment library.
 */
#include "onnxruntime_c_api.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <math.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef struct { const OrtApi* api; int parallel; int pre; } KernelState;
typedef struct { const float* x; const float* r; const float* w; float* y; float* sum; size_t rows,cols; int pre; } RowTask;
static const char* ORT_API_CALL post_serial_name(const OrtCustomOp* op){return "RmsPostSerial";}
static const char* ORT_API_CALL post_parallel_name(const OrtCustomOp* op){return "RmsPostParallel";}
static const char* ORT_API_CALL pre_serial_name(const OrtCustomOp* op){return "RmsPreSerial";}
static const char* ORT_API_CALL pre_parallel_name(const OrtCustomOp* op){return "RmsPreParallel";}
static const char* ORT_API_CALL provider(const OrtCustomOp* op){return "CPUExecutionProvider";}
static size_t ORT_API_CALL input_count(const OrtCustomOp* op){return 3;}
static size_t ORT_API_CALL output_count(const OrtCustomOp* op){return strstr(op->GetName(op),"Pre")?2:1;}
static ONNXTensorElementDataType ORT_API_CALL dtype(const OrtCustomOp* op,size_t i){return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL required(const OrtCustomOp* op,size_t i){return INPUT_OUTPUT_REQUIRED;}
static OrtMemType ORT_API_CALL memory(const OrtCustomOp* op,size_t i){return OrtMemTypeDefault;}
static int ORT_API_CALL version(const OrtCustomOp* op){return 1;}
static void ORT_API_CALL destroy(void* state){free(state);}
static OrtStatus* ORT_API_CALL create(const OrtCustomOp* op,const OrtApi* api,const OrtKernelInfo* info,void** output){
    KernelState* state=calloc(1,sizeof(KernelState));
    if(!state)return api->CreateStatus(ORT_FAIL,"kernel allocation failed");
    state->api=api;state->parallel=strstr(op->GetName(op),"Parallel")!=NULL;state->pre=strstr(op->GetName(op),"Pre")!=NULL;
    *output=state;return NULL;
}
static void rows_compute(RowTask* task,size_t start,size_t end){
    size_t cols=task->cols;
    for(size_t row=start;row<end;++row){
        const float* x=task->x+row*cols;const float* r=task->r+row*cols;
        float* y=task->y+row*cols;float* added=task->pre?task->sum+row*cols:NULL;
        size_t j=0;double total=0;
#if defined(__aarch64__)
        float64x2_t lo=vdupq_n_f64(0),hi=vdupq_n_f64(0);
        for(;j+4<=cols;j+=4){
            float32x4_t v=vld1q_f32(x+j);
            if(task->pre){v=vaddq_f32(v,vld1q_f32(r+j));vst1q_f32(added+j,v);}
            float32x4_t q=vmulq_f32(v,v);
            lo=vaddq_f64(lo,vcvt_f64_f32(vget_low_f32(q)));
            hi=vaddq_f64(hi,vcvt_f64_f32(vget_high_f32(q)));
        }
        total=vaddvq_f64(lo)+vaddvq_f64(hi);
#endif
        for(;j<cols;++j){float v=x[j];if(task->pre){v=v+r[j];added[j]=v;}float q=v*v;total+=(double)q;}
        float scale=(float)(1.0/sqrt(total/(double)cols+1e-5));
        const float* input=task->pre?added:x;j=0;
#if defined(__aarch64__)
        float32x4_t scales=vdupq_n_f32(scale);
        for(;j+4<=cols;j+=4){
            float32x4_t v=vmulq_f32(vld1q_f32(input+j),scales);v=vmulq_f32(v,vld1q_f32(task->w+j));
            if(!task->pre)v=vaddq_f32(v,vld1q_f32(r+j));vst1q_f32(y+j,v);
        }
#endif
        for(;j<cols;++j){float v=input[j]*scale;v=v*task->w[j];if(!task->pre)v=v+r[j];y[j]=v;}
    }
}
static void row_task(void* pointer,size_t chunk){RowTask* task=pointer;size_t end=(chunk+1)*8;if(end>task->rows)end=task->rows;rows_compute(task,chunk*8,end);}
static OrtStatus* ORT_API_CALL compute(void* pointer,OrtKernelContext* context){
    KernelState* state=pointer;const OrtApi* api=state->api;OrtStatus* error=NULL;
    const OrtValue* inputs[3]={0};OrtValue* output=NULL;OrtTensorTypeAndShapeInfo* info=NULL;
    int64_t dims[8],other[8];size_t rank=0,n=0,total=0,wrank=0;float* data[3]={0};float* y=NULL;float* sum=NULL;
    #define TRY(call) do{error=(call);if(error)goto cleanup;}while(0)
    #define INVALID(message) do{error=api->CreateStatus(ORT_INVALID_ARGUMENT,message);goto cleanup;}while(0)
    for(size_t i=0;i<3;++i)TRY(api->KernelContext_GetInput(context,i,&inputs[i]));
    TRY(api->GetTensorTypeAndShape(inputs[0],&info));TRY(api->GetDimensionsCount(info,&rank));
    if(rank<1||rank>8)INVALID("rank must be 1..8");
    TRY(api->GetDimensions(info,dims,rank));TRY(api->GetTensorShapeElementCount(info,&total));
    for(size_t j=0;j<rank;++j)if(dims[j]<=0)INVALID("positive dimensions required");
    api->ReleaseTensorTypeAndShapeInfo(info);info=NULL;
    TRY(api->GetTensorTypeAndShape(inputs[1],&info));TRY(api->GetDimensionsCount(info,&n));
    if(n!=rank)INVALID("residual rank mismatch");TRY(api->GetDimensions(info,other,n));
    for(size_t j=0;j<rank;++j)if(other[j]!=dims[j])INVALID("residual shape mismatch");
    api->ReleaseTensorTypeAndShapeInfo(info);info=NULL;
    TRY(api->GetTensorTypeAndShape(inputs[2],&info));TRY(api->GetDimensionsCount(info,&wrank));
    if(wrank!=1)INVALID("weight must be one dimensional");TRY(api->GetDimensions(info,other,1));
    if(other[0]!=dims[rank-1])INVALID("weight width mismatch");
    for(size_t i=0;i<3;++i)TRY(api->GetTensorMutableData((OrtValue*)inputs[i],(void**)&data[i]));
    TRY(api->KernelContext_GetOutput(context,0,dims,rank,&output));TRY(api->GetTensorMutableData(output,(void**)&y));
    if(state->pre){TRY(api->KernelContext_GetOutput(context,1,dims,rank,&output));TRY(api->GetTensorMutableData(output,(void**)&sum));}
    size_t cols=(size_t)dims[rank-1],rows=total/cols;
    RowTask task={data[0],data[1],data[2],y,sum,rows,cols,state->pre};
    if(state->parallel&&rows>=32&&total>=16384){TRY(api->KernelContext_ParallelFor(context,row_task,(rows+7)/8,0,&task));}
    else rows_compute(&task,0,rows);
    cleanup:if(info)api->ReleaseTensorTypeAndShapeInfo(info);return error;
    #undef TRY
    #undef INVALID
}
static const OrtCustomOp post_serial_op={
    .version=ORT_API_VERSION,.GetName=post_serial_name,.GetExecutionProviderType=provider,
    .GetInputType=dtype,.GetInputTypeCount=input_count,.GetOutputType=dtype,.GetOutputTypeCount=output_count,
    .KernelDestroy=destroy,.GetInputCharacteristic=required,.GetOutputCharacteristic=required,
    .GetInputMemoryType=memory,.CreateKernelV2=create,.KernelComputeV2=compute,
    .GetStartVersion=version,.GetEndVersion=version
};
static const OrtCustomOp post_parallel_op={
    .version=ORT_API_VERSION,.GetName=post_parallel_name,.GetExecutionProviderType=provider,
    .GetInputType=dtype,.GetInputTypeCount=input_count,.GetOutputType=dtype,.GetOutputTypeCount=output_count,
    .KernelDestroy=destroy,.GetInputCharacteristic=required,.GetOutputCharacteristic=required,
    .GetInputMemoryType=memory,.CreateKernelV2=create,.KernelComputeV2=compute,
    .GetStartVersion=version,.GetEndVersion=version
};
static const OrtCustomOp pre_serial_op={
    .version=ORT_API_VERSION,.GetName=pre_serial_name,.GetExecutionProviderType=provider,
    .GetInputType=dtype,.GetInputTypeCount=input_count,.GetOutputType=dtype,.GetOutputTypeCount=output_count,
    .KernelDestroy=destroy,.GetInputCharacteristic=required,.GetOutputCharacteristic=required,
    .GetInputMemoryType=memory,.CreateKernelV2=create,.KernelComputeV2=compute,
    .GetStartVersion=version,.GetEndVersion=version
};
static const OrtCustomOp pre_parallel_op={
    .version=ORT_API_VERSION,.GetName=pre_parallel_name,.GetExecutionProviderType=provider,
    .GetInputType=dtype,.GetInputTypeCount=input_count,.GetOutputType=dtype,.GetOutputTypeCount=output_count,
    .KernelDestroy=destroy,.GetInputCharacteristic=required,.GetOutputCharacteristic=required,
    .GetInputMemoryType=memory,.CreateKernelV2=create,.KernelComputeV2=compute,
    .GetStartVersion=version,.GetEndVersion=version
};
ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
    const OrtApi* api=base->GetApi(ORT_API_VERSION);static OrtCustomOpDomain* domain=NULL;OrtStatus* error;
    if(!domain){
        error=api->CreateCustomOpDomain("kernel_lab.rms",&domain);if(error)return error;
        error=api->CustomOpDomain_Add(domain,&post_serial_op);if(error)return error;
        error=api->CustomOpDomain_Add(domain,&post_parallel_op);if(error)return error;
        error=api->CustomOpDomain_Add(domain,&pre_serial_op);if(error)return error;
        error=api->CustomOpDomain_Add(domain,&pre_parallel_op);if(error)return error;
    }
    return api->AddCustomOpDomain(options,domain);
}
