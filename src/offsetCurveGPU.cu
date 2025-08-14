/**
 * offsetCurveGPU.cu
 * OCD GPU 가속화를 위한 CUDA 커널
 * 소니 특허(US8400455) 기반 최적화
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// GPU 상수 메모리
__constant__ float d_falloffRadius;
__constant__ int d_maxInfluences;
__constant__ float d_deformationStrength;

// GPU 공유 메모리
__shared__ float s_curveData[256];  // 곡선 데이터 캐시

/**
 * GPU 커널: 오프셋 벡터 계산
 * 각 스레드가 하나의 정점을 처리
 */
__global__ void calculateOffsetVectorsGPU(
    const float3* __restrict__ modelPoints,
    const float3* __restrict__ curvePoints,
    const float* __restrict__ curveTangents,
    const float* __restrict__ curveNormals,
    const float* __restrict__ curveBinormals,
    float3* __restrict__ offsetVectors,
    const int* __restrict__ influenceIndices,
    const float* __restrict__ weights,
    const int numVertices,
    const int numCurvePoints
) {
    int vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexIdx >= numVertices) return;
    
    float3 modelPoint = modelPoints[vertexIdx];
    float3 totalOffset = make_float3(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;
    
    // 각 영향 곡선에 대해 오프셋 계산
    for (int i = 0; i < d_maxInfluences; ++i) {
        int curveIdx = influenceIndices[vertexIdx * d_maxInfluences + i];
        if (curveIdx < 0) continue;
        
        float weight = weights[vertexIdx * d_maxInfluences + i];
        if (weight <= 0.0f) continue;
        
        // 가장 가까운 곡선 점 찾기
        float minDistance = FLT_MAX;
        int closestPointIdx = 0;
        
        for (int j = 0; j < numCurvePoints; ++j) {
            float3 curvePoint = curvePoints[curveIdx * numCurvePoints + j];
            float3 diff = make_float3(
                modelPoint.x - curvePoint.x,
                modelPoint.y - curvePoint.y,
                modelPoint.z - curvePoint.z
            );
            float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            
            if (distance < minDistance) {
                minDistance = distance;
                closestPointIdx = j;
            }
        }
        
        // 오프셋 벡터 계산 (특허 공식)
        float3 curvePoint = curvePoints[curveIdx * numCurvePoints + closestPointIdx];
        float3 tangent = make_float3(
            curveTangents[curveIdx * numCurvePoints + closestPointIdx * 3 + 0],
            curveTangents[curveIdx * numCurvePoints + closestPointIdx * 3 + 1],
            curveTangents[curveIdx * numCurvePoints + closestPointIdx * 3 + 2]
        );
        float3 normal = make_float3(
            curveNormals[curveIdx * numCurvePoints + closestPointIdx * 3 + 0],
            curveNormals[curveIdx * numCurvePoints + closestPointIdx * 3 + 1],
            curveNormals[curveIdx * numCurvePoints + closestPointIdx * 3 + 2]
        );
        float3 binormal = make_float3(
            curveBinormals[curveIdx * numCurvePoints + closestPointIdx * 3 + 0],
            curveBinormals[curveIdx * numCurvePoints + closestPointIdx * 3 + 1],
            curveBinormals[curveIdx * numCurvePoints + closestPointIdx * 3 + 2]
        );
        
        // 월드 오프셋을 로컬 좌표계로 변환 (특허 핵심 공식)
        float3 worldOffset = make_float3(
            modelPoint.x - curvePoint.x,
            modelPoint.y - curvePoint.y,
            modelPoint.z - curvePoint.z
        );
        
        float3 localOffset;
        localOffset.x = worldOffset.x * tangent.x + worldOffset.y * tangent.y + worldOffset.z * tangent.z;
        localOffset.y = worldOffset.x * normal.x + worldOffset.y * normal.y + worldOffset.z * normal.z;
        localOffset.z = worldOffset.x * binormal.x + worldOffset.y * binormal.y + worldOffset.z * binormal.z;
        
        // 가중치 적용
        totalOffset.x += localOffset.x * weight;
        totalOffset.y += localOffset.y * weight;
        totalOffset.z += localOffset.z * weight;
        totalWeight += weight;
    }
    
    // 정규화 및 최종 오프셋 저장
    if (totalWeight > 0.0f) {
        offsetVectors[vertexIdx] = make_float3(
            totalOffset.x / totalWeight,
            totalOffset.y / totalWeight,
            totalOffset.z / totalWeight
        );
    } else {
        offsetVectors[vertexIdx] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

/**
 * GPU 커널: 변형 적용
 * 오프셋 벡터를 사용하여 최종 위치 계산
 */
__global__ void applyDeformationGPU(
    const float3* __restrict__ originalPoints,
    const float3* __restrict__ offsetVectors,
    float3* __restrict__ deformedPoints,
    const float deformationStrength,
    const int numVertices
) {
    int vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexIdx >= numVertices) return;
    
    float3 originalPoint = originalPoints[vertexIdx];
    float3 offset = offsetVectors[vertexIdx];
    
    // 변형 강도 적용
    deformedPoints[vertexIdx] = make_float3(
        originalPoint.x + offset.x * deformationStrength,
        originalPoint.y + offset.y * deformationStrength,
        originalPoint.z + offset.z * deformationStrength
    );
}

/**
 * GPU 메모리 관리 클래스
 */
class OffsetCurveGPUManager {
private:
    // GPU 메모리
    float3* d_modelPoints;
    float3* d_curvePoints;
    float* d_curveTangents;
    float* d_curveNormals;
    float* d_curveBinormals;
    float3* d_offsetVectors;
    int* d_influenceIndices;
    float* d_weights;
    float3* d_deformedPoints;
    
    // 호스트 메모리
    thrust::host_vector<float3> h_modelPoints;
    thrust::host_vector<float3> h_curvePoints;
    thrust::host_vector<float> h_curveTangents;
    thrust::host_vector<float> h_curveNormals;
    thrust::host_vector<float> h_curveBinormals;
    thrust::host_vector<int> h_influenceIndices;
    thrust::host_vector<float> h_weights;
    thrust::host_vector<float3> h_deformedPoints;
    
    // GPU 설정
    int maxThreadsPerBlock;
    int numBlocks;
    
public:
    OffsetCurveGPUManager() : d_modelPoints(nullptr), d_curvePoints(nullptr),
                              d_curveTangents(nullptr), d_curveNormals(nullptr),
                              d_curveBinormals(nullptr), d_offsetVectors(nullptr),
                              d_influenceIndices(nullptr), d_weights(nullptr),
                              d_deformedPoints(nullptr) {
        // GPU 설정 초기화
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    }
    
    ~OffsetCurveGPUManager() {
        cleanup();
    }
    
    // GPU 메모리 할당
    bool allocateMemory(int numVertices, int numCurves, int maxInfluences) {
        size_t vertexSize = numVertices * sizeof(float3);
        size_t curveSize = numCurves * sizeof(float3);
        size_t influenceSize = numVertices * maxInfluences * sizeof(int);
        size_t weightSize = numVertices * maxInfluences * sizeof(float);
        
        // GPU 메모리 할당
        cudaMalloc(&d_modelPoints, vertexSize);
        cudaMalloc(&d_curvePoints, curveSize);
        cudaMalloc(&d_curveTangents, curveSize * 3 * sizeof(float));
        cudaMalloc(&d_curveNormals, curveSize * 3 * sizeof(float));
        cudaMalloc(&d_curveBinormals, curveSize * 3 * sizeof(float));
        cudaMalloc(&d_offsetVectors, vertexSize);
        cudaMalloc(&d_influenceIndices, influenceSize);
        cudaMalloc(&d_weights, weightSize);
        cudaMalloc(&d_deformedPoints, vertexSize);
        
        // 호스트 메모리 할당
        h_modelPoints.resize(numVertices);
        h_curvePoints.resize(numCurves);
        h_curveTangents.resize(numCurves * 3);
        h_curveNormals.resize(numCurves * 3);
        h_curveBinormals.resize(numCurves * 3);
        h_influenceIndices.resize(numVertices * maxInfluences);
        h_weights.resize(numVertices * maxInfluences);
        h_deformedPoints.resize(numVertices);
        
        // 블록 수 계산
        numBlocks = (numVertices + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
        
        return true;
    }
    
    // GPU 메모리 해제
    void cleanup() {
        if (d_modelPoints) cudaFree(d_modelPoints);
        if (d_curvePoints) cudaFree(d_curvePoints);
        if (d_curveTangents) cudaFree(d_curveTangents);
        if (d_curveNormals) cudaFree(d_curveNormals);
        if (d_curveBinormals) cudaFree(d_curveBinormals);
        if (d_offsetVectors) cudaFree(d_offsetVectors);
        if (d_influenceIndices) cudaFree(d_influenceIndices);
        if (d_weights) cudaFree(d_weights);
        if (d_deformedPoints) cudaFree(d_deformedPoints);
        
        d_modelPoints = d_curvePoints = d_curveTangents = d_curveNormals = 
        d_curveBinormals = d_offsetVectors = d_influenceIndices = d_weights = 
        d_deformedPoints = nullptr;
    }
    
    // GPU 가속 변형 실행
    bool executeDeformation(const float3* modelPoints, int numVertices, 
                           const float3* curvePoints, int numCurves,
                           const int* influenceIndices, const float* weights,
                           float deformationStrength, float falloffRadius,
                           int maxInfluences) {
        // 호스트에서 GPU로 데이터 복사
        cudaMemcpy(d_modelPoints, modelPoints, numVertices * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_curvePoints, curvePoints, numCurves * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_influenceIndices, influenceIndices, numVertices * maxInfluences * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, weights, numVertices * maxInfluences * sizeof(float), cudaMemcpyHostToDevice);
        
        // GPU 상수 메모리 설정
        cudaMemcpyToSymbol(d_falloffRadius, &falloffRadius, sizeof(float));
        cudaMemcpyToSymbol(d_maxInfluences, &maxInfluences, sizeof(int));
        cudaMemcpyToSymbol(d_deformationStrength, &deformationStrength, sizeof(float));
        
        // 오프셋 벡터 계산 커널 실행
        calculateOffsetVectorsGPU<<<numBlocks, maxThreadsPerBlock>>>(
            d_modelPoints, d_curvePoints, d_curveTangents, d_curveNormals, d_curveBinormals,
            d_offsetVectors, d_influenceIndices, d_weights, numVertices, numCurves
        );
        
        // 변형 적용 커널 실행
        applyDeformationGPU<<<numBlocks, maxThreadsPerBlock>>>(
            d_modelPoints, d_offsetVectors, d_deformedPoints, deformationStrength, numVertices
        );
        
        // GPU에서 호스트로 결과 복사
        cudaMemcpy(h_deformedPoints.data(), d_deformedPoints, numVertices * sizeof(float3), cudaMemcpyDeviceToHost);
        
        return true;
    }
    
    // 결과 가져오기
    const thrust::host_vector<float3>& getDeformedPoints() const {
        return h_deformedPoints;
    }
};
