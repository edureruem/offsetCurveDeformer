/*
  offsetCurve OpenCL kernel (cvWrap 방식 벤치마킹)
*/

__kernel void offsetCurveDeform(__global float* finalPos,
                                __global const float* initialPos,
                                __global const float* curvePoints,
                                __global const float* curveTangents,
                                __global const float* paintWeights,
                                __global const int* sampleCounts,
                                __global const int* sampleOffsets,
                                __global const int* sampleIds,
                                __global const float* sampleWeights,
                                __global const int* triangleVerts,
                                __global const float* baryCoords,
                                __global const float4* bindMatrices,
                                __global const float4* scaleMatrix,
                                __global const float* drivenWorldMatrix,
                                __global const float* drivenInvMatrix,
                                const float envelope,
                                const uint positionCount) {
    unsigned int positionId = get_global_id(0);
    if (positionId >= positionCount) {
        return;          
    }
    unsigned int positionOffset = positionId * 3;

    // 초기 위치 읽기
    float initialX = initialPos[positionOffset];
    float initialY = initialPos[positionOffset + 1];
    float initialZ = initialPos[positionOffset + 2];

    // 바인딩 데이터 읽기 (cvWrap 방식)
    float baryA = baryCoords[positionOffset];
    float baryB = baryCoords[positionOffset + 1];
    float baryC = baryCoords[positionOffset + 2];
    
    int triVertA = triangleVerts[positionOffset] * 3;
    int triVertB = triangleVerts[positionOffset + 1] * 3;
    int triVertC = triangleVerts[positionOffset + 2] * 3;

    // 곡선 기반 오프셋 계산 (OCD 특화)
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    float offsetZ = 0.0f;

    // 샘플 가중치 기반 오프셋 계산
    int offset = sampleOffsets[positionId];
    int sampleCount = sampleCounts[positionId];
    
    for (int j = 0; j < sampleCount; j++) {
        int sampleId = sampleIds[offset + j] * 3;
        float weight = sampleWeights[offset + j];
        
        // 곡선 포인트에서의 오프셋 계산
        float curveX = curvePoints[sampleId];
        float curveY = curvePoints[sampleId + 1];
        float curveZ = curvePoints[sampleId + 2];
        
        // 곡선 접선 벡터
        float tangentX = curveTangents[sampleId];
        float tangentY = curveTangents[sampleId + 1];
        float tangentZ = curveTangents[sampleId + 2];
        
        // OCD 알고리즘: 곡선 기반 오프셋
        float distanceX = curveX - initialX;
        float distanceY = curveY - initialY;
        float distanceZ = curveZ - initialZ;
        
        // 접선 방향으로의 투영
        float dotProduct = distanceX * tangentX + distanceY * tangentY + distanceZ * tangentZ;
        
        // 오프셋 벡터 계산
        offsetX += (distanceX - dotProduct * tangentX) * weight;
        offsetY += (distanceY - dotProduct * tangentY) * weight;
        offsetZ += (distanceZ - dotProduct * tangentZ) * weight;
    }

    // 엔벨로프 적용
    offsetX *= envelope;
    offsetY *= envelope;
    offsetZ *= envelope;

    // 페인트 가중치 적용
    float paintWeight = paintWeights[positionId];
    offsetX *= paintWeight;
    offsetY *= paintWeight;
    offsetZ *= paintWeight;

    // 최종 위치 계산
    finalPos[positionOffset] = initialX + offsetX;
    finalPos[positionOffset + 1] = initialY + offsetY;
    finalPos[positionOffset + 2] = initialZ + offsetZ;
}
