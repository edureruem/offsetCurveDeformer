/**
  OCD 프로젝트를 위한 공통 헬퍼 함수들
*/

#ifndef OCD_COMMON_H
#define OCD_COMMON_H

#include <maya/MDagPath.h>
#include <maya/MDoubleArray.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MIntArray.h>
#include <maya/MMatrix.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MString.h>
#include <maya/MVector.h>
#include <maya/MVectorArray.h>
#include <map>
#include <vector>
#include <set>

#ifdef __AVX__
#include <xmmintrin.h>
#include <immintrin.h>
#endif

// OCD 태스크 데이터 구조체
struct OffsetCurveTaskData {
  MMatrix drivenMatrix;
  MMatrix drivenInverseMatrix;
  float envelope;
  float offsetDistance;
  float falloffRadius;

  MIntArray membership;
  MFloatArray paintWeights;
  MPointArray points;

  // 영향 곡선 데이터
  std::vector<MPointArray> influenceCurvePoints;
  std::vector<MVectorArray> influenceCurveTangents;
  std::vector<MVectorArray> influenceCurveNormals;
  
  // 바인딩 데이터
  MMatrixArray bindMatrices;
  std::vector<MIntArray> sampleCurveIds;
  std::vector<MFloatArray> sampleWeights;
  MVectorArray offsetVectors;
  MPointArray samplePoints;
};

// 프로그레스 바 관리 함수들
void StartProgress(const MString& title, unsigned int count);
void StepProgress(int step);
bool ProgressCancelled();
void EndProgress();

// DAG 노드 및 메시 처리 함수들
bool IsShapeNode(MDagPath& path);
MStatus GetShapeNode(MDagPath& path, bool intermediate=false);
MStatus GetDagPath(MString& name, MDagPath& path);
MStatus DeleteIntermediateObjects(MDagPath& path);

// 곡선 관련 유틸리티 함수들
MStatus GetCurveData(const MDagPath& curvePath, MPointArray& points, 
                     MVectorArray& tangents, MVectorArray& normals);
MStatus GetCurveLength(const MPointArray& points, double& length);
MStatus GetCurveParameterAtDistance(const MPointArray& points, double distance, double& parameter);

// 수학적 유틸리티 함수들
double CalculateDistance(const MPoint& p1, const MPoint& p2);
MVector CalculateNormal(const MPoint& p1, const MPoint& p2, const MPoint& p3);
MVector CalculateTangent(const MPoint& p1, const MPoint& p2);
double ClampValue(double value, double min, double max);
double Lerp(double a, double b, double t);

// 곡선 오프셋 계산 함수들
MStatus CalculateCurveOffset(const MPointArray& baseCurve, 
                            const MVectorArray& baseNormals,
                            double offsetDistance,
                            MPointArray& offsetCurve);
MStatus CalculateCurveOffsetWithFalloff(const MPointArray& baseCurve,
                                       const MVectorArray& baseNormals,
                                       const MPoint& targetPoint,
                                       double offsetDistance,
                                       double falloffRadius,
                                       MPoint& offsetPoint,
                                       float& weight);

// 멀티스레딩 지원
template <typename T>
struct ThreadData {
  unsigned int start;
  unsigned int end;
  unsigned int numTasks;
  double* alignedStorage;
  T* pData;

#ifdef __AVX__
  ThreadData() {
    alignedStorage = (double*) _mm_malloc(4*sizeof(double), 256);
  }
  ~ThreadData() {
    _mm_free(alignedStorage);
  }
#endif
};

template <typename T>
void CreateThreadData(int taskCount, unsigned int elementCount, T* taskData, ThreadData<T>* threadData) {
  unsigned int taskLength = (elementCount + taskCount - 1) / taskCount;
  unsigned int start = 0;
  unsigned int end = taskLength;
  int lastTask = taskCount - 1;
  
  for(int i = 0; i < taskCount; i++) {
    if (i == lastTask) {
      end = elementCount;
    }
    threadData[i].start = start;
    threadData[i].end = end;
    threadData[i].numTasks = taskCount;
    threadData[i].pData = taskData;

    start += taskLength;
    end += taskLength;
  }
}

#ifdef __AVX__
// AVX 최적화된 벡터 연산
template <typename T>
__m256d Dot4(double w1, double w2, double w3, double w4,
             const T& p1, const T& p2, const T& p3, const T& p4) {
  __m256d xxx = _mm256_set_pd(p1.x, p2.x, p3.x, p4.x);
  __m256d yyy = _mm256_set_pd(p1.y, p2.y, p3.y, p4.y);
  __m256d zzz = _mm256_set_pd(p1.z, p2.z, p3.z, p4.z);
  __m256d www = _mm256_set_pd(w1, w2, w3, w4);
  __m256d xw = _mm256_mul_pd(xxx, www);
  __m256d yw = _mm256_mul_pd(yyy, www);
  __m256d zw = _mm256_mul_pd(zzz, www);
  __m256d ww = _mm256_mul_pd(www, www);
  
  __m256d temp01 = _mm256_hadd_pd(xw, yw);   
  __m256d temp23 = _mm256_hadd_pd(zw, ww);
  __m256d swapped = _mm256_permute2f128_pd(temp01, temp23, 0x21);
  __m256d blended = _mm256_blend_pd(temp01, temp23, 0xC);
  __m256d dotproduct = _mm256_add_pd(swapped, blended);
  return dotproduct;
}
#endif

#endif
