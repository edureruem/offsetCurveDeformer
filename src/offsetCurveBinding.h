/**
 * offsetCurveBinding.h
 * 곡선-정점 바인딩 정보를 저장하는 클래스
 * 소니 특허(US8400455) 구현을 위한 데이터 구조
 */

#ifndef OFFSETCURVEBINDING_H
#define OFFSETCURVEBINDING_H

#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>

/**
 * @class offsetCurveBinding
 * @brief 메시의 각 정점과 곡선 사이의 바인딩 정보를 캡슐화
 * 
 * 이 클래스는 오프셋 곡선 디포머에서 사용되는 바인딩 데이터를 저장합니다.
 * 바인딩 위치, 로컬 좌표계, 가중치 및 아티스트 제어를 위한 추가 데이터를 포함합니다.
 */
class offsetCurveBinding {
public:
    /**
     * @brief 기본 생성자
     * 모든 멤버 변수를 기본값으로 초기화합니다.
     */
    offsetCurveBinding();
    
    /**
     * @brief 소멸자
     */
    ~offsetCurveBinding();
    
    //---------- 데이터 설정 메서드 ----------//
    
    /**
     * @brief 영향을 주는 곡선의 인덱스 설정
     * @param index 곡선 인덱스
     */
    void setCurveIndex(int index);
    
    /**
     * @brief 곡선 상의 파라미터 위치 설정
     * @param param 파라미터 값 (보통 0.0 ~ 1.0)
     */
    void setParamU(double param);
    
    /**
     * @brief 영향 가중치 설정
     * @param weight 가중치 값 (0.0 ~ 1.0)
     */
    void setWeight(double weight);
    
    /**
     * @brief 바인딩 시점의 로컬 좌표 설정
     * @param point 로컬 좌표 (프레넷 프레임 기준)
     */
    void setBindLocalPoint(const MPoint& point);
    
    /**
     * @brief 바인딩 시점의 변환 행렬 설정
     * @param matrix 변환 행렬
     */
    void setBindMatrix(const MMatrix& matrix);
    
    /**
     * @brief 탄젠트 벡터 설정
     * @param tangent 정규화된 탄젠트 벡터
     */
    void setTangent(const MVector& tangent);
    
    /**
     * @brief 노말 벡터 설정
     * @param normal 정규화된 노말 벡터
     */
    void setNormal(const MVector& normal);
    
    /**
     * @brief 바이노멀 벡터 설정
     * @param binormal 정규화된 바이노멀 벡터
     */
    void setBinormal(const MVector& binormal);
    
    /**
     * @brief 곡률 값 설정
     * @param curvature 곡률 값
     */
    void setCurvature(double curvature);
    
    /**
     * @brief 세그먼트 인덱스 설정 (아크 모드용)
     * @param index 세그먼트 인덱스
     */
    void setSegmentIndex(int index);
    
    /**
     * @brief 연결부 여부 설정 (아크 모드용)
     * @param isJunction 연결부 여부
     */
    void setIsJunction(bool isJunction);
    
    /**
     * @brief 연결부 반경 설정 (아크 모드용)
     * @param radius 연결부 반경
     */
    void setJunctionRadius(double radius);
    
    /**
     * @brief 세그먼트 길이 설정
     * @param length 세그먼트 길이
     */
    void setSegmentLength(double length);
    
    //---------- 데이터 접근 메서드 ----------//
    
    /**
     * @brief 곡선 인덱스 반환
     * @return 영향을 주는 곡선 인덱스
     */
    int getCurveIndex() const;
    
    /**
     * @brief 파라미터 위치 반환
     * @return 곡선 상의 파라미터 위치
     */
    double getParamU() const;
    
    /**
     * @brief 영향 가중치 반환
     * @return 가중치 값 (0.0 ~ 1.0)
     */
    double getWeight() const;
    
    /**
     * @brief 바인딩 시점의 로컬 좌표 반환
     * @return 로컬 좌표 (프레넷 프레임 기준)
     */
    MPoint getBindLocalPoint() const;
    
    /**
     * @brief 바인딩 시점의 변환 행렬 반환
     * @return 변환 행렬에 대한 참조
     */
    const MMatrix& getBindMatrix() const;
    
    /**
     * @brief 탄젠트 벡터 반환
     * @return 탄젠트 벡터에 대한 참조
     */
    const MVector& getTangent() const;
    
    /**
     * @brief 노말 벡터 반환
     * @return 노말 벡터에 대한 참조
     */
    const MVector& getNormal() const;
    
    /**
     * @brief 바이노멀 벡터 반환
     * @return 바이노멀 벡터에 대한 참조
     */
    const MVector& getBinormal() const;
    
    /**
     * @brief 곡률 값 반환
     * @return 곡률 값
     */
    double getCurvature() const;
    
    /**
     * @brief 세그먼트 인덱스 반환 (아크 모드용)
     * @return 세그먼트 인덱스
     */
    int getSegmentIndex() const;
    
    /**
     * @brief 연결부 여부 반환 (아크 모드용)
     * @return 연결부 여부
     */
    bool isJunction() const;
    
    /**
     * @brief 연결부 반경 반환 (아크 모드용)
     * @return 연결부 반경
     */
    double getJunctionRadius() const;
    
    /**
     * @brief 세그먼트 길이 반환
     * @return 세그먼트 길이
     */
    double getSegmentLength() const;

private:
    // 멤버 변수 (비공개)
    int mCurveIndex;           // 영향을 주는 곡선 인덱스
    double mParamU;            // 곡선 상의 파라미터 위치
    double mWeight;            // 영향 가중치
    MPoint mBindLocalPoint;    // 바인딩 시점의 로컬 좌표
    MMatrix mBindMatrix;       // 바인딩 시점의 변환 행렬
    MVector mTangent;          // 탄젠트 벡터
    MVector mNormal;           // 노말 벡터
    MVector mBinormal;         // 바이노멀 벡터
    
    // 아티스트 제어를 위한 추가 데이터
    double mCurvature;         // 곡률 (곡률 기반 효과용)
    double mSegmentLength;     // 영향 세그먼트 길이
    int mSegmentIndex;         // 세그먼트 인덱스 (아크 모드용)
    bool mIsJunction;          // 연결부 여부 (아크 모드용)
    double mJunctionRadius;    // 연결부 반경 (아크 모드용)
};

#endif // OFFSETCURVEBINDING_H