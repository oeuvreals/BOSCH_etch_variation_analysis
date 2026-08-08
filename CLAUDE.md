# 프로젝트
BOSCH DRIE 식각 공정 산포 구조 분해 + 챔버 drift 감지 (2주, 2인)

# 데이터 위치
- data/raw/ 에 원본, data/processed/ 에 산출물
- 원본·대용량 산출물(*.nc/*.parquet/*.xlsx)은 gitignore. 절대 커밋 금지.
- 예외: data/processed/*.csv (마스터 테이블 등 수십 KB 인터페이스 테이블)만 커밋 허용.

# 파일 구조
- Si_Oxide_etch_9_points.csv : 웨이퍼당 9행 (측정점별)
  컬럼: experiment_key, lot_number, wafer_number, loc_id, X, Y,
        preox_thickness, postox_thickness, stepheight, oxide_etch, si_etch
- Si_Oxide_etch_89_points.csv : 웨이퍼당 89행
  컬럼: 위와 유사 + postox_thickness_nan (IDW 보간 표시, "N/A")
- Lot_status.xlsx : lot ↔ 날짜 ↔ 컨디셔닝 조건(C/Si/SiO2, 1/3/9회) 매핑
- Process_data.nc : 공정 파라미터 31종, 5Hz
  그룹명 형식 "Day_2024_07_02_Wafer_02", 디코더는 Dictionary_process.nc
- Day_2024_*.nc : OES. 그룹 Wafer_01~10, 디코더는 Dictionary_OES.nc
  3648 파장 채널(185.89~883.97nm), 25Hz, 웨이퍼당 약 15000 프레임

# 반드시 지킬 규칙
1. .nc 를 통째로 메모리에 올리지 말 것.
   웨이퍼 단위 + 청크(2000프레임) + float32 다운캐스트.
   float64 디코딩 시 웨이퍼당 440MB.
   Dataset 열 때 set_auto_mask(False) 사용.
   g["data"][:] 처럼 슬라이스 없이 읽지 말 것.
2. 모델 검증은 반드시 GroupKFold(groups=lot). 랜덤 분할 절대 금지.
   같은 lot 웨이퍼는 매우 유사해 성능이 거짓으로 부풀려짐.
3. 9점과 89점은 측정 장비가 달라 약 1.5 µm 계통 offset 존재. 혼용 금지.
4. 89점의 postox_thickness_nan == "N/A" 는 IDW 보간값. 플래그로 별도 관리.
5. 좌표 X, Y 단위는 µm. 두께·식각량 단위도 µm.
6. 결측: lot 7은 5장만 / lot 8, 10은 9점 없음 / 89점엔 lot1-wafer7 없음.
   experiment_key 가 빈 행 9줄 존재 → 제외하고 기록.
7. 샘플이 웨이퍼 97장뿐인 p >> n 문제. 딥러닝 쓰지 말 것.

# 파일 담당 (충돌 방지)
- A: src/process_*.py, src/model.py
- B: src/analysis_*.py, src/spc.py, src/build_master_table.py

# 코드 스타일
- 중간 결과는 data/processed/ 에 parquet 또는 csv
- 그림은 figures/ 에 PNG, dpi=150
- 새 스크립트 실행 전 반드시 웨이퍼 1장으로 먼저 테스트할 것
