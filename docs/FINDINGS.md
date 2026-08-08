# 발견 기록

발견한 사실을 날짜별로 누적 기록한다. 최종 발표 자료의 뼈대가 된다.

## 2026-08-03 (사전 조사)

- lot 1 중심점 si_etch: wafer 1 → 40.89 µm, wafer 10 → 40.18 µm (-1.7%) drift 확인
- Si 식각 39~43 µm, SiO2 소모 0.54~0.74 µm → 선택비 약 60~70
- 89점 기준 중심 42.4 µm, r=95mm 약 50 µm → 최외곽 edge effect 심함
- 9점 vs 89점 (0,0) 지점 1.5 µm 차이 → 측정 장비 상이, 혼용 불가
- 결측: lot 7은 5장만 / lot 8, 10은 9점 데이터 없음 / 89점엔 lot1-wafer7 없음

## 2026-08-04

-

## 2026-08-08 (A: Process_data.nc 파싱 착수)

- `Process_data.nc` = 96개 웨이퍼 그룹. 각 그룹에 `data`(uint16, time×feature),
  `times`, **`feature`(문자열 변수, 채널 이름 존재)**.
- **문서 수정**: `feature` 이름이 `.nc` 안에 있음 (Readme가 맞고 data_structure.md의
  "이름 없음"은 h5py 점검 오류였음). `netCDF4`로 `grp["feature"][:]` 읽으면 됨.
- 31채널은 44채널(07/02)의 **부분집합** → 공통 정렬은 이름 기준 31개 선택으로 해결.
  07/02 추가 13채널: Gas6, Heater5-8, ThermoCouple1-4, 추가 LoadCapacitor 2종,
  attenuatorRatio, moriOuterCurrent.
- 샘플링 dt=0.2s = **5Hz** 확인 (data_structure.md의 4.7Hz는 부정확 → 수정).
- BOSCH 스위칭 펄스 채널: Gas1Flow(0↔200), Gas4Flow(0↔300), Gas5Flow(0↔600).
  소스 ICP SourceRFLoadPower≈2586W, 바이어스 PlatenRF* 펄싱.
- 상수/죽은 채널(분산≈0, 피처 제외 후보): Gas3Flow=0, Gas8Flow=0,
  SourceRF2LoadPower=0, SourceRF2ReflectedPower=0.
  (정정: EpdIntensity는 상수가 아니라 std≈122로 살아있음. 초기 1장 관측 오기)

## 2026-08-08 (A: 파싱 파이프라인 구축 + 96장 전수 채널 스캔)

- **코드**: `src/config.py`(경로/상수, DATA_DIR 환경변수), `src/process_parse.py`
  (웨이퍼 단위 파싱). 규칙 준수: 2000프레임 청크 + float32 + `set_auto_mask(False)`.
- **디코딩 방식 확정**: `Dictionary_process.nc`의 `data`는 **평면 코드북**(float32,
  49290개). uint16 코드가 이 배열의 인덱스 → `codebook[codes]` 한 번에 디코딩. 채널 공용.
- **1장 검증 통과**(07_02_Wafer_01): 3245프레임, 5.00Hz, 값 물리적으로 타당
  (Pressure 0.001~0.074, PlatenDcBias ±30V대, SourceRFPeakToPeak ~3000).
- **96장 전수 스캔으로 채널 확정**:
  - 공통 **31채널** (44채널 07_02는 상위집합, 여분은 대부분 죽은 채널이라 손실 無).
  - **죽은 채널 4개**(전 웨이퍼 std=0, 제외): Gas3Flow, Gas8Flow,
    SourceRF2LoadPower, SourceRF2ReflectedPower.
  - **유효 27채널**. 단 거의 상수(std<0.2, 저분산 필터 후보):
    Gas2Flow, Gas7Flow, Heater1Temp, Heater3Temp, Heater4Temp, Pressure.
  - 정보량 큰 채널(std_max): SourceRFPeakToPeak(1294), SourceRFLoadPower(1111),
    Gas5Flow(289), PlatenRFPeakToPeak(283), SourceRFReflectedPower(195),
    EpdIntensity(122), Gas4Flow(123) → 피처 엔지니어링 주력 후보.
- **다음(2단계)**: `src/process_features.py` — 웨이퍼당 시계열 → 요약 피처
  (채널별 mean/std/기울기(drift)/BOSCH 사이클 진폭) → `process_features.parquet`.
