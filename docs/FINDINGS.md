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

## 2026-08-08 (A: 2단계 요약 피처 생성 + 공정 드리프트 인사이트)

**목적**: 27채널 시계열을 웨이퍼당 1행으로 축약해 VM 모델 입력을 만들고,
그 과정에서 "-1.7% si_etch 드리프트를 설명할 공정 신호"를 미리 탐색한다.

**산출물**: `src/process_features.py` → `data/processed/process_features.parquet`
(96 웨이퍼 × 108 피처 + 키). 채널별 4종:
- `mean`(평균 수준), `std`(변동성·펄싱 진폭 반영),
- `slope`(웨이퍼 **내부** 시간 기울기 /s = 런 내부 드리프트),
- `range`(robust p95-p05 = 펄싱 스위칭 폭 프록시).
왜 4종인가: BOSCH는 펄싱 공정이라 평균만으론 스위칭 진폭·듀티를 못 잡는다.
mean(레벨)·std/range(스위칭)·slope(드리프트)로 **레벨/변동/추세**를 분리.

### 인사이트 1 — 지휘 채널 vs 응답 채널 (피처 스크리닝 근거)
- 웨이퍼 간 std≈0인 `range` 6종: **Gas4Flow/Gas5Flow(C4F8/SF6 펄스 진폭),
  EpdIntensity, Heater1Temp, SourceRF2** 등. Gas4/5는 진폭이 매 웨이퍼 동일(설정값).
- **판단**: 설정값(지휘) 채널의 진폭은 웨이퍼마다 똑같아 VM 정보량 0. 정보는
  **규제/응답 채널**(매칭 캐패시터, 반사파, 온도)의 미세 변동에 있다 → 저분산
  피처는 `model.py`에서 분산필터로 자동 제거 예정.
- **주의(방법론)**: CV(=std/|mean|)로 정보량 랭킹하면 slope처럼 평균≈0 피처가
  CV 폭발로 상위권을 오염시킨다. **0중심 피처는 절대 std로 스크리닝**해야 함.

### 인사이트 2 — 챔버 임피던스 드리프트가 유력한 식각 드리프트 원인
- **웨이퍼 진행위치(1→10)와 채널 상관, lot 교락 제거(within-date) 후**:
  - `PlatenRFTuningCapacitor__mean` **within-lot r=+0.768** (pooled +0.743과
    거의 동일 → lot/컨디셔닝 교락 아님, **진짜 웨이퍼-진행 드리프트**).
  - 보강: `Heater3/4Temp` +0.34, `HeliumBPFlow` -0.27, `PlatenDcBias` -0.25,
    `SourceRFReflectedPower` +0.17 (모두 약하지만 방향 일관).
  - **음성 결과**: `SourceRFLoadPower` within-lot r≈-0.03 → **ICP 소스 파워는
    규제되어 드리프트 없음.** 즉 식각 드리프트는 파워 감소가 아니다.
- **공정 해석**: 플래튼(바이어스) RF 매칭의 튜닝 캐패시터가 웨이퍼가 쌓일수록
  한 방향으로 이동 = 챔버 RF **임피던스가 계통적으로 드리프트**. 원인 후보는
  (a) C4F8 패시베이션 폴리머·식각 부산물의 챔버벽/전극 누적, (b) 전극·벽 온도
  상승(Heater·He backside 흐름 동반 변화). 둘 다 전형적 챔버 컨디셔닝 드리프트.
- **왜 중요**: 이 캐패시터 위치는 사실상 **내장 챔버-상태 센서**. -1.7% si_etch
  드리프트의 1순위 기전 후보이자 SPC 조기경보 지표 후보. 소스 파워가 규제됨을
  확인했으므로 기전을 "임피던스/열"로 좁힘.
- **미해결(교락 한계)**: date를 lot 프록시로 썼다. B의 `lot_conditions.csv` 도착 후
  실제 lot·컨디셔닝(C/Si/SiO2 ×1/3/9)으로 재계층화해 확정할 것.

### 인사이트 3 — EpdIntensity는 연속 피처가 아니라 first-wafer 플래그
- `EpdIntensity__mean`: 위치2~10에서 `min=max=0.123`(완전 상수), **위치1만
  ~18.5–19.4** (10개 date 전부 일관). 위치와의 상관 -0.52는 **단일점 지레**일 뿐
  단조 드리프트 아님.
- **해석**: 매 lot **첫 웨이퍼만 EPD가 살아있고** 이후는 바닥값에 고정 → 취득
  설정상 첫 장만 로깅되거나 first-wafer effect. 연속 VM 피처로 부적합.
- **판단/조치**: EpdIntensity mean/range를 그대로 쓰지 말고 `is_first_wafer`
  이진 플래그로 변환하거나 제외. B에게 EPD 취득 설정 확인 요청.
- (정정) 8/8 초기 스캔의 "EpdIntensity std≈122 살아있음"은 **위치1에 집중된
  변동**이었음. 대다수 웨이퍼에선 사실상 상수.

### 인사이트 4 — 런 내부(11분) 드리프트도 방향 100% 일관
- `slope` 부호가 전 웨이퍼 동일한 채널: `SourceRFPeakToPeak`(+0.67/s),
  `PlatenRF*Capacitor`(매칭 캐패 이동), `SourceRFReflectedPower`(-0.056/s, 반사파
  감소=플라즈마 안정화), `Gas1Flow`(+, SF6↑)·`Gas4Flow`(-, C4F8↓)=식각/패시베이션
  균형이 런 내에서 식각쪽으로 이동.
- **뉘앙스**: 방향이 전 웨이퍼 동일 = 이 slope는 웨이퍼 간 분산이 작다 →
  물리적으론 실재하나 **웨이퍼 구분(VM 예측)엔 기여 적음**. 인사이트 2의
  '웨이퍼 간' 드리프트(캐패 mean)와는 **다른 시간척도**(런 내부 vs 웨이퍼 간)이나
  같은 임피던스-드리프트 기전을 두 척도에서 함께 가리킴.
