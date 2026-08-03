# 데이터 취득 방법

원본 데이터는 용량 문제로 저장소에 포함하지 않는다. 아래 절차로 각자 로컬에 준비할 것.

## 1. 다운로드

Zenodo: https://zenodo.org/records/17122442

전체 (7.9 GB):

    pip install zenodo_get
    zenodo_get 10.5281/zenodo.17122442

## 2. 배치

data/raw/ 아래에 아래 파일들을 둔다.

| 파일 | 크기 | 본 프로젝트 사용 |
|---|---|---|
| Lot_status.xlsx | 12 kB | 필수 |
| Si_Oxide_etch_9_points.csv | 47 kB | 메인 |
| Si_Oxide_etch_89_points.csv | 648 kB | 메인 |
| Process_data.nc | 8.8 MB | 모델링 입력 |
| Dictionary_process.nc | 88 kB | 디코더 |
| Dictionary_OES.nc | 90 kB | 보너스 |
| Day_2024_*.nc | 830 MB x 10 | 1개만 사용 (보너스) |

## 3. 중간 산출물 공유

Google Drive: (링크 추가 예정)

- wafers_9pt.csv — 마스터 테이블 (9점 기준)
- wafers_89pt.csv — 마스터 테이블 (89점 기준)
- process_features.parquet — 공정 파라미터 요약

## 4. 주의사항

- .nc 파일은 dictionary encoding 되어 있어 디코더 필요
- 9점과 89점은 측정 장비가 달라 약 1.5 µm 계통 offset 존재 → 혼용 금지
- 좌표 X, Y 단위는 µm / 두께·식각량 단위도 µm
- 89점 CSV의 postox_thickness_nan == "N/A" 는 IDW 보간값
