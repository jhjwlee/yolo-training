# YOLO Colab 부트캠프 실습 자료

Google Colab에서 Ultralytics YOLO로 따라 하는 단계별 핸즈온 실습 노트북과 해설서입니다. 모든 노트북은 Colab **T4 GPU** 런타임 기준이고, 결과는 Google Drive(`MyDrive/yolo_day1`, `MyDrive/yolo_day2`, `MyDrive/yolo_day3`, `MyDrive/yolo_workshop`)에 저장됩니다. 3회차에는 내 PC에서 실시간으로 돌려 보는 로컬 실습도 있습니다.

## 전체 흐름

| 회차 | 하는 일 |
|---|---|
| 1회차 | 현업 사례 데모 — YOLO가 현장에서 무엇을 하는지 보고 포트폴리오 한 장 만들기 |
| 2회차 | 직접 만들기 — "모델이 찾고, 규칙이 판단한다" 구조를 코드로 짜 보기 |
| 2~3회차 워크숍 | 주제를 골라 같은 구조를 내 데이터에 적용하기 (개인 또는 2인 1조) |
| 3회차 | 현장 배포 맛보기 — 내 PC에서 실시간 추론, 산업별 응용 데모 4종 |

## 폴더 구성

```
day1/        1회차 노트북(01~04)과 해설서(md)
day2/        2회차 노트북(01~06)
day2/guides/ 2회차 해설서(md) — 노트북 번호와 같은 번호
day3/        3회차 로컬 실습 가이드(01)·Colab판(02)·산업 데모 노트북(03~06)
day3/local/  내 PC에서 실행하는 파이썬 스크립트(01~04)와 requirements.txt
workshop/    워크숍 안내서(01)와 시작 노트북(02)
concepts/    개념 참고자료 (지표 읽는 법 등)
*/word/      같은 해설서의 Word(docx)·PDF 파일 — 내려받아 인쇄하거나 보관할 때
```

해설서는 GitHub에서 **md 파일을 바로 열어 읽으면** 됩니다. `word/` 폴더는 같은 내용의 내려받기용 파일입니다.

**참고자료** — [탐지 성능 지표 읽기: mAP와 혼동 행렬](concepts/01_yolo_metrics_mAP_confusion_v1.0.md) (IoU · TP/FP/FN · 정밀도/재현율 · PR 곡선 · mAP50/50-95 · Box vs Mask · 혼동 행렬을 실습 01·06 결과로 설명)

---

## 1회차 — 현업 사례로 만드는 포트폴리오

| 번호 | 노트북 | 내용 | 열기 |
|---|---|---|---|
| 01 | [`01_yolo_day1_ppe_lab_v1.1.ipynb`](day1/01_yolo_day1_ppe_lab_v1.1.ipynb) | 건설 현장 안전장비(PPE) 미착용 감지 — 학습·평가·판정 규칙·포트폴리오 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day1/01_yolo_day1_ppe_lab_v1.1.ipynb) |
| 02 | [`02_yolo_day1_case_demos_v1.1.ipynb`](day1/02_yolo_day1_case_demos_v1.1.ipynb) | 박스 카운팅 · 운동/낙상 포즈 · 주차장/대기열 사례 시연 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day1/02_yolo_day1_case_demos_v1.1.ipynb) |
| 03 | [`03_yolo_day1_pill_lab_v1.0.ipynb`](day1/03_yolo_day1_pill_lab_v1.0.ipynb) | (참고) 알약 탐지 — 라벨·IoU·NMS·AP부터 오류 분석까지 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day1/03_yolo_day1_pill_lab_v1.0.ipynb) |
| 04 | [`04_yolo_day1_brain_tumor_lab_v1.0.ipynb`](day1/04_yolo_day1_brain_tumor_lab_v1.0.ipynb) | (참고) 뇌종양 영상 탐지 — 클래스별 AP·혼동 행렬·분할 점검 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day1/04_yolo_day1_brain_tumor_lab_v1.0.ipynb) |

해설서: [01·02 핸즈온 가이드](day1/01_02_yolo_day1_handson_guide_v1.1.md) (Word·PDF는 [`day1/word/`](day1/word))

---

## 2회차 — 데모를 넘어 직접 구현하기

### 진행 순서

```
[필수]  01 상자 크기별 집계  →  02 위험구역 감시  →  워크숍 시작
[선택]  물류센터 갈래:   01 → 03 들기 자세 → 04 통합 관제 (01·03 결과 필요)
        데이터 도구 갈래: 05 말로 찾는 탐지기 → 06 오토 라벨링
```

노트북끼리는 Google Drive에 저장한 결과로 이어집니다. 여러 노트북을 동시에 띄울 필요 없이 **하나씩 순서대로** 실행하면 됩니다.

| 번호 | 노트북 | 한 줄 의미 | 구분 | 해설서 | 열기 |
|---|---|---|---|---|---|
| 01 | [`01_yolo_day2_box_lab_v1.2`](day2/01_yolo_day2_box_lab_v1.2.ipynb) | 직접 학습한 분할 모델에 추적·기준선·크기 규칙을 붙여 상자를 S/M/L로 세기 | 필수 | [읽기](day2/guides/01_yolo_day2_box_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/01_yolo_day2_box_lab_v1.2.ipynb) |
| 02 | [`02_yolo_day2_zone_lab_v1.1`](day2/02_yolo_day2_zone_lab_v1.1.ipynb) | 학습 없이 사람을 찾고 위험구역 진입·체류를 이벤트로 판정 | 필수 | [읽기](day2/guides/02_yolo_day2_zone_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/02_yolo_day2_zone_lab_v1.1.ipynb) |
| 03 | [`03_yolo_day2_pose_lab_v1.3`](day2/03_yolo_day2_pose_lab_v1.3.ipynb) | 관절 각도 규칙으로 상자 들기 동작의 위험도를 매기기 | 물류 갈래 | [읽기](day2/guides/03_yolo_day2_pose_lab_guide_v1.2.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/03_yolo_day2_pose_lab_v1.3.ipynb) |
| 04 | [`04_yolo_day2_monitor_lab_v1.1`](day2/04_yolo_day2_monitor_lab_v1.1.ipynb) | 01과 03의 모델·기준값을 이어 하나의 관제 화면으로 묶기 | 물류 갈래 | [읽기](day2/guides/04_yolo_day2_monitor_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/04_yolo_day2_monitor_lab_v1.1.ipynb) |
| 05 | [`05_yolo_day2_prompt_lab_v1.1`](day2/05_yolo_day2_prompt_lab_v1.1.ipynb) | 학습 없이 단어만으로 물체 찾기(YOLOE), 학습 모델과 비교 | 데이터 갈래 | [읽기](day2/guides/05_yolo_day2_prompt_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/05_yolo_day2_prompt_lab_v1.1.ipynb) |
| 06 | [`06_yolo_day2_autolabel_lab_v1.1`](day2/06_yolo_day2_autolabel_lab_v1.1.ipynb) | 05의 방법으로 라벨을 자동으로 만들고 사람·자동·하이브리드 라벨 비교 | 데이터 갈래 | [읽기](day2/guides/06_yolo_day2_autolabel_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/06_yolo_day2_autolabel_lab_v1.1.ipynb) |

해설서에는 셀마다 코드, 코드 설명, 실행 결과, 막혔을 때 해결법이 있고 끝에 도전 과제·자주 겪는 문제·용어 정리가 있습니다. Word 파일은 [`day2/word/`](day2/word)에 같은 번호로 있습니다.

- 03·04는 상자를 드는 동작을 옆에서 찍은 10~20초 영상(`lift_sample.mp4`)을 씁니다. 03(v1.3)은 영상이 없으면 이 리포의 예시 영상(`day2/lift_sample.mp4`, `SAMPLE_NO = 2`면 `lift_sample_02.mp4`)을 자동으로 받습니다. 직접 찍은 영상을 쓰려면 `USE_SAMPLE = False`로 바꾸고 파일 선택 창에서 올리세요(아이폰 `.mov`도 가능). 찍는 요령은 [03 해설서](day2/guides/03_yolo_day2_pose_lab_guide_v1.2.md)에 있습니다.
- 05의 Step 4는 01에서 Drive에 저장한 모델(`MyDrive/yolo_day2/box_seg_best.pt`)을 사용합니다.
- 무료 Colab은 GPU 사용 시간에 한도가 있습니다. 시간이 부족하면 노트북의 `QUICK_RUN = True`로 흐름만 먼저 확인하세요.

---

## 프로젝트 워크숍 (2~3회차)

주제 8개(산업 안전, 화재·연기, 재활용 쓰레기, 도로 파손, 매장 선반, 낙상, 작물 해충, PCB 결함) 중 하나를 골라 개인 또는 2인 1조로 진행합니다.

| 번호 | 자료 | 내용 | 열기 |
|---|---|---|---|
| 01 | [워크숍 안내서](workshop/01_yolo_workshop_student_guide_v1.2.md) | 진행 흐름, 체크포인트 CP1~CP4, 주제 카드 8개, 데이터 확보 방법 | Word: [`workshop/word/`](workshop/word) |
| 02 | [`02_yolo_workshop_starter_v1.2`](workshop/02_yolo_workshop_starter_v1.2.ipynb) | 시작 노트북 — 캐글·URL·Drive 데이터 받기, YOLO/VOC/COCO 형식 자동 변환, 학습·평가·임계값·오류 분석·실험 기록·리포트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/workshop/02_yolo_workshop_starter_v1.2.ipynb) |

> 캐글 API 토큰은 코드에 적지 말고 Colab 왼쪽의 **보안 비밀(열쇠 아이콘)** 에 `KAGGLE_API_TOKEN`으로 등록하세요.

**v1.2에서 바뀐 점** — `QUICK_RUN = True`로 먼저 돌린 뒤 본 학습을 하면 `experiments.csv`의 `baseline` 행이 본 학습 결과로 바뀝니다(`epochs` 열 추가). 개선 실험으로 최종 모델이 바뀌면 7-2단계에서 클래스별 지표·임계값 표를 최종 모델로 다시 계산하고, 리포트에도 최종 모델 기준 숫자가 들어갑니다. 이미 v1.1로 기록이 쌓인 프로젝트도 v1.2로 이어서 실행하면 됩니다.

---

## 3회차 — 현장으로 가져가기

### 로컬 실습 — 내 PC에서 실시간으로

Colab에서 학습한 모델을 **카메라 옆 컴퓨터**에서 돌리는 연습입니다. GPU가 없는 노트북이어도 됩니다. 설치나 카메라가 잘 안 되면 같은 내용의 Colab판(02)으로 진행하세요.

| 번호 | 자료 | 내용 | 열기 |
|---|---|---|---|
| 01 | [로컬 실습 가이드](day3/01_yolo_day3_local_guide_v1.0.md) | 가상환경 설치 → 환경 점검 → 웹캠 실시간 탐지 → 위험구역 규칙 실시간 실행 → ONNX·OpenVINO 내보내기와 CPU 속도 비교 | 스크립트: [`day3/local/`](day3/local) |
| 02 | [`02_yolo_day3_edge_colab_v1.0`](day3/02_yolo_day3_edge_colab_v1.0.ipynb) | 같은 스크립트를 Colab에서 영상 파일로 실행 (로컬 설치가 안 될 때) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day3/02_yolo_day3_edge_colab_v1.0.ipynb) |

### 산업 응용 데모 — 새로운 작업 유형 맛보기

각 노트북은 10~15분 안에 끝나며, 2회차와 같은 **"모델이 찾고, 규칙이 판단한다"** 구조로 현장 규칙까지 붙입니다.

| 번호 | 노트북 | 산업 | 새로 배우는 것 | 열기 |
|---|---|---|---|---|
| 03 | [`03_yolo_day3_obb_demo_v1.0`](day3/03_yolo_day3_obb_demo_v1.0.ipynb) | 항만·위성 | 회전 박스(OBB) — 항공 사진 속 선박의 각도, 정박 방향 규칙 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day3/03_yolo_day3_obb_demo_v1.0.ipynb) |
| 04 | [`04_yolo_day3_retail_demo_v1.0`](day3/04_yolo_day3_retail_demo_v1.0.ipynb) | 리테일 매장 | 통로별 방문·체류, 혼잡 경보, 동선 히트맵, Ultralytics `solutions` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day3/04_yolo_day3_retail_demo_v1.0.ipynb) |
| 05 | [`05_yolo_day3_cls_demo_v1.0`](day3/05_yolo_day3_cls_demo_v1.0.ipynb) | 제조 품질 | 이미지 분류(classify) — 주조 부품 정상/불량, 3구간 판정 규칙 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day3/05_yolo_day3_cls_demo_v1.0.ipynb) |
| 06 | [`06_yolo_day3_traffic_demo_v1.0`](day3/06_yolo_day3_traffic_demo_v1.0.ipynb) | 교통·주차 관제 | 기준선 통행량, 픽셀→미터 환산과 속도 추정, 서행 위반 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day3/06_yolo_day3_traffic_demo_v1.0.ipynb) |

- 05는 캐글 데이터(`ravirajsinh45/real-life-industrial-dataset-of-casting-product`)를 받으므로 워크숍과 같은 캐글 토큰 설정이 필요할 수 있습니다.
- 04·06 영상은 Intel IoT DevKit 공개 샘플 영상(CC BY 4.0), 03은 Ultralytics `DOTA8` 샘플입니다.

## 참고

- 실습 데이터셋과 사전학습 모델은 [Ultralytics](https://docs.ultralytics.com) 공개 자산(AGPL-3.0)과 각 캐글 데이터셋의 라이선스를 따릅니다. 3회차 04·06 영상 출처: [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) (CC BY 4.0).
- 노트북은 ultralytics 8.4 버전에서 동작을 확인했습니다.
