# YOLO Colab 부트캠프 실습 자료

Google Colab에서 Ultralytics YOLO로 따라 하는 단계별 핸즈온 실습 노트북과 해설서입니다. 모든 노트북은 Colab **T4 GPU** 런타임 기준이고, 결과는 Google Drive(`MyDrive/yolo_day1`, `MyDrive/yolo_day2`, `MyDrive/yolo_workshop`)에 저장됩니다.

## 전체 흐름

| 회차 | 하는 일 |
|---|---|
| 1회차 | 현업 사례 데모 — YOLO가 현장에서 무엇을 하는지 보고 포트폴리오 한 장 만들기 |
| 2회차 | 직접 만들기 — "모델이 찾고, 규칙이 판단한다" 구조를 코드로 짜 보기 |
| 2~3회차 워크숍 | 주제를 골라 같은 구조를 내 데이터에 적용하기 (개인 또는 2인 1조) |

## 폴더 구성

```
day1/        1회차 노트북(01~04)과 해설서(md)
day2/        2회차 노트북(01~06)
day2/guides/ 2회차 해설서(md) — 노트북 번호와 같은 번호
workshop/    워크숍 안내서(01)와 시작 노트북(02)
*/word/      같은 해설서의 Word(docx)·PDF 파일 — 내려받아 인쇄하거나 보관할 때
```

해설서는 GitHub에서 **md 파일을 바로 열어 읽으면** 됩니다. `word/` 폴더는 같은 내용의 내려받기용 파일입니다.

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
| 03 | [`03_yolo_day2_pose_lab_v1.2`](day2/03_yolo_day2_pose_lab_v1.2.ipynb) | 관절 각도 규칙으로 상자 들기 동작의 위험도를 매기기 | 물류 갈래 | [읽기](day2/guides/03_yolo_day2_pose_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/03_yolo_day2_pose_lab_v1.2.ipynb) |
| 04 | [`04_yolo_day2_monitor_lab_v1.1`](day2/04_yolo_day2_monitor_lab_v1.1.ipynb) | 01과 03의 모델·기준값을 이어 하나의 관제 화면으로 묶기 | 물류 갈래 | [읽기](day2/guides/04_yolo_day2_monitor_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/04_yolo_day2_monitor_lab_v1.1.ipynb) |
| 05 | [`05_yolo_day2_prompt_lab_v1.1`](day2/05_yolo_day2_prompt_lab_v1.1.ipynb) | 학습 없이 단어만으로 물체 찾기(YOLOE), 학습 모델과 비교 | 데이터 갈래 | [읽기](day2/guides/05_yolo_day2_prompt_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/05_yolo_day2_prompt_lab_v1.1.ipynb) |
| 06 | [`06_yolo_day2_autolabel_lab_v1.1`](day2/06_yolo_day2_autolabel_lab_v1.1.ipynb) | 05의 방법으로 라벨을 자동으로 만들고 사람·자동·하이브리드 라벨 비교 | 데이터 갈래 | [읽기](day2/guides/06_yolo_day2_autolabel_lab_guide_v1.1.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/day2/06_yolo_day2_autolabel_lab_v1.1.ipynb) |

해설서에는 셀마다 코드, 코드 설명, 실행 결과, 막혔을 때 해결법이 있고 끝에 도전 과제·자주 겪는 문제·용어 정리가 있습니다. Word 파일은 [`day2/word/`](day2/word)에 같은 번호로 있습니다.

- 03·04는 상자를 드는 동작을 옆에서 찍은 10~20초 영상(`lift_sample.mp4`)이 필요합니다. 휴대폰으로 직접 찍어 노트북의 파일 선택 창에서 올리면 되고, 아이폰 `.mov`도 그대로 쓸 수 있습니다. 찍는 요령은 [03 해설서](day2/guides/03_yolo_day2_pose_lab_guide_v1.1.md)에 있습니다.
- 05의 Step 4는 01에서 Drive에 저장한 모델(`MyDrive/yolo_day2/box_seg_best.pt`)을 사용합니다.
- 무료 Colab은 GPU 사용 시간에 한도가 있습니다. 시간이 부족하면 노트북의 `QUICK_RUN = True`로 흐름만 먼저 확인하세요.

---

## 프로젝트 워크숍 (2~3회차)

주제 8개(산업 안전, 화재·연기, 재활용 쓰레기, 도로 파손, 매장 선반, 낙상, 작물 해충, PCB 결함) 중 하나를 골라 개인 또는 2인 1조로 진행합니다.

| 번호 | 자료 | 내용 | 열기 |
|---|---|---|---|
| 01 | [워크숍 안내서](workshop/01_yolo_workshop_student_guide_v1.1.md) | 진행 흐름, 체크포인트 CP1~CP4, 주제 카드 8개, 데이터 확보 방법 | Word: [`workshop/word/`](workshop/word) |
| 02 | [`02_yolo_workshop_starter_v1.1`](workshop/02_yolo_workshop_starter_v1.1.ipynb) | 시작 노트북 — 캐글·URL·Drive 데이터 받기, YOLO/VOC/COCO 형식 자동 변환, 학습·평가·임계값·오류 분석·실험 기록·리포트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjwlee/yolo-training/blob/main/workshop/02_yolo_workshop_starter_v1.1.ipynb) |

> 캐글 API 토큰은 코드에 적지 말고 Colab 왼쪽의 **보안 비밀(열쇠 아이콘)** 에 `KAGGLE_API_TOKEN`으로 등록하세요.

## 참고

- 실습 데이터셋과 사전학습 모델은 [Ultralytics](https://docs.ultralytics.com) 공개 자산(AGPL-3.0)과 각 캐글 데이터셋의 라이선스를 따릅니다.
- 노트북은 ultralytics 8.4 버전에서 동작을 확인했습니다.
