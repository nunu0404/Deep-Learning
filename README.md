# GAP GUI Bug

웹 UI 스크린샷에서 시각적 버그를 탐지하는 VLM 평가 파이프라인과, 비전 토큰을 학습 없이 줄이는 `GAP (GUI-Aware Pruning)` 실험 코드를 정리한 재현용 프로젝트입니다.

이 README는 다음 두 가지를 목표로 작성했습니다.

1. 이 프로젝트가 무엇을 하는지 처음 보는 사람도 바로 이해할 수 있게 설명합니다.
2. 이 README만 따라가도 환경 설치, 데모 실행, 전체 실험 실행까지 진행할 수 있게 안내합니다.

## 1. 이 프로젝트는 무엇을 하나요?

이 프로젝트는 크게 4단계로 구성됩니다.

1. 웹 UI HTML을 수집하고, 인위적으로 시각 버그를 주입해 `GUI-BugBench` 형태의 데이터셋을 만듭니다.
2. 여러 VLM이 이 버그를 얼마나 잘 맞추는지 baseline 성능을 측정합니다.
3. `GAP`으로 비전 토큰 일부를 제거한 뒤, 성능 저하 없이 계산량을 줄일 수 있는지 평가합니다.
4. 결과를 논문용 Figure / Table 형태로 정리합니다.

즉, 핵심 질문은 아래와 같습니다.

`"GUI 화면에서 중요한 시각 토큰만 남기고 나머지를 버려도, 버그 탐지 성능을 유지할 수 있는가?"`

## 2. 다루는 버그 종류

데이터셋 생성 단계에서 아래 5가지 버그를 인위적으로 주입합니다.

- `B1`: Layout Overlap
- `B2`: Text Overflow
- `B3`: Z-index Collision
- `B4`: Truncation
- `B5`: Color Contrast

모델은 각 스크린샷에 대해 아래 둘 중 하나로 답하도록 평가됩니다.

- `CLEAN`
- `BUG: <bug_type>`

평가 시 내부적으로는 아래 라벨로 변환해 사용합니다.

- `B1 -> OVERLAP`
- `B2 -> OVERFLOW`
- `B3 -> ZINDEX`
- `B4 -> TRUNCATION`
- `B5 -> CONTRAST`

## 3. GAP이란?

`GAP (GUI-Aware Pruning)`은 ViT 기반 vision encoder에서 덜 중요한 patch token을 제거하는 학습 없는 pruning 방법입니다.

각 패치에 대해 아래 신호를 계산합니다.

- attention score
- color entropy
- edge density

그리고 아래 식으로 GUI saliency score를 만듭니다.

`GUI_SS = alpha * A + beta * (1 - E) + gamma * D`

이 점수가 높은 patch token만 남기고 나머지는 제거합니다. 목표는 다음과 같습니다.

- 추론 속도 개선
- 메모리 사용량 감소
- FLOPs 감소
- GUI 버그 탐지 성능 최대한 유지

## 4. 프로젝트 구조

```text
gap-gui-bug/
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
├── configs/
│   └── default.yaml
├── scripts/
│   ├── run_all.sh
│   └── run_quick_demo.sh
├── src/
│   ├── dataset/
│   │   ├── build_dataset.py
│   │   └── bug_injectors.py
│   ├── models/
│   │   ├── gap_pruning.py
│   │   └── vlm_wrapper.py
│   ├── evaluation/
│   │   ├── evaluate_baseline.py
│   │   └── evaluate_gap.py
│   └── analysis/
│       └── analyze_results.py
└── tests/
    ├── test_bug_injectors.py
    └── test_gap_pruner.py
```

각 디렉토리 역할은 아래와 같습니다.

- `src/dataset`
  데이터셋 생성, HTML 저장, 버그 주입, Playwright 렌더링, 메타데이터 기록
- `src/models`
  GAP pruning 로직과 모델 래퍼
- `src/evaluation`
  baseline 평가, GAP 평가
- `src/analysis`
  Figure, Table, 통계 분석 생성
- `scripts`
  데모 실행, 전체 실험 실행
- `configs/default.yaml`
  데이터 수, drop rate, 모델 이름 등 기본 설정
- `tests`
  핵심 컴포넌트 단위 테스트

## 5. 어떤 모델을 지원하나요?

Baseline 평가에서 아래 모델을 지원합니다.

- `qwen2vl`
  `Qwen/Qwen2-VL-7B-Instruct-AWQ`
- `llava`
  `llava-hf/llava-v1.6-mistral-7b-hf`
- `internvl`
  `OpenGVLab/InternVL2-8B-AWQ`

백엔드는 다음과 같습니다.

- `qwen2vl`, `internvl`: `vllm`
- `llava`: `transformers + bitsandbytes 4-bit`

현재 GAP pruning 경로는 `Qwen2-VL` 계열에 맞춰 구현되어 있습니다.

## 6. 실행 환경 권장 사항

최소 요구 수준과 권장 수준을 구분해서 보는 것이 좋습니다.

최소:

- Linux
- Python 3.11 권장
- CUDA 사용 가능 GPU
- VRAM 8GB 이상

권장:

- VRAM 24GB 이상
- 여유 있는 디스크 공간
  모델 캐시, 데이터셋, 결과물까지 고려하면 수십 GB 이상이 필요할 수 있습니다.

주의:

- `vLLM`은 첫 실행 시 커널 컴파일과 엔진 초기화 때문에 시간이 더 걸릴 수 있습니다.
- Hugging Face 다운로드 중 일부 환경에서 `xet` 문제가 발생할 수 있어, 이 프로젝트의 실행 스크립트는 기본적으로 `HF_HUB_DISABLE_XET=1`을 사용합니다.

## 7. 설치 방법

가장 권장하는 방식은 `conda` 환경을 사용하는 것입니다.

### Conda 설치

```bash
cd gap-gui-bug
conda env create -f environment.yml
conda activate gap-gui-bug
python -m pip install -e .
python -m playwright install --with-deps chromium
```

### pip 설치

이미 적절한 Python 환경이 있다면 아래도 가능합니다.

```bash
cd gap-gui-bug
pip install -r requirements.txt
pip install -e .
python -m playwright install --with-deps chromium
```

### 설치 확인

```bash
pytest -q
```

정상이라면 테스트 2개가 통과해야 합니다.

## 8. 가장 빠른 시작 방법

가장 먼저 아래 데모를 실행해 보세요.

```bash
cd gap-gui-bug
./scripts/run_quick_demo.sh
```

이 스크립트는 아래를 순서대로 수행합니다.

1. 소규모 데이터셋 생성
   - 총 10장
   - clean 5장
   - bug 5장
   - 각 버그 타입당 1장
2. `qwen2vl` baseline dry run
3. GAP dry run (`drop_rate=0.0, 0.5`)
4. Figure / Table 생성

데모 결과는 아래에 저장됩니다.

- `demo_artifacts/data`
- `demo_artifacts/results`
- `demo_artifacts/figures`
- `demo_artifacts/tables`

## 9. 전체 실험 실행

전체 파이프라인은 아래 스크립트 하나로 실행할 수 있습니다.

```bash
cd gap-gui-bug
./scripts/run_all.sh
```

이 스크립트는 아래 순서로 동작합니다.

1. GPU 메모리를 확인하고 8GB 미만이면 경고 출력
2. `configs/default.yaml`을 읽음
3. 데이터셋 생성
4. baseline 평가
   - `qwen2vl`
   - `llava`
   - `internvl`
5. GAP 평가
   - 현재 `qwen2vl` 기준 drop-rate sweep
6. 결과 분석
7. stdout 요약 표 출력

기본 출력 경로는 아래입니다.

- `data/`
- `results/`
- `figures/`
- `tables/`

## 10. 팀 단위 실험 운영 방식

이 프로젝트는 한 사람이 모든 실험을 혼자 돌리는 방식뿐 아니라, 여러 명이 역할을 나눠 병렬로 실험하는 방식도 염두에 두고 구성했습니다.

가장 권장하는 운영 방식은 아래와 같습니다.

1. 코드 오너 1명이 실험 코드를 정리해서 GitHub `main` 또는 별도 tag로 고정합니다.
2. 데이터셋 생성은 1명이 한 번만 수행합니다.
3. 생성된 데이터셋을 팀 전체가 공통으로 사용합니다.
4. 팀원들은 같은 코드 버전과 같은 `metadata.csv`를 사용해, 맡은 모델 또는 drop rate만 실행합니다.
5. 결과 JSON / CSV를 한곳에 모아 마지막 분석은 1명이 수행합니다.

핵심은 아래 두 가지입니다.

- 코드 버전은 모두 같아야 합니다.
- 데이터셋도 모두 같아야 합니다.

즉, 팀원마다 데이터셋을 따로 다시 만들지 않는 것이 중요합니다.

### 10-1. 권장 역할 분담

보통은 아래처럼 나누는 것이 가장 실용적입니다.

- 코드 오너
  - GitHub 저장소 관리
  - README / 설정 파일 관리
  - 공통 데이터셋 생성
  - 최종 결과 취합 및 분석
- 팀원 A
  - 맡은 baseline 모델 실험
- 팀원 B
  - 맡은 baseline 모델 실험
- 팀원 C
  - 맡은 GAP drop-rate sweep 일부

예시:

- 당신: `qwen2vl` baseline + GAP `0.0~0.3`
- 팀원1: `llava` baseline + GAP `0.4~0.6`
- 팀원2: `internvl` baseline + GAP `0.7~0.9`

### 10-2. 코드 버전 고정

팀원 전원이 반드시 같은 코드 버전을 써야 합니다.

가장 쉬운 방법은 실험 시작 전에 tag를 하나 만드는 것입니다.

코드 오너:

```bash
git checkout main
git pull origin main
git tag exp-v1
git push origin exp-v1
```

팀원:

```bash
git clone https://github.com/nunu0404/Deep-Learning.git
cd Deep-Learning/gap-gui-bug
git checkout exp-v1
```

이렇게 하면 “이번 실험은 `exp-v1` 기준”이라고 명확하게 고정할 수 있습니다.

### 10-3. 데이터셋은 한 번만 생성

데이터셋은 코드 오너 또는 지정된 1명이 생성하는 것을 권장합니다.

예시:

```bash
cd gap-gui-bug
conda activate gap-gui-bug
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET=1

python -m dataset.build_dataset \
  --n_samples 5000 \
  --samples-per-bug 500 \
  --seed 42 \
  --output-dir /shared/gap-gui-bug/data_v1
```

공유 대상:

- `/shared/gap-gui-bug/data_v1/metadata.csv`
- `/shared/gap-gui-bug/data_v1/screenshots/`
- `/shared/gap-gui-bug/data_v1/html/`

이후 팀원들은 모두 이 공통 데이터셋만 읽습니다.

### 10-4. 팀원 공통 초기 세팅

팀원들은 실험 전에 아래까지만 먼저 수행하면 됩니다.

```bash
git clone https://github.com/nunu0404/Deep-Learning.git
cd Deep-Learning/gap-gui-bug
git checkout exp-v1

conda env create -f environment.yml
conda activate gap-gui-bug
python -m pip install -e .
python -m playwright install --with-deps chromium

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET=1

pytest -q
./scripts/run_quick_demo.sh
```

이 단계가 성공하면 해당 팀원 환경은 실험 준비가 끝난 것입니다.

### 10-5. RQ별 실험 분담 예시

#### RQ1. Baseline 모델 비교

질문:

`pruning 없이 어떤 VLM이 GUI bug를 가장 잘 탐지하는가?`

필요 실험:

- `qwen2vl` baseline
- `llava` baseline
- `internvl` baseline

예시 분담:

- 당신: `qwen2vl`
- 팀원1: `llava`
- 팀원2: `internvl`

실행 예시:

```bash
python -m evaluation.evaluate_baseline \
  --model qwen2vl \
  --metadata-path /shared/gap-gui-bug/data_v1/metadata.csv \
  --results-dir /shared/gap-gui-bug/results/rq1/baseline/your_name
```

```bash
python -m evaluation.evaluate_baseline \
  --model llava \
  --metadata-path /shared/gap-gui-bug/data_v1/metadata.csv \
  --results-dir /shared/gap-gui-bug/results/rq1/baseline/member1
```

```bash
python -m evaluation.evaluate_baseline \
  --model internvl \
  --metadata-path /shared/gap-gui-bug/data_v1/metadata.csv \
  --results-dir /shared/gap-gui-bug/results/rq1/baseline/member2
```

#### RQ2. GAP drop-rate 실험

질문:

`qwen2vl에서 토큰을 얼마나 버려도 성능이 유지되는가?`

예시 분담:

- 당신: `0.0,0.1,0.2,0.3`
- 팀원1: `0.4,0.5,0.6`
- 팀원2: `0.7,0.8,0.9`

실행 예시:

```bash
python -m evaluation.evaluate_gap \
  --model qwen2vl \
  --metadata-csv /shared/gap-gui-bug/data_v1/metadata.csv \
  --output-dir /shared/gap-gui-bug/results/rq2/gap/your_name \
  --drop-rates "0.0,0.1,0.2,0.3" \
  --alpha 0.4 \
  --beta 0.3 \
  --gamma 0.3
```

```bash
python -m evaluation.evaluate_gap \
  --model qwen2vl \
  --metadata-csv /shared/gap-gui-bug/data_v1/metadata.csv \
  --output-dir /shared/gap-gui-bug/results/rq2/gap/member1 \
  --drop-rates "0.4,0.5,0.6" \
  --alpha 0.4 \
  --beta 0.3 \
  --gamma 0.3
```

```bash
python -m evaluation.evaluate_gap \
  --model qwen2vl \
  --metadata-csv /shared/gap-gui-bug/data_v1/metadata.csv \
  --output-dir /shared/gap-gui-bug/results/rq2/gap/member2 \
  --drop-rates "0.7,0.8,0.9" \
  --alpha 0.4 \
  --beta 0.3 \
  --gamma 0.3
```

#### RQ3. Sensitivity 분석

질문:

`어떤 bug type이 pruning에 가장 민감한가?`

이건 새로운 실행이 따로 필요한 실험이 아니라, RQ2에서 모인 GAP 결과를 분석 스크립트가 다시 읽어 계산하는 방식입니다.

즉:

- 팀원들이 RQ2 결과만 제대로 생성하면 됨
- 최종 분석 담당 1명이 sensitivity curve를 생성

#### RQ4. Ablation 실험

질문:

`attention / entropy / edge density 중 어떤 성분이 중요한가?`

예시 분담:

- 당신: full GAP `alpha=0.4 beta=0.3 gamma=0.3`
- 팀원1: attention only `alpha=1.0 beta=0.0 gamma=0.0`
- 팀원2: attention + entropy `alpha=0.7 beta=0.3 gamma=0.0`

예시:

```bash
python -m evaluation.evaluate_gap \
  --model qwen2vl \
  --metadata-csv /shared/gap-gui-bug/data_v1/metadata.csv \
  --output-dir /shared/gap-gui-bug/results/ablation/member1 \
  --drop-rates "0.0,0.3,0.5" \
  --alpha 1.0 \
  --beta 0.0 \
  --gamma 0.0
```

### 10-6. 어떤 스크립트를 팀원에게 맡겨야 하나

팀 분산 실험에서는 `run_all.sh`보다 개별 명령 실행 방식을 권장합니다.

이유:

- `run_all.sh`는 1인 end-to-end 실행용입니다.
- 팀 단위 분산 실행에서는 모델 또는 drop rate를 잘라서 나누는 것이 더 효율적입니다.
- 각 팀원이 자기 역할만 정확히 수행하도록 하기 쉽습니다.

즉:

- 코드 오너 1명: `build_dataset.py`, 마지막 `analyze_results.py`
- 팀원들: `evaluate_baseline.py`, `evaluate_gap.py`

### 10-7. 결과물 취합 구조

팀 결과는 아래처럼 모으는 것을 권장합니다.

```text
results/
  rq1/
    baseline/
      your_name/
      member1/
      member2/
  rq2/
    gap/
      your_name/
      member1/
      member2/
  ablation/
    your_name/
    member1/
    member2/
```

baseline에서 팀원이 제출해야 할 파일:

- `*_baseline.json`
- `*_predictions.csv`

GAP에서 팀원이 제출해야 할 파일:

- `qwen2vl_dr*.json`
- `qwen2vl_dr*_predictions.csv`

### 10-8. 최종 분석은 한 명이 수행

결과물이 모두 모이면 마지막 분석 담당자가 한 번만 실행합니다.

```bash
python -m analysis.analyze_results \
  --model-name qwen2vl \
  --metadata-csv /shared/gap-gui-bug/data_v1/metadata.csv \
  --figures-dir /shared/gap-gui-bug/final_figures \
  --tables-dir /shared/gap-gui-bug/final_tables \
  --baseline-glob "/shared/gap-gui-bug/results/rq1/baseline/**/*baseline.json" \
  --gap-glob "/shared/gap-gui-bug/results/rq2/gap/**/*.json" \
  --random-glob "/shared/gap-gui-bug/results/random/**/*.json" \
  --fastv-glob "/shared/gap-gui-bug/results/fastv/**/*.json" \
  --ablation-glob "/shared/gap-gui-bug/results/ablation/**/*.json" \
  --skip-patch-viz
```

### 10-9. 팀 운영 시 가장 중요한 규칙

- 팀원마다 데이터셋을 따로 만들지 말 것
- 같은 git tag 또는 같은 commit을 사용할 것
- 모두 같은 `metadata.csv`를 사용할 것
- 각 팀원은 서로 다른 `results-dir`를 사용할 것
- 결과물은 공용 저장소나 공유 스토리지에 모을 것
- 최종 Figure / Table은 한 명이 취합해서 생성할 것

### 10-10. 현재 구현 범위 기준 주의 사항

- `Random drop` 생성기는 아직 자동 구현되어 있지 않습니다.
- `FastV` 생성기도 아직 자동 구현되어 있지 않습니다.
- 따라서 현재 팀 분산 실험은 우선 `Baseline + GAP + Ablation` 중심으로 운영하는 것이 현실적입니다.

## 11. 설정 파일

기본 설정은 `configs/default.yaml`에 있습니다.

예를 들면 아래 값들이 들어 있습니다.

- 데이터셋 샘플 수
- 버그 타입별 샘플 수
- random seed
- train / val / test 비율
- GAP 하이퍼파라미터
  - `alpha`
  - `beta`
  - `gamma`
- drop rate 목록
- 사용할 모델 이름

설정을 바꾸고 싶다면 `configs/default.yaml`을 수정한 뒤 `run_all.sh`를 다시 실행하면 됩니다.

## 12. 단계별로 직접 실행하고 싶은 경우

자동 스크립트 대신 각 단계별로 직접 실행할 수도 있습니다.

### 11-1. 데이터셋 생성

기본 balanced mode 예시는 아래와 같습니다.

```bash
python -m dataset.build_dataset \
  --n_samples 5000 \
  --samples-per-bug 500 \
  --seed 42 \
  --output-dir data
```

이 설정은 아래 의미입니다.

- clean 2500장
- bug 2500장
- 버그 타입별 500장

생성 결과:

- `data/screenshots/*.png`
- `data/html/*.html`
- `data/metadata.csv`
- `data/dataset_stats.json`
- `data/checkpoint.json`
- `data/sample_status.jsonl`

### 11-2. Baseline 평가

예시: `qwen2vl`

```bash
HF_HUB_DISABLE_XET=1 python -m evaluation.evaluate_baseline \
  --model qwen2vl \
  --metadata-path data/metadata.csv \
  --results-dir results/baseline
```

예시: dry run

```bash
HF_HUB_DISABLE_XET=1 python -m evaluation.evaluate_baseline \
  --model qwen2vl \
  --metadata-path data/metadata.csv \
  --results-dir results/baseline \
  --dry_run
```

출력:

- `results/baseline/{model}_baseline.json`
- `results/baseline/{model}_predictions.csv`
- `results/baseline/errors.log`

### 11-3. GAP 평가

```bash
HF_HUB_DISABLE_XET=1 python -m evaluation.evaluate_gap \
  --model qwen2vl \
  --metadata-csv data/metadata.csv \
  --output-dir results/gap \
  --drop-rates "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
```

예시: dry run

```bash
HF_HUB_DISABLE_XET=1 python -m evaluation.evaluate_gap \
  --model qwen2vl \
  --metadata-csv data/metadata.csv \
  --output-dir results/gap \
  --drop-rates "0.0,0.5" \
  --dry-run
```

출력:

- `results/gap/qwen2vl_dr0.0.json`
- `results/gap/qwen2vl_dr0.0_predictions.csv`
- `results/gap/qwen2vl_dr0.5.json`
- `results/gap/qwen2vl_dr0.5_predictions.csv`
- `results/gap/errors.log`

### 11-4. 분석 및 Figure / Table 생성

```bash
python -m analysis.analyze_results \
  --model-name qwen2vl \
  --metadata-csv data/metadata.csv \
  --figures-dir figures \
  --tables-dir tables \
  --baseline-glob "results/baseline/**/*baseline.json" \
  --gap-glob "results/gap/**/*.json" \
  --random-glob "results/random/**/*.json" \
  --fastv-glob "results/fastv/**/*.json" \
  --ablation-glob "results/ablation/**/*.json" \
  --skip-patch-viz
```

출력:

- `figures/pareto_curve.pdf`
- `figures/pareto_curve.png`
- `figures/sensitivity_curves.pdf`
- `figures/sensitivity_curves.png`
- `figures/vss_correlation.pdf`
- `tables/main_results.tex`
- `tables/ablation.tex`

## 13. 주요 산출물 설명

### `data/metadata.csv`

각 이미지에 대한 메타데이터입니다.

주요 컬럼:

- `sample_id`
- `image_path`
- `label`
  - `0 = clean`
  - `1 = bug`
- `bug_type`
  - `None` 또는 `B1~B5`
- `vss_score`

### `results/*_predictions.csv`

모델이 각 샘플에 대해 어떻게 예측했는지 기록합니다.

주요 컬럼:

- 정답 라벨
- 예측 라벨
- raw model output
- latency
- peak VRAM
- retry 여부
- error 여부

### `results/*.json`

실험 요약 지표 파일입니다.

예시 포함 항목:

- accuracy
- precision / recall / f1
- macro F1
- per bug type 지표
- confusion matrix
- latency
- VRAM
- drop rate 관련 요약

## 14. 테스트

```bash
pytest -q
```

현재 테스트는 아래를 검증합니다.

- 각 bug injector가 유효한 HTML을 생성하는지
- clean / buggy가 충분히 다른 시각 구조를 가지는지
- GAP pruner가 지정된 비율만큼 token을 제거하는지
- `CLS` 토큰을 유지하는지

## 15. 현재 구현 상태에서 알아둘 점

이 프로젝트는 데이터 생성, baseline 평가, GAP 평가, 분석까지 전체 흐름이 실행되도록 구성되어 있습니다. 다만 아래는 미리 알고 시작하는 것이 좋습니다.

- `Random drop` 결과 생성기는 아직 자동 구현되어 있지 않습니다.
- `FastV` 결과 생성기도 아직 자동 구현되어 있지 않습니다.
- 분석 스크립트는 `results/random`과 `results/fastv`가 있으면 함께 읽어 그래프에 포함합니다.
- 즉, 논문용 4-way 비교를 완전히 채우려면 해당 결과를 외부에서 추가로 준비해야 합니다.

## 16. 자주 겪는 문제

### Hugging Face 다운로드가 이상하게 멈추거나 `416 Range Not Satisfiable`가 발생하는 경우

아래처럼 실행하세요.

```bash
HF_HUB_DISABLE_XET=1 python -m evaluation.evaluate_baseline ...
```

`run_quick_demo.sh`와 `run_all.sh`에는 이미 이 설정이 들어 있습니다.

### 첫 실행이 너무 느린 경우

정상일 가능성이 높습니다.

- `vLLM` 엔진 초기화
- CUDA graph capture
- torch compile
- kernel cache 생성

같은 환경에서 두 번째 실행은 더 빨라질 수 있습니다.

### VRAM 사용량이 예상보다 크게 보이는 경우

정상일 수 있습니다.

- `vLLM`은 별도 엔진 프로세스를 사용합니다.
- 따라서 일부 측정값은 `transformers` 경로와 다르게 보일 수 있습니다.

## 17. 한 번에 따라 하기

아무것도 모르는 상태에서 가장 안전한 실행 순서는 아래입니다.

```bash
cd gap-gui-bug
conda env create -f environment.yml
conda activate gap-gui-bug
python -m pip install -e .
python -m playwright install --with-deps chromium
pytest -q
./scripts/run_quick_demo.sh
```

문제가 없으면 그 다음에 전체 실험으로 넘어갑니다.

```bash
./scripts/run_all.sh
```

`run_quick_demo.sh`와 `run_all.sh`는 모두 현재 단계가 무엇인지, 결과물이 어디에 저장되는지 로그로 출력하도록 구성되어 있습니다.

## 18. 라이선스 / 참고

이 저장소는 재현 가능한 연구용 스캐폴드 성격의 프로젝트입니다. 사용한 외부 모델과 데이터셋의 라이선스 및 사용 조건은 각 원본 저장소의 안내를 반드시 확인해야 합니다.
