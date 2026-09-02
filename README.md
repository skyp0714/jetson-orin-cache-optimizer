# Jetson Orin NS-Cache + Accel-Sim 최적화 파이프라인

이 디렉터리는 NS-Cache의 cache-array PPA와 Jetson Orin용 Accel-Sim trace simulation을 연결한다. SRAM L1/L2 기준 설정은 변경하지 않고, gain cell과 STT-MRAM 후보의 capacity/associativity를 반복 탐색한 뒤 4차원 Pareto frontier와 power budget별 최적점을 표와 SVG/HTML 리포트로 자동 생성한다.

> 성능 지표는 trace에 포함된 모든 GPU kernel의 `gpu_tot_sim_cycle`이다. CPU 전처리·후처리, host/device transfer, 애플리케이션 wall-clock time까지 포함한 실제 system end-to-end latency는 아니다. 자세한 범위는 [모델 범위와 제한](#모델-범위와-제한)을 참고한다.

## 최적화 목적

SRAM baseline의 area, performance, energy, average cache power를 각각 1.0으로 정규화한다. 기본 hard constraint는 애플리케이션별로 적용된다.

| 이름 | 탐색 대상 | 목적 함수 | 기본 제약 |
|---|---|---|---|
| `pareto_full_cache` | 아래 두 candidate family의 합집합: SRAM L1 고정 + target L2 retrofit, target technology L1/L2 full-cache | area·energy·power는 최소화하고 performance는 최대화하는 4차원 비지배 frontier 탐색 | budget별 선택 시 total L1+L2 area ≤ SRAM, 설정한 average cache-power cap |
| `min_energy_runtime_bound` | SRAM L1은 고정하고 target technology의 L2 capacity/associativity만 탐색 | suite normalized cache energy 최소화 | 각 application runtime ≤ SRAM × 1.02, total L1+L2 area ≤ SRAM |
| `max_performance_energy_bound` | target technology의 L1/L2 capacity/associativity를 모두 탐색 | suite geometric-mean speedup 최대화 | 각 application cache energy ≤ SRAM, total L1+L2 area ≤ SRAM |

제공된 example config는 공통 후보 집합을 한 번 평가해 여러 operating policy에 재사용하는 `pareto_full_cache`만 실행한다. 기존 두 objective는 이전 설정/결과와의 호환을 위해 계속 지원한다.

`pareto_full_cache`는 한 technology마다 다음 두 placement family를 함께 탐색한다.

1. `l2_retrofit`: immutable baseline SRAM L1을 그대로 두고 target technology L2의 capacity/associativity만 변경한다.
2. `full_cache`: L1과 L2를 모두 target technology로 바꾸고 두 level의 capacity/associativity를 변경한다.

Pareto dominance는 total area, normalized cache energy, normalized average cache power를 작게, application performance를 크게 만드는 방향의 **4차원** 비교다. 따라서 power budget을 바꿀 때마다 Accel-Sim을 다시 돌리지 않고, 동일한 evaluated frontier를 budget별로 필터링할 수 있다.

각 power budget에서는 다음 세 operating point를 별도로 고른다.

| selector | candidate family와 추가 조건 | 선택 기준 |
|---|---|---|
| `min_energy_iso_performance` | `l2_retrofit`; runtime ≤ SRAM × `(1 + runtime_tolerance)` | normalized cache energy 최소 |
| `max_performance_iso_energy` | `full_cache`; cache energy ≤ SRAM × `(1 + energy_tolerance)` | application performance 최대 |
| `balanced_knee` | 두 family 합집합; area와 power hard cap | budget 안의 4D Pareto frontier에서 normalized energy/performance ideal corner에 가장 가까운 절충점 |

위 수치는 예시 설정의 `runtime_tolerance: 0.02`, `energy_tolerance: 0.0`, `max_area_ratio: 1.0`에 해당한다. 기본 `constraint_scope: "per_application"`에서는 relative/absolute power cap, iso-performance runtime bound, iso-energy energy bound를 **모든 application이 각각** 통과해야 한다. 즉 suite 평균이 cap 이하여도 application 하나가 넘으면 feasible이 아니다. `constraint_scope`를 `suite`로 바꾸면 application별 최악값 대신 weighted suite aggregate를 사용한다. feasible 후보가 하나도 없으면 리포트는 최적해라고 가장하지 않고 `NO_FEASIBLE_CANDIDATE` 상태와 실패 이유를 표시한다. Legacy objective의 기존 리포트는 가장 가까운 infeasible 후보도 함께 표시한다.

탐색기는 deterministic iterative beam/Pareto search를 사용한다. baseline, 작은/큰 cache, associativity 경계에서 시작해 feasible/비지배 후보의 이웃을 확장한다. `saturation_area_ratio_max`까지는 strict area constraint를 넘는 점도 saturation graph를 위해 평가할 수 있으며, 이 값을 넘는 후보는 PPA 단계에서 pre-screen한다.

보고서의 선택값은 기록된 discrete search budget 안에서 찾은 **best evaluated candidate**다. 유한 budget의 iterative search이므로 전체 Cartesian space의 global optimum을 수학적으로 보장하지 않으며, HTML/Markdown에 technology/objective별 attempted/completed 수와 stop reason을 함께 기록한다.

## 사전 준비

필요한 항목은 다음과 같다.

- Python 3.8 이상. 실행과 SVG/CSV 생성은 Python 표준 라이브러리만 사용한다. `pytest`는 테스트를 실행할 때만 필요하며, repository root에서 `pytest`를 실행하면 된다(경로 설정은 root의 `pyproject.toml`이 담당한다).
- 빌드된 NS-Cache 실행 파일과 SRAM/gain-cell/STT-MRAM `.cfg`/`.cell` 입력.
- 빌드된 trace-driven Accel-Sim 실행 파일, Jetson Orin `gpgpusim.config`, `trace.config`.
- Accel-Sim이 사용할 CUDA 설치. 현재 환경 생성기는 기본적으로 `/usr/local/cuda`를 사용한다.
- `gpu-simulator/gpgpu-sim/lib/gcc-*/cuda-*/release` 아래에 정확히 하나의 빌드된 GPGPU-Sim release library.
- 각 application의 `kernelslist.g`와 그 목록이 가리키는 `.traceg` 또는 `.traceg.xz` 파일.
- 후보 수 × application 수만큼의 simulation 시간과 저장 공간. 한 후보도 수 분에서 수 시간 이상 걸릴 수 있다.

NS-Cache가 아직 빌드되지 않았다면 repository root에서 다음을 실행한다.

```bash
make -C src -j2
```

기본 예시는 다음 위치를 전제로 한다.

```text
/home/hnpark2/
├── NSCache_UIUC_Collaboration/
│   ├── nsc
│   ├── config_uiuc/
│   └── cache_optimizer/
└── accel-sim-framework/
    ├── gpu-simulator/bin/release/accel-sim.out
    ├── gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM87_ORIN/gpgpusim.config
    ├── gpu-simulator/configs/tested-cfgs/SM87_ORIN/trace.config
    └── hw_run/.../traces/
        ├── kernelslist.g
        └── *.traceg 또는 *.traceg.xz
```

## 설정과 trace 배치

예시를 복사해 같은 디렉터리에 local 설정을 만들면 상대 경로를 그대로 유지할 수 있다.

```bash
cd /home/hnpark2/NSCache_UIUC_Collaboration
cp cache_optimizer/configs/jetson_orin.example.json \
   cache_optimizer/configs/jetson_orin.local.json
```

모든 상대 경로는 **JSON 파일이 있는 디렉터리**를 기준으로 해석된다. 설정 파일을 다른 위치로 옮겼다면 `paths`, `baseline`, `technologies`, application trace 경로도 함께 수정해야 한다.

### Application 지정

예시는 아래 glob으로 모든 `kernelslist.g`를 찾는다.

```json
"application_trace_globs": [
  "../../../accel-sim-framework/hw_run/**/traces/kernelslist.g"
]
```

trace가 다른 위치에 있거나 이름/가중치를 직접 관리하려면 `applications`를 사용한다.

```json
"application_trace_globs": [],
"applications": [
  {
    "name": "openvla",
    "trace": "../../../accel-sim-framework/hw_run/openvla/traces/kernelslist.g",
    "weight": 1.0
  },
  {
    "name": "second_app",
    "trace": "/absolute/path/to/second_app/traces/kernelslist.g",
    "weight": 2.0
  }
]
```

`weight`는 suite 성능/에너지/전력 normalized ratio의 weighted geometric mean에 사용되며 0보다 커야 한다. glob과 explicit list를 함께 써도 동일한 trace 경로는 한 번만 포함된다. 하나 이상의 유효한 `kernelslist.g`가 없으면 validation 단계에서 실행을 중단한다.

### Pareto search와 power constraint

제공된 example config는 다음처럼 unified Pareto search와 여러 power budget을 한 번에 지정한다.

```json
"objectives": [
  "pareto_full_cache"
],
"power_constraints": [
  {
    "name": "sram_1x",
    "max_power_ratio": 1.0
  },
  {
    "name": "sram_1p5x",
    "max_power_ratio": 1.5
  },
  {
    "name": "absolute_300mw",
    "max_power_mw": 300
  }
]
```

`max_power_ratio`는 application별 candidate average cache power / SRAM baseline average cache power의 상한이다. 따라서 `1.0`은 동일 application의 SRAM cache power 이하, `1.5`는 그 1.5배 이하다. `max_power_mw`는 absolute 상한이며, 각 항목은 `max_power_ratio`와 `max_power_mw` 중 정확히 하나만 가져야 한다. `name`은 결과 디렉터리 이름에도 사용되므로 중복될 수 없고 영문자/숫자로 시작하는 안전한 이름이어야 한다.

여기서 absolute power는 simulated workload 동안의 **L1+L2 cache-array average power(mW)** 다. NS-Cache dynamic/leakage/refresh energy를 simulated runtime으로 나눈 값이며 Jetson module/board TDP, GPU 전체 전력, 순간 peak power가 아니다. 기본 `per_application` scope에서는 모든 workload의 average cache-array power가 absolute cap 이하여야 한다.

### 주요 설정 필드

- `paths`: NS-Cache/Accel-Sim root, 실행 파일, Orin config, 결과 디렉터리.
- `baseline`: immutable SRAM L1/L2 NS-Cache config, physical instance 수, bank 수, access counter scope.
- `technologies`: SRAM 이외의 탐색 대상. `l1_template`/`l2_template`과 필요한 경우 `cell_template`, retention 가정을 지정한다. `sram`은 여기에 추가하지 않는다.
- `search_space`: L1/L2 capacity(KiB)와 associativity의 discrete 후보 목록.
- `objectives`: 제공 example은 `pareto_full_cache`만 사용한다. 두 legacy objective도 호환 지원된다.
- `power_constraints`: SRAM-relative 또는 absolute mW hard cap 목록. 각 budget에서 세 selector를 독립적으로 계산한다.
- `constraints`: area/runtime/energy bound와 `per_application`/`suite` 적용 범위.
- `optimizer`: objective·technology당 simulation budget, iteration/beam 크기, 병렬 worker, timeout, resume, saturation 탐색 범위.
- `latency_mapping`: calibrated Orin latency에 NS-Cache의 SRAM 대비 hit-latency 차이만 더하는 mapping 설정.

Accel-Sim cache access counter는 기본적으로 모든 L1/L2 instance를 합친 값이므로 예시는 `l1_counts_scope`와 `l2_counts_scope`를 `all_instances`로 둔다. 자체 parser가 per-instance count를 제공하는 경우에만 `per_instance`로 바꾼다. Area, leakage, refresh는 counter scope와 무관하게 physical instance 수를 곱한다.

`l1_banks_per_instance`와 `l2_banks_per_instance`는 refresh energy multiplier다. 제공된 config는 전체 cache instance를 하나의 NS-Cache bank object로 모델링하므로 예시는 1을 사용한다. `-ForceBankA: AxB`의 A×B는 그 bank **내부 subarray 조직**이며 replicated physical bank 수가 아니므로 이 multiplier로 사용하면 안 된다.

## 검증, 실행, 재개

먼저 경로, trace, Orin/NS-Cache baseline mapping, CUDA 및 GPGPU-Sim library를 검증한다. 이 단계에서는 NS-Cache나 Accel-Sim simulation을 실행하지 않는다.

```bash
cd /home/hnpark2/NSCache_UIUC_Collaboration
python3 -m cache_optimizer \
  --config cache_optimizer/configs/jetson_orin.local.json \
  --validate-only
```

검증이 통과하면 전체 최적화를 실행한다.

```bash
python3 -m cache_optimizer \
  --config cache_optimizer/configs/jetson_orin.local.json
```

짧은 smoke run이나 병렬 실행은 CLI override로 조절할 수 있다.

```bash
python3 -m cache_optimizer \
  --config cache_optimizer/configs/jetson_orin.local.json \
  --max-evaluations 4 \
  --max-workers 2
```

`--max-evaluations`는 technology/objective 조합마다 허용할 최대 candidate **시도 수**다. Geometry preflight, PPA, simulation 단계에서 실패한 후보도 budget을 소비한다. `--max-workers`는 candidate simulation 병렬도이며, 각 Accel-Sim process의 메모리 사용량을 고려해 보수적으로 정한다.

중단된 작업은 **같은 config와 같은 `output_dir`로 같은 명령을 다시 실행**하면 된다. 탐색은 deterministic하게 다시 진행하면서 다음 결과를 재사용한다.

- `work/cache/ppa.json`의 NS-Cache PPA cache
- 입력 hash가 일치하고 정상 종료한 `work/simulations/.../accelsim.stdout.log`

binary/config/kernels list가 바뀌거나 trace file의 크기·수정 시간이 바뀌면 simulation cache key가 달라져 자동 재실행된다. `checkpoint.json`과 `partial_evaluations.jsonl`은 진행 상황/진단용이며, checkpoint의 Python 탐색 상태를 그대로 deserialize하는 방식은 아니다. 정상 완료된 simulation log를 의도적으로 무시하려면 다음처럼 실행한다. 이 옵션은 PPA cache를 삭제하지 않는다.

```bash
python3 -m cache_optimizer \
  --config cache_optimizer/configs/jetson_orin.local.json \
  --no-resume
```

## 결과 파일

결과는 JSON의 `paths.output_dir` 아래에 생성된다. 예시 기본값은 `results/cache_optimizer_jetson_orin`이다.

| 파일 | 내용 |
|---|---|
| `report.html` | 최적 후보, power-budget별 선택 표와 그래프를 한 페이지에서 보는 최종 리포트 |
| `summary.md` | objective/power-budget별 최적 후보와 해석 범위를 요약한 Markdown |
| `optimal_configs.csv` | technology/objective별 선택 후보, feasibility, L1/L2 config 및 normalized metrics |
| `power_constrained_optima.csv` | technology/power budget별 세 selector의 상태, 선택 후보, observed power와 normalized metrics |
| `all_evaluations.csv` | 모든 simulated 후보의 aggregate area/performance/energy/power |
| `per_application.csv` | application별 cycles, IPC, runtime, cache energy/power, L1/L2 access count |
| `pareto_front.csv` | technology/objective 그룹별 area/energy/performance/power 4D 비지배 후보 |
| `saturation_points.csv` | technology별 saturation 검출 여부, 해당 area/performance와 L1/L2 config |
| `results.json` | baseline, provenance, assumptions, 모든 평가, objective별 및 power-budget별 최적 후보를 포함한 machine-readable 결과 |
| `baseline.json` | immutable SRAM PPA와 application별 baseline 결과 |
| `manifest.json` | pipeline version, input hash, git revision, warning/error, search outcome |
| `optimal_configs/<technology>/<objective>/` | 선택된 `nsc_l1.cfg`, `nsc_l2.cfg`, `gpgpusim.config`, `selection.json`, `cells/l1.cell`, `cells/l2.cell` |
| `optimal_configs/<technology>/power_constraints/<budget>/<selector>/` | 해당 power budget/selector의 self-contained config와 `selection.json`; 해가 없을 때도 상태/이유를 기록 |
| `work/` | 생성 config, isolated simulation log, PPA/simulation cache |
| `errors.json`, `partial_evaluations.jsonl`, `checkpoint.json` | 실패 후보와 중간 진행 기록. 상황에 따라 일부 파일은 없을 수 있음 |

자동 생성 그래프는 plotting package가 필요 없는 SVG다.

- `optimization_history.svg`: iteration에 따른 best feasible energy와 speedup. SRAM normalized value 1.0을 점선으로 표시한다.
- `energy_performance_tradeoff.svg`: normalized cache energy 대 geometric-mean speedup, Pareto frontier와 infeasible 점을 구분한다. SRAM energy/performance bound는 점선 1.0이다.
- `power_constrained_optima.svg`: budget마다 candidate의 observed average cache-array power 대 performance를 표시한다. 수직선은 relative 또는 absolute hard cap이며 `E`, `P`, `K`는 각각 iso-performance 최소 energy, iso-energy 최대 performance, balanced knee 선택점이다.
- `area_saturation.svg`: unified Pareto 후보가 있으면 이를 우선 사용하고, 없으면 legacy max-performance 후보를 사용해 area 대비 performance와 average cache power를 표시한다. SRAM area/performance/power 1.0을 점선으로 표시한다. performance envelope에서 인접 두 구간 모두 normalized slope가 0.10 미만인 첫 점을 heuristic saturation point로 표시한다. 이는 “10% area 증가당 1% 미만 speedup”에 해당하며, 설계 의사결정에 맞춰 threshold를 별도로 검토해야 한다.

복사된 optimal NS-Cache config의 cell 경로는 폴더 내부 상대경로다. 독립 실행할 때는 해당 objective 또는 budget/selector 디렉터리를 working directory로 사용한다.

```bash
cd results/cache_optimizer_jetson_orin/optimal_configs/gain_cell/power_constraints/sram_1x/min_energy_iso_performance
/home/hnpark2/NSCache_UIUC_Collaboration/nsc nsc_l2.cfg
```

## 성능, 에너지, 면적 정의

Application runtime은 다음과 같이 계산한다.

```text
runtime_s = gpu_tot_sim_cycle / Orin core_clock_hz
speedup   = SRAM_runtime / candidate_runtime
```

`performance_score`는 application별 speedup의 weighted geometric mean이다. `runtime_ratio`는 candidate runtime / SRAM runtime이고, `worst_runtime_ratio`는 모든 application 중 최댓값이다.

각 cache level의 에너지는 다음 세 항의 합이다.

```text
dynamic = read_hit × hit_energy
        + read_miss × miss_energy
        + write × write_energy

leakage = NS-Cache leakage_power × physical_instances × simulated_runtime

refresh = refresh_energy_per_bank / effective_retention
        × banks_per_instance × physical_instances × simulated_runtime
```

`cache_energy_nj`는 L1+L2 dynamic/leakage/refresh energy이고, `cache_power_mw`는 이 값을 simulated workload runtime으로 나눈 L1+L2 평균 cache-array power다. `energy_score`와 `power_score`는 application별 SRAM ratio의 weighted geometric mean이며, `power_mw_score`는 application별 absolute `cache_power_mw`의 weighted geometric mean이다. `worst_power_ratio`와 `max_power_mw`는 `per_application` hard cap 판정에 사용된다. DRAM, interconnect, compute core, CPU, fan/board energy와 Jetson TDP는 포함하지 않는다.

Total area는 `L1 area × l1_instances + L2 area × l2_instances`다. 예시 Orin 설정은 L1 16개, L2 modeled instance 1개를 사용한다. Dynamic access counters가 `all_instances`이면 이미 aggregate이므로 16을 다시 곱하지 않지만, area/leakage/refresh에는 instance 수를 곱한다.

NS-Cache latency를 calibrated Accel-Sim latency로 그대로 치환하지 않는다. 후보 latency cycle은 다음 relative-delta mapping을 사용한다.

```text
candidate_sim_cycles = calibrated_SRAM_sim_cycles
                     + round_away_from_zero(
                         (candidate_NS_hit_ns - SRAM_NS_hit_ns) × clock_GHz
                       )
```

이 방식은 Orin baseline의 pipeline/architecture latency를 보존하고 cell/cache 변화에 따른 NS-Cache 차이만 반영한다. Cache replacement/write/allocation/indexing policy는 baseline suffix를 유지한다.

## 모델 범위와 제한

결과를 해석할 때 다음 제한을 반드시 함께 기록한다.

1. **GPU trace 성능이지 실제 system E2E가 아니다.** `gpu_tot_sim_cycle`은 trace에 들어 있는 GPU kernel 실행을 누적한 값이다. CPU preprocessing/postprocessing, framework scheduling, host-device transfer와 I/O는 모델링하지 않는다. 실제 end-to-end latency가 필요하면 이 파이프라인 결과에 별도로 측정·검증한 non-GPU 고정/가변 시간을 결합하거나 해당 구간까지 포괄하는 trace/model이 필요하다.
2. **Gain-cell retention은 현재 가정값이다.** 예시는 300 K에서 315,000 µs를 기존 Jetson suite 가정으로 둔다. Pipeline은 bundled NS-Cache PVT 식으로 candidate config temperature에 맞춘 effective retention을 사용해 refresh energy를 계산한다. 측정 또는 technology-calibrated retention이 있으면 `retention_time_us`와 `retention_reference_temperature_k`를 반드시 교체한다.
3. **STT-MRAM은 독립적으로 보정된 N7 macro가 아니다.** 현재 cell parameter는 IBM 14 nm STT-MRAM 기반이며 N7 cache wrapper 안에서 평가한다. 따라서 절대 PPA나 SRAM/gain-cell 대비 결론을 silicon-calibrated N7 STT 결과로 해석하면 안 된다.
4. **Tag와 data는 같은 technology다.** 제공된 NS-Cache template은 tag array에도 data array와 같은 cell technology를 사용한다. “SRAM tag + gain-cell/STT data” 혼합 macro 결과가 아니다.
5. **Energy 범위는 L1/L2 array다.** Cache 변화가 DRAM traffic, NoC, memory-controller, core/system power에 미치는 에너지 효과는 더하지 않는다. `energy_score`가 곧 Jetson board energy ratio라는 뜻은 아니다.
6. **NS-Cache miss-energy 해석에 제한이 있다.** 현재 bundled NS-Cache의 Normal/Fast 경로는 miss dynamic energy를 hit energy와 동일하게 보고하는 구현 경로가 있어 miss-specific energy 차이는 보수적으로 해석해야 한다.
7. **Adaptive unified L1/shared mapping은 비율 기반이며 shared-memory limit는 고정한다.** 예시 NS-Cache L1 template은 modeled instance당 256 KiB이고 Orin Accel-Sim의 adaptive unified L1/shared pool은 SM당 192 KiB다. Pipeline은 baseline 대비 capacity ratio로 simulator geometry와 unified pool을 scale하지만, Orin의 shared-memory option/size는 고정해 CTA occupancy limit를 임의로 바꾸지 않는다. 따라서 scaled unified pool이 164 KiB shared-memory limit보다 작은 L1 후보는 preflight에서 거절된다. 예시 search space는 이 때문에 256 KiB 이상부터 시작한다.
8. **제공된 L2 PPA baseline은 monolithic total-array 모델이다.** Accel-Sim은 4 MiB L2를 16개 subpartition으로 동작시키지만, immutable NS-Cache baseline은 하나의 4 MiB array다. 비교는 supplied SRAM baseline과 같은 방식으로 일관되게 수행하지만, 실제 256 KiB slice 16개의 절대 area/latency/energy와 같다고 볼 수 없다.
9. **성능에 반영되는 technology latency는 hit-latency delta뿐이다.** Candidate write latency와 gain-cell refresh stall/availability는 Accel-Sim timing에 주입하지 않는다. Refresh energy는 포함하지만 refresh로 인한 성능 손실은 빠지므로 특히 gain-cell 결과가 낙관적일 수 있다.

모든 최종 리포트의 `results.json`과 `manifest.json`에는 metric 범위, mapping 가정, input hash, git revision을 남긴다. 논문/보고서에는 CSV 수치뿐 아니라 이 provenance와 위 제한도 함께 보관하는 것이 좋다.

## 권장 실행 순서

1. `kernelslist.g`와 모든 referenced trace가 실제 경로에 있는지 확인한다.
2. local JSON에서 technology model/retention, search space, constraint를 검토한다.
3. `--validate-only`를 통과시킨다.
4. `--max-evaluations 4 --max-workers 1`로 작은 smoke run을 수행한다.
5. log와 per-application counter가 합리적인지 확인한 뒤 전체 budget으로 재실행한다.
6. `optimal_configs.csv`의 `status`를 먼저 확인하고 `report.html`, Pareto/saturation graph, `results.json` provenance를 함께 검토한다.
