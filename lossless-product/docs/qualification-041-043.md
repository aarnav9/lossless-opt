# Workload qualification, deployment limits and independent search

These are unreleased source changes and local studies on Apple M2, dated 3 October 2026. The published `v0.1.0a1` assets remain unchanged. No adapter, model/device admission rule or default kernel is promoted by this report.

## 041: broader MLX workloads and a stronger library control

The retained recipe and both optional graph recipes passed **360 request comparisons** across **32 discovery/evaluation profiles**, checking token IDs, every emitted probability array and active KV state bit for bit. Ten of sixteen discovery choices confirmed an improvement on their held-out profiles. Six profiles retained the reference or failed confirmation.

The comparison includes unmodified stock `mlx_lm` batching, not just stock serial execution. Stock batching failed the full bitwise contract in **24 of 32 profiles**, covering every multirequest profile. It is useful diagnostic context, but its different probabilities/state cannot serve as an interchangeable deployment baseline under this contract. Single-request stock batching passed but was slower; the fastest eligible library baseline was serial in every discovery profile. This is not evidence that the optimized recipe beats stock batching under a relaxed contract.

Across 120 stock-batching request comparisons, 112 differed in probabilities/KV and 17 also differed in generated token IDs. Its natural four/eight-request held-out runs were 1.326×/1.614× serial, illustrating the importance of declaring the output contract before choosing a comparator. Merely observing faster batching does not establish interchangeable outputs.

| Held-out profile | Frozen choice | Speed versus eligible library baseline | Confirmed? |
| --- | --- | ---: | --- |
| Synthetic, 1 short / 1 long | Serial | 1.000× | Reference retained |
| Synthetic, 1 mixed | Retained | 1.055× | Yes |
| Synthetic, 2 short / long | Retained | 1.376× / 1.413× | Yes |
| Synthetic, 2 mixed | Serial | 1.000× | Reference retained |
| Synthetic, 4 short / long | Retained | 1.658× / 1.790× | Yes |
| Synthetic, 4 mixed | Serial | 1.000× | Reference retained |
| Synthetic, 8 short / long | Retained | 1.843× / 2.187× | Yes |
| Synthetic, 8 mixed | Full-head 027 | 1.700× | Yes, including the incremental graph gate |
| Natural/code/multilingual, 1 | Decoder 026 | 1.002× | No |
| Natural/code/multilingual, 2 | Retained | 1.059× | Yes |
| Natural/code/multilingual, 4 | Retained | 1.055× | Yes |
| Natural/code/multilingual, 8 | Serial | 1.000× | Reference retained |

The synthetic matrix deliberately controls prefix/output lengths. A supplementary frozen suite uses curated English explanations, code, arithmetic, JSON and multilingual prompts. Neither is a production arrival trace. The natural suite supports much smaller gains than the structured batching fixtures; approximately 2× should not be generalized to arbitrary traffic. The single-request decoder graph's discovery improvement did not repeat, and evaluation was not used to select a replacement.

Each profile has seven interleaved warm repeats per method, exact captured validation, first-use timing, complete-call latency, request TTFT/completion and MLX allocation. Graph caches are bounded to 64 specializations. Seven-sample p95 values here are descriptive, not acceptance guarantees. Discovery and evaluation run in separate processes; selectors and comparators are frozen before evaluation.

[Public evidence](assets/mlx-041.json) contains both complete frozen plans, selections, timing samples, execution orders, correctness records, hardware/runtime/model identities, source hashes and the exact measured harness versions. After the synthetic run, the harness received a closure-binding lint fix and the natural-fixture option; the original measured source remains embedded with its matching hash. These changes did not retroactively alter recorded results.

## 042: deployment constraints control acceptance

MLX jobs now accept optional frozen limits for request p95 TTFT, p95 completion time, warm peak active MLX allocation, minimum tokens per second and break-even calls. Candidates must satisfy these in addition to the existing correctness and speed gates. Payback is checked again using the completed search cost before acceptance. See the [configuration and measurement scope](alpha.md#mlx-deployment-limits-unreleased).

Five policies were declared before launching real workers. The permissive policy accepted the retained implementation; each deliberately impossible policy rejected a fast, correct candidate.

| Frozen policy | Observed value | Result |
| --- | ---: | --- |
| All permissive limits | Final TTFT p95 89.6 ms; completion p95 279.9 ms; 264,442,764 active MLX bytes; 509.2 tokens/s; 145 calls to repay search | Accepted |
| TTFT p95 ≤ 1 µs | 91.3 ms in discovery | Reference retained |
| Active MLX allocation ≤ 1 byte | 264,377,228 bytes in discovery | Reference retained |
| Search payback ≤ 1 call | 23 calls in discovery | Reference retained |
| Throughput ≥ 1 billion tokens/s | 510.9 tokens/s in discovery | Reference retained |

The accepted export preserved its frozen constraints and loaded without fallback. Two different deployment requests again matched tokens, probabilities and KV state. The complete five-policy validation took 85.1 seconds. [Public evidence](assets/constraints-042.json) contains configurations, samples, decisions, the source harness and implementation, plus original record hashes.

Latency gates require at least 20 repeats and use the worst per-request-position p95, rather than pooling short and long requests together. Memory excludes allocator caches, untracked driver allocations and process RSS. These offline checks do not enforce limits on arbitrary future calls; reference fallback does not certify that the reference meets the requested limits. Native deployment constraints remain unsupported and are rejected explicitly.

MLX artifact admission includes adapter source identity. Updating to this worker can cause artifacts from earlier source versions to fall back; optimize and export again to obtain a deployment artifact qualified for the updated runtime.

## 043: independent model authoring

The five-minute pilot returned no complete proposals in three sessions. Simple connectivity and structured-output diagnostics succeeded, but that does not establish why the authoring sessions timed out or how good their unfinished kernels would have been. The pilot remains recorded separately; its final evaluation cases were never opened.

The follow-up freezes a **30-minute cap per author** before three fresh independent `gpt-6-astra` / `xhigh` Codex CLI sessions. Every arm receives three proposal slots and the same numerical softmax contract, compiler settings, six discovery cases and nine held-out cases. Retained recipes and deterministic enumeration are controls. Candidate and timing-block budgets are matched; total authoring/search wall time is reported separately. This is not an equal-wall-time experiment.

Authors receive the discovery problem and retained source, but have no tools, repository/history access, other authors' responses, measurement feedback or held-out cases. No repairs or retries are permitted within an arm. All choices are sealed before final evaluation. Additional paired measurements compare those frozen winners against the independently selected retained winner without reselection. CLI token usage is recorded when returned; billing dollars are unknown rather than estimated from subscription access.

This one-shot experiment measures independent proposal quality. An eight-hour autonomous campaign would answer a different question: whether repeated proposal, measurement and revision outperforms simpler adaptive search at equal wall time and evaluation cost. That needs a separate frozen plan, checkpointed state and untouched final cases.

**Time is a performance tuning input.** More time is likely to help when a short cap prevents useful reasoning or evaluation from completing. This study shows that threshold effect; it does not measure a general monotonic time/speedup curve. Configure total and per-response allowances separately through the [job budget and time overrides](alpha.md#search-time-and-performance-unreleased-cli-overrides).

All three authors finished successfully in **410.4, 607.6 and 441.4 seconds** (6:50, 10:08 and 7:21). Every run exceeded the original five-minute cap. All nine authored candidates compiled and passed all discovery and held-out numerical checks without repairs: 810 candidate/distribution checks for the LLM arm, with no failures. Across all arms and their references, the search recorded 9,720 passing checks; the separate paired replay added 1,296 passing checks. These counts include repeated reference checks, not that many independent workloads.

| Arm | Held-out acceptance versus frozen NumPy comparator | Geomean speed versus NumPy, runs 1/2/3 | Fresh paired speed versus retained, runs 1/2/3 | Total authoring + search/evaluation, three runs |
| --- | --- | --- | --- | ---: |
| Retained recipes | 3/3 | 1.772× / 1.769× / 1.765× | Reference | 42.7 s |
| Enumeration | 1/3 | 1.435× / 1.447× / 1.444× | 0.817× / 0.824× / 0.817× | 43.4 s |
| Independent LLM | 3/3 | 2.133× / 2.123× / 2.115× | **1.210× / 1.206× / 1.207×** | 1,502.6 s |

Acceptance requires at least 1.02× geomean speed and a paired interval lower bound above 1 on every held-out case. Enumeration's positive aggregate versus NumPy therefore does not imply acceptance in every run. All three LLM winners also passed the stronger incremental gate against retained on all nine held-out cases. Their per-case paired speedups ranged 1.175–1.262×, and the smallest interval lower bound was 1.124×. The selected sources use explicit NEON reductions/normalization plus either stride-aware traversal or exact repeated-value detection. This establishes a local candidate-quality gain from these independent sessions, not general model superiority, new mathematics or end-to-end inference acceleration.

LLM authoring cost dominates: accepted implementations per observed minute versus NumPy were 0.120 for LLM, 4.213 for retained and 1.382 for enumeration. Re-selecting a retained recipe is not a new discovery, so this metric must be read alongside incremental speed. Search-only payback for LLM versus retained spans **6.8 million–1.19 billion calls**, depending on shape/run; deployment setup is unmeasured, making these lower bounds. CLI receipts report 35,467 input and 50,281 output tokens in total, including 32,341 reported reasoning-output tokens within the output total. Provider billing remains unknown. The earlier 900-second timeout pilot, study engineering/curation and the shared freeze step are not included in the follow-up arm totals; pilot receipts are retained separately.

Keep the kernels experimental. The study covers one machine, one numerical operator, three shape families, three layouts and six validation distributions; it does not repeat campaign 039's wider adversarial qualification or test complete application requests. No built-in recipe changes. An overnight adaptive study is now more worthwhile, but should be compared against an adaptive non-LLM control at equal elapsed time and measured evaluation cost.

[Public search evidence](assets/search-043.json) retains prompts, controls, all nine model-authored sources, token/time receipts, every search job result, sealed selections, paired samples and the pilot. A payback-reporting lookup initially used `candidate` instead of the worker's `proposal` timing key. The correction reused the nine already completed paired records after checking job identities; it did not rerun measurements, change choices or tune from evaluation. The prior script and bookkeeping amendment are retained. The pilot also needed explicit bookkeeping for zero-proposal arms; it was never used to tune the follow-up from final evaluation data.

## README search-time figure

The [PNG](assets/search-time-043.png) and [SVG](assets/search-time-043.svg) use the slate, teal and lavender palette of the earlier MLX retention figure. They are rendered from the [public campaign 043 evidence](assets/search-043.json), not manually entered timing values. The script validates timeout/completion receipts, all 810 LLM candidate/distribution checks, each selected kernel's paired correctness and timing records, and geometric means before drawing.

Top cards compare author completion under the two caps. Five-minute kernel performance is **unmeasured**, so the figure does not assign it zero speed or a baseline speed. These were separate independent authoring sessions, not interrupted/continued versions of the same run. The 30-minute cap is per author, and no successful author used the full allowance.

Each small point is the ratio of retained/candidate median latency for one held-out shape/layout, measured in 15 paired confirmation blocks. Vertical offsets only separate overlapping points. Diamonds are geometric means across each selected kernel's nine cases; neither the point spread nor the diamonds are confidence intervals. The lavender line marks retained performance at 1.00×. This numerical CPU softmax figure is separate from the exact MLX inference result.

The cost band totals authoring, discovery and evaluation for the three LLM arms, excluding the earlier pilot and study overhead. Payback is search-only, per case, against retained; deployment setup is unmeasured. Regenerating the figure does not rerun the experiment or establish a time/speedup scaling relationship. With the optional plotting dependency installed, render from the repository root:

```sh
python lossless-product/scripts/plot_search_time.py
```

## Reproduction

Run from the repository root with the source package installed, the numerical dependencies for native search, and the pinned MLX extra for GPU work. Set `MODEL` to the previously qualified local SmolLM2 artifact; generic model downloads do not automatically pass admission. Use new output directories for every freeze. Full source, fixtures and rules are in [041](../experiments/mlx_041/suite.py), [042](../experiments/constraints_042/run.py), [043](../experiments/search_043/study.py) and [paired comparison](../experiments/search_043/compare.py).

```sh
set -o pipefail
export PYTHONPATH="$PWD/lossless-product/src"
MODEL="$PWD/research/artifacts/fast_smollm2_mlx_8"
mkdir -p research/results
python -u lossless-product/experiments/mlx_041/suite.py freeze --output research/results/mlx_041_replay
python -u lossless-product/experiments/mlx_041/suite.py discovery --model "$MODEL" --output research/results/mlx_041_replay 2>&1 | tee research/results/mlx_041_replay_discovery.log
python -u lossless-product/experiments/mlx_041/suite.py evaluation --model "$MODEL" --output research/results/mlx_041_replay 2>&1 | tee research/results/mlx_041_replay_evaluation.log
python -u lossless-product/experiments/mlx_041/suite.py freeze --natural --output research/results/mlx_041_natural_replay
python -u lossless-product/experiments/mlx_041/suite.py discovery --model "$MODEL" --output research/results/mlx_041_natural_replay 2>&1 | tee research/results/mlx_041_natural_replay_discovery.log
python -u lossless-product/experiments/mlx_041/suite.py evaluation --model "$MODEL" --output research/results/mlx_041_natural_replay 2>&1 | tee research/results/mlx_041_natural_replay_evaluation.log
python -u lossless-product/experiments/constraints_042/run.py --model "$MODEL" --output research/results/constraints_042_replay 2>&1 | tee research/results/constraints_042_replay.log
```

The MLX phases take roughly 30–90 seconds each on the qualified M2. Independent authoring can take up to 90 minutes total with three sequential 30-minute caps; run this foreground command in Terminal unless asking Codex to execute it. It prints flushed timestamped progress every 30 seconds and result/log paths at startup and completion. It requires an authenticated Codex CLI exposing the selected model and feature flags; the study uses its non-interactive structured-output interface, not the packaged Lossless provider integration.

```sh
python -u lossless-product/experiments/search_043/study.py freeze --author-seconds 1800 --output research/results/search_043_replay
python -u lossless-product/experiments/search_043/study.py author --output research/results/search_043_replay 2>&1 | tee research/results/search_043_replay_author.log
python -u lossless-product/experiments/search_043/study.py discovery --output research/results/search_043_replay 2>&1 | tee research/results/search_043_replay_discovery.log
python -u lossless-product/experiments/search_043/study.py evaluation --output research/results/search_043_replay 2>&1 | tee research/results/search_043_replay_evaluation.log
python -u lossless-product/experiments/search_043/compare.py --output research/results/search_043_replay 2>&1 | tee research/results/search_043_replay_compare.log
```

A numerical softmax study does not establish bitwise model inference gains. Independent sessions still share one model, prompt, machine and narrow workload family. Larger adaptive studies and independent hardware replication remain separate experiments.
