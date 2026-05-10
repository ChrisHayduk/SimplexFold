# SimplexFold Experiment Results

Last updated: 2026-05-10.

Best validation C-alpha lDDT so far: **E15**, `val_lddt_ca=0.3556` at step
9000. The target remains `val_lddt_ca > 0.7`, so the goal is not yet met.

This file records only returned Runpod results. In-flight plans, launch notes,
and partial diagnostics belong in `EXPERIMENTS_NOTES.md` until a run returns a
final or early-stop validation point.

`Final/stop` means the final checkpoint for completed runs or the early-stop
validation checkpoint for rejected pilot runs. `-` means the metric was not
recorded in the running notes for that run.

| Run | Status | Best step | Best `val_lddt_ca` | Final/stop `val_lddt_ca` | Final/stop FoldScore | Final/stop `val_ca_drmsd` | Final/stop C-alpha Rg | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| main/full control | completed | 400 | 0.0992 | 0.0401 | 0.2024 | - | - | baseline |
| E01 balanced contact | completed | 400 | 0.1096 | 0.0316 | 0.1996 | - | - | rejected |
| E02 topology neighborhood | completed | 800 | 0.1127 | 0.0281 | 0.1944 | - | - | rejected |
| E03 warm boundary | completed | 800 | 0.1311 | 0.0742 | 0.2132 | - | - | kept for scaled follow-up |
| E03 warm boundary scaled | completed | 1000 | 0.1219 | 0.1009 | 0.2346 | - | - | rejected |
| E04 coordinate cells diagnostic | completed | 1000 | 0.2200 | 0.2200 | 0.2422 | 14.9499 | 5.3194 / 14.6867 | kept for scaled follow-up |
| E04 coordinate cells scaled | completed | 1500 | 0.2394 | 0.1985 | 0.2399 | 16.7843 | 4.8384 / 15.7622 | rejected |
| E05 coordinate weights 0.5 scaled | completed | 3000 | 0.2948 | 0.2948 | 0.2647 | 14.9898 | 6.5441 / 15.7622 | kept |
| E06 coordinate weights 1.0 scaled | completed | 3000 | 0.3127 | 0.3127 | 0.2511 | 14.5496 | 7.1388 / 15.7622 | kept |
| E07 boundary coordinate d=0.5 scaled | completed | 2000 | 0.3247 | 0.3187 | 0.2617 | 12.9483 | 9.1383 / 15.7622 | kept |
| E08 boundary coordinate d=1.0 scaled | stopped early | 500 | 0.2636 | 0.2636 | 0.2386 | 15.1396 | 6.0681 / - | rejected |
| E09 full MSA-to-face d=0.5 scaled | completed | 3000 | 0.3429 | 0.3429 | 0.2689 | 12.9189 | 8.6544 / 15.7622 | kept |
| E10 warm MSA-to-face initializer | stopped early | 500 | 0.2232 | 0.2232 | 0.2190 | 12.3197 | 10.5325 / - | rejected |
| E11 long-range topology bias | stopped early | 500 | 0.2288 | 0.2288 | 0.2244 | 15.2809 | 6.0858 / - | rejected |
| E12 E09 continuation to 6000 | completed | 5000 | 0.3472 | 0.3449 | 0.2856 | 11.7918 | 9.8828 / 15.7622 | kept |
| E13 mixed local/global selector | stopped early | 500 | 0.2371 | 0.2371 | 0.2238 | 15.3413 | 6.2290 / - | rejected |
| E14 soft mixed selector | completed | 2000 | 0.3264 | 0.3015 | 0.2589 | 12.1838 | 10.1755 / 15.7622 | rejected |
| E15 simplex aux anneal to 0.5 | completed | 9000 | 0.3556 | 0.3556 | 0.3025 | 12.3527 | 9.0217 / 15.7622 | current best |
| E16 deeper aux anneal to 0.25 | stopped early | 9500 | 0.3506 | 0.3438 | - | - | - | rejected |
| E17 continue E15 aux at 0.5 | stopped early | 9500 | 0.3554 | 0.3441 | - | - | - | rejected |
| E18 simplex-only topology capacity | completed | 3000 | 0.3350 | 0.3350 | 0.2655 | 13.4524 | 8.1784 / 15.7622 | rejected |
| E19 selected boundary lDDT 0.25 | stopped early | 500 | 0.2832 | 0.2832 | 0.2448 | 14.4789 | 7.1624 / 15.4034 | rejected |
| E20 selected boundary lDDT 0.05 | stopped early | 500 | 0.2364 | 0.2364 | 0.2447 | 15.5881 | 5.6076 / 15.4034 | rejected |
| E21 strong simplex messages | stopped early | 500 | 0.2315 | 0.2315 | 0.2328 | 15.1343 | 6.3715 / 15.4034 | rejected |
| E22 damped simplex messages | stopped early | 500 | 0.2917 | 0.2917 | 0.2458 | 14.4541 | 6.6487 / 15.4034 | rejected |
| E23 edge-biased simplex messages | stopped early | 500 | 0.2509 | 0.2509 | 0.2355 | 15.0561 | 6.2181 / 15.4034 | rejected |
| E24 degree-normalized boundary realization | stopped early | 500 | 0.2724 | 0.2724 | 0.2383 | 14.1528 | 7.2673 / 15.4034 | rejected |
| E25 effective-batch-8 optimization | completed | 500 | 0.2946 | 0.2946 | 0.2466 | 14.3073 | 7.6818 / 15.7622 | rejected |
| E26 MSA-to-face two-skeleton | completed | 250 | 0.2517 | 0.2489 | 0.2214 | 15.8143 | 5.9651 / 15.7622 | rejected |
| E27 no recycled-coordinate topology | completed | 500 | 0.2369 | 0.2369 | 0.2354 | 16.3061 | 5.7967 / 15.7622 | rejected |
| E28 topology teacher forcing 1.0->0.0 | completed | 500 | 0.2398 | 0.2398 | 0.2222 | 15.5485 | 6.1752 / 15.7622 | rejected |
| E29 soft topology teacher forcing 0.25->0.0 | completed | 500 | 0.2451 | 0.2451 | 0.2169 | 15.4451 | 6.7226 / 15.7622 | rejected |
| E30 simplex coupling warmup 0.0->1.0 | completed | 500 | 0.2854 | 0.2854 | 0.2405 | 13.9247 | 8.9047 / 16.3091 | rejected |
| E31 damped simplex coupling warmup 0.0->0.5 | completed | 500 | 0.2578 | 0.2578 | 0.2332 | 14.7889 | 8.9024 / 16.3091 | rejected |
| E32 topology-plus aux anneal | stopped early | 250 | 0.2545 | 0.2545 | 0.2059 | 14.2821 | 7.2877 / 15.4034 | rejected |
| E33 simplicial structure readout | stopped early | 250 | 0.2405 | 0.2405 | 0.2108 | 14.8467 | 6.9826 / 15.4034 | rejected |
| E34 readout-only simplicial sidecar | stopped early | 250 | 0.2426 | 0.2426 | 0.2103 | 14.9311 | 6.5743 / 15.4034 | rejected |
| E35 face-only structure sidecar | stopped early | 250 | 0.2406 | 0.2406 | 0.2062 | 13.0352 | 9.1316 / 15.4034 | rejected |
| E36 topology margin selector | stopped early | 250 | 0.1286 | 0.1286 | 0.1857 | 13.5268 | 13.1096 / 15.4034 | rejected |
| E37 selected face normal orientation | stopped early | 250 | 0.2464 | 0.2464 | 0.2109 | 14.9943 | 6.4679 / 15.4034 | rejected |
| E38 selected simplex shape realization | stopped early | 250 | 0.2402 | 0.2402 | 0.2113 | 14.9614 | 6.6367 / 15.4034 | rejected |
| E39 outer-edge cell communication | stopped early | 250 | 0.2460 | 0.2460 | 0.2163 | 14.7805 | 6.7531 / 15.4034 | rejected |
| E40 edge-frame scalarized messages | stopped early | 250 | 0.2350 | 0.2350 | 0.2139 | 15.2338 | 6.3502 / 15.4034 | rejected |
| E41 latent rank-2 segment cells | stopped early | 250 | 0.2393 | 0.2393 | 0.2125 | 15.2012 | 6.2747 / 15.4034 | rejected |
| E42 damped Hodge face residual | stopped early | 250 | 0.2545 | 0.2545 | 0.2112 | 14.7096 | 6.7897 / 15.4034 | rejected |
| E43 Hodge residual aux anneal | completed | 500 | 0.2492 | 0.2492 | 0.2232 | 15.1139 | 6.1772 / 15.4034 | rejected |
| E44 soft flag-complex closure | completed | 250 | 0.2449 | 0.2111 | 0.2241 | 16.1468 | 5.0536 / 15.4034 | rejected |
| E45 light flag-complex closure | completed | 250 | 0.2477 | 0.2273 | 0.1992 | 14.9228 | 7.3539 / 15.4034 | rejected |
| E46 expanded selected complex | completed | 250 | 0.2517 | 0.2327 | 0.2215 | 15.5059 | 5.7840 / 15.4034 | rejected |
| E47 auxiliary flag-closure curriculum | completed | 250 | 0.2466 | 0.2262 | 0.2182 | 15.7332 | 5.5581 / 15.4034 | rejected |
| E48 adaptive local-to-global topology curriculum | completed | 500 | 0.2274 | 0.2274 | 0.2191 | 15.7749 | 5.5326 / 15.4034 | rejected |
