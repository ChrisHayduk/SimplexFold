# SimplexFold Experiment Results

Last updated: 2026-05-12.

Best validation C-alpha lDDT so far: **E87**, `val_lddt_ca=0.3992` at step
8500. The target remains `val_lddt_ca > 0.7`, so the goal is not yet met.

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
| E49 directed outer-edge context | completed | 500 | 0.2695 | 0.2695 | 0.2429 | 14.5377 | 6.7858 / 15.4034 | rejected |
| E50 selected boundary expansion hinge | completed | 500 | 0.2731 | 0.2731 | 0.2334 | 14.7809 | 6.6087 / 15.4034 | rejected |
| E51 expansion hinge + structure readout | completed | 250 | 0.2375 | 0.2272 | 0.2233 | 15.7161 | 5.7622 / 15.4034 | rejected |
| E52 selected cell dropout | completed | 500 | 0.2630 | 0.2630 | 0.2301 | 14.2399 | 7.2057 / 15.4034 | rejected |
| E53 longer effective-batch-8 scaffold | completed | 1000 | 0.3480 | 0.3480 | 0.2729 | 12.6378 | 8.5184 / 15.4034 | kept for E54 continuation |
| E54 effective-batch-8 aux anneal | completed | 2000 | 0.3539 | 0.3539 | 0.3241 | 11.9339 | 9.2409 / 15.4034 | kept for E55 continuation |
| E55 effective-batch-8 aux 0.5 continuation | completed | 3000 | 0.3604 | 0.3604 | 0.3451 | 11.3280 | 10.0507 / 15.4034 | current best; branch checkpoint |
| E56 effective-batch-8 aux 0.5 to 4000 | completed | 4000 | 0.3575 | 0.3575 | 0.3478 | 10.9804 | 10.3192 / 15.4034 | stop; improves FoldScore not lDDT |
| E57 aux-0.75 rewarm from E55 | completed | 4000 | 0.3465 | 0.3465 | 0.3495 | 10.7091 | 10.8574 / 15.4034 | rejected; best FoldScore but worse lDDT |
| E58 outer-edge context from E55 weights | stopped early | 3500 | 0.3419 | 0.3419 | 0.3507 | 10.9020 | 11.1250 / 15.4034 | rejected; best FoldScore but worse lDDT |
| E59 damped outer-edge context from E55 | completed | 3500 | 0.3500 | 0.3500 | 0.3516 | 10.9502 | 11.1978 / 15.4034 | rejected; best FoldScore but still below E55 lDDT |
| E60 scheduled damped outer-edge context | completed | 3500 | 0.3462 | 0.3462 | 0.3431 | 10.9235 | 10.8522 / 15.4034 | rejected; scheduling did not preserve E55 lDDT |
| E61 scheduled edge-frame messages | completed | 3500 | 0.3456 | 0.3456 | 0.3471 | 10.7730 | 11.1613 / 15.4034 | rejected; better dRMSD/expansion but worse lDDT |
| E62 scheduled Hodge face residual | completed | 3500 | 0.3468 | 0.3468 | 0.3450 | 10.9016 | 10.7278 / 15.4034 | rejected; boundary lDDT slightly improves but main lDDT remains below E55 |
| E63 selected-boundary lDDT 0.05 | completed | 3500 | 0.3611 | 0.3611 | 0.3576 | 10.6815 | 11.4310 / 15.4034 | kept; new best and improves selected-boundary diagnostics |
| E64 E63 confirmation to 4000 | completed | 4000 | 0.3739 | 0.3739 | 0.3634 | 10.5481 | 11.3344 / 15.4034 | current best; confirms selected-boundary lDDT direction |
| E65 boundary lDDT 0.05->0.025 | completed | 5000 | 0.3684 | 0.3684 | 0.3666 | 10.8445 | 11.7879 / 15.4034 | rejected; relaxed continuation improved FoldScore but lost E64 lDDT |
| E66 coface-balanced boundary lDDT | completed | 4500 | 0.3505 | 0.3505 | 0.3602 | 10.6237 | 11.8892 / 15.4034 | rejected; coface balancing weakened lDDT and boundary diagnostics |
| E67 structure readout 0.05 | completed | 4500 | 0.3647 | 0.3647 | 0.3619 | 10.3503 | 11.6688 / 15.4034 | rejected; dRMSD improved but lDDT stayed below E64 and FoldScore regressed |
| E68 structure readout 0.025 | completed | 4500 | 0.3617 | 0.3617 | 0.3625 | 10.2115 | 11.9645 / 15.4034 | rejected; weaker readout further reduced lDDT despite best dRMSD |
| E69 face normal 0.05 | completed | 4500 | 0.3653 | 0.3653 | 0.3632 | 10.5833 | 11.8750 / 15.4034 | rejected; selected face orientation stayed below E64 and weakened selected-boundary diagnostics |
| E70 edge-frame messages 0.025 | completed | 4500 | 0.3742 | 0.3742 | 0.3653 | 10.3425 | 11.4815 / 15.4034 | kept; tiny new best with improved selected-boundary diagnostics |
| E71 continue edge-frame 0.025 | completed | 5000 | 0.3751 | 0.3751 | 0.3679 | 10.1926 | 11.4483 / 15.4034 | kept; new best lDDT/FoldScore/dRMSD but boundary lDDT softened |
| E72 continue edge-frame 0.025 to 5500 | completed | 5500 | 0.3718 | 0.3718 | 0.3722 | 10.1027 | 12.0872 / 15.4034 | rejected; FoldScore/dRMSD/boundary diagnostics improved but primary lDDT fell below E71 |
| E73 half-scale edge-frame 0.0125 to 5500 | completed | 5500 | 0.3807 | 0.3807 | 0.3720 | 10.0777 | 11.6741 / 15.4034 | kept; previous best lDDT/FoldScore/dRMSD, boundary diagnostics below E72 |
| E74 light geometry selector 0.025 | completed | 6000 | 0.3841 | 0.3841 | 0.3666 | 10.1893 | 11.4266 / 15.4034 | keep as new primary lDDT leader; continue same light-geometry topology-construction branch from E74 |
| E76 continue edge-frame 0.0125 to 6000 | completed | 6000 | 0.3713 | 0.3713 | 0.3723 | 10.2191 | 12.0036 / 15.4034 | rejected; continuing half-scale edge-frame reduced lDDT below E73 despite a tiny FoldScore uptick |
| E77 coface-degree atten 0.25 | completed | 6000 | 0.3733 | 0.3733 | 0.3710 | 10.1286 | 11.8632 / 15.4034 | rejected; improves selected-boundary diagnostics but primary lDDT remains below E73 |
| E78 light geometry selector continue | completed | 6500 | 0.3853 | 0.3853 | 0.3718 | 10.1595 | 11.3783 / 15.4034 | kept; improves E74 lDDT/FoldScore/dRMSD and selected-boundary lDDT, continue light-geometry branch another short gate |
| E80 light geometry selector continue | completed | 7000 | 0.3820 | 0.3820 | 0.3682 | 10.2493 | 11.2472 / 15.4034 | rejected; continuing E78 reduced primary lDDT and selected-boundary lDDT/length, pivot to scheduled sparse-cell complex from E78 |
| E79 scheduled sparse cell caps | completed | 7000 | 0.3885 | 0.3885 | 0.3728 | 10.2661 | 11.1540 / 15.4034 | kept; new best lDDT and much stronger selected-boundary diagnostics, continue sparse-cell branch |
| E82 fixed sparse cell caps | completed | 7500 | 0.3924 | 0.3924 | 0.3788 | 10.2523 | 11.3363 / 15.4034 | kept; fixed sparse-cell complex improves E79 lDDT/FoldScore and selected-boundary diagnostics, continue one short sparse-cell gate |
| E83 fixed sparse cell continue | completed | 8000 | 0.3876 | 0.3876 | 0.3747 | 10.3539 | 11.1757 / 15.4034 | rejected; fixed sparse-cap continuation fell below E82 and softened selected-boundary diagnostics, pivot to E81 degree-penalized cell scoring from E82 |
| E81 degree-penalized sparse cells | completed | 8000 | 0.3980 | 0.3980 | 0.3826 | 10.0954 | 11.4973 / 15.4034 | kept; new best lDDT/FoldScore/dRMSD with lower boundary-edge reuse, continue degree-penalized sparse selector one short gate |
| E84 degree-penalized sparse continue | completed | 8500 | 0.3964 | 0.3964 | 0.3767 | 10.4047 | 11.0245 / 15.4034 | regressed vs E81 at step 8500; launch incidence-normalized E85 from E81 checkpoint |
| E85 incidence-normalized boundary transport | completed | 8500 | 0.3858 | 0.3858 | 0.3767 | 10.1112 | 11.7053 / 15.4034 | rejected; incidence normalization worsened lDDT and selected-boundary diagnostics vs E81 |
| E86 weak directed outer-edge transport | completed | 8500 | 0.3990 | 0.3990 | 0.3858 | 10.0281 | 11.5381 / 15.4034 | kept; tiny new best lDDT and improved dRMSD, continue one short weak outer-edge gate from E86 |
| E91 continue weak directed outer-edge transport | completed | 9000 | 0.3897 | 0.3897 | 0.3820 | 9.9309 | 11.8230 / 15.4034 | rejected; continuing weak outer-edge transport improved dRMSD/selected-boundary diagnostics but primary lDDT fell below E86 |
| E87 directed boundary readout | completed | 8500 | 0.3992 | 0.3992 | 0.3831 | 10.2428 | 11.4322 / 15.4034 | kept; tiny new primary-lDDT best over E86 with directed source/target boundary readout, but FoldScore/dRMSD softened |
| E92 continue directed boundary readout | completed | 9000 | 0.3968 | 0.3968 | 0.3829 | 9.9617 | 11.7362 / 15.4034 | rejected; continuation fell below E87/E86 primary lDDT despite improved dRMSD, pivot to outer-edge-supported cell scoring |
| E90 outer-edge-supported cell scoring | completed | 8500 | 0.3920 | 0.3920 | 0.3783 | 10.0407 | 11.5245 / 15.4034 | rejected; outer-edge-supported scoring fell below E81/E86/E87 primary lDDT |
| E88 runtime-gated latent segment cells | completed | 8500 | 0.3891 | 0.3891 | 0.3824 | 10.1986 | 11.5027 / 15.4034 | rejected; primary lDDT regressed and parameters exceeded AF2-medium +5% budget |
| E89 pair-preserving simplex readout | completed | 8500 | 0.3947 | 0.3947 | 0.3861 | 10.0603 | 11.6927 / 15.4034 | rejected; pair-preserving readout fell below E81/E86/E87 primary lDDT despite stronger FoldScore |
| E93 sparse filtration from E81 | completed | 8500 | 0.3973 | 0.3973 | 0.3819 | 10.2949 | 11.0952 / 15.4034 | rejected; stricter top-k filtration improved selected-boundary diagnostics but primary lDDT fell below E81/E86/E87 |
| E94 moderate filtration + directed boundary readout | completed | 8500 | 0.3914 | 0.3914 | 0.3769 | 10.3028 | 11.3960 / 15.4034 | rejected; combining moderate filtration with directed boundary readout fell below E81/E86/E87 primary lDDT despite lower boundary contraction |
| E95 outer-edge + directed boundary readout | completed | 8500 | 0.3931 | 0.3931 | 0.3817 | 9.9984 | 11.7152 / 15.4034 | rejected; stacking outer-edge context with directed boundary readout improved dRMSD but fell below E86/E87 primary lDDT |
| E96 annealed directed boundary readout | completed | 9000 | 0.4043 | 0.4043 | 0.3852 | 10.1973 | 11.2733 / 15.4034 | keep; annealing directed boundary readout from 0.5 to 0.25 produced a new primary lDDT best over E87 |
| E98 continue partial directed boundary readout | completed | 9500 | 0.3939 | 0.3939 | 0.3807 | 10.0459 | 11.5860 / 15.4034 | rejected; holding partial directed boundary readout at 0.25 fell below E96 primary lDDT, so pivot to queued outer-edge-supported cell scoring |
