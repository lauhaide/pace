* GRPO-R1
```
   --prompt_type "ours" \
   --naive_grpo \
```
* TIRESRAG
```
   --prompt_type "ours" \
```
* MT-GRPO
```
   --prompt_type "ours" \
   --naive_grpo \
   --reward_shape \
   --reshape_method "mtgrpo"\
   -ir nite \
   -ir gola
```
* LeTS
```
   --prompt_type "ours" \
   --naive_grpo \
   --reward_shape \
   --reshape_method "sgrpo" \
   -ir rlredun \
   -ir glmatch
```
* ΦMT-GRPO
```
   --prompt_type "trajel3" \
   --naive_grpo \
   --reward_shape \
   --reshape_method "mtgrpo" \
   --diversity_level "step" \
   --approx_coverage $DATA_PATH_ACOV \
   --adaptive \
   --deep_penalty 0.6 \
   --with_process_r \
   -ir rets \
   -ir acov \
   -ir sdiv
```
* ΦLeTS
```
   --prompt_type "trajel3" \
   --naive_grpo \
   --reward_shape \
   --reshape_method "sgrpo" \
   --approx_coverage $DATA_PATH_ACOV \
   --adaptive \
   --deep_penalty 0.6 \
   --with_process_r \
   --with_group_d \
   -ir rets \
   -ir acov \
   -ir sdiv
```
* GRPO_avgΦ
```
   --prompt_type "trajel3" \
   --naive_grpo \
   --reward_shape \                
   --reshape_method "average" \   
```   
* Naive-GRPO
```
   --prompt_type "trajel3" \
   --naive_grpo \
```   

Note that some flags such as the ones below can still be on despite not being used by the algorithm, this can be done to track behaviour for baselines:
```
   --diversity_level "step" \
   --approx_coverage $DATA_PATH_ACOV \
```

LeTS is our re-implementation of the algorighm in [(Zhang et al., 2025)](https://aclanthology.org/2025.emnlp-main.257/) as there was not code release. We also implemented MT-GRPO following [(Xie et al., 2026)](https://openreview.net/forum?id=eBMOr6a84z).