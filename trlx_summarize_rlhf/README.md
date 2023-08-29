## RLHF Pipeline - trlx

Before running everything, we need some extra packages not included in the `trlx` dependency list. Specifically, we need HuggingFace's [`evaluate`](https://huggingface.co/docs/evaluate/index) package and Google's re-implementation of ROUGE, [`rouge-score`](https://github.com/google-research/google-research/tree/master/rouge). To install them, run `requirements.txt` in this example's root directory:

```bash
pip install -r requirements.txt
```

### Training Process


1. Train SFT:
    ```bash
    deepspeed sft/train_gptneo_summarize.py
    ```

2. Train Reward Model:
    ```bash
    deepspeed reward_model/train_reward_model_gptneo.py
    ```

3. PPO training:
    ```bash
    accelerate launch --num_processes 3 --config_file configs/default_accelerate_config.yaml trlx_gptneo_text_summarization.py
    ```
    
    After PPO training, model param need to transform from deepspeed ZeRO format to normal
    ```
    python ckpts/best_checkpoint/zero_to_fp32.py ckpts/best_checkpoint/best_checkpoint ckpts/best_checkpoint/best_checkpoint/pytorch_model.bin
    ```

4. Inference
    Default Inference Will use GPU0 to load reward model ( GPT Neo 125m ) and GPU1 to do inference ( GPT Neo 1.3B )
    ```
    python trlx_inference_gptneo.py 
    ```

### Results

The following tables display ROUGE and reward scores on the test set of the TL;DR dataset between SFT and PPO models. ( Actor(SFT): GPT Neo 1.3B, Critic(RM): GPT Neo 125m )

1. SFT vs PPO

    __ROUGE scores__

    | Model | Rouge-1 | Rouge-2 | Rouge-L |
    | --- | --- | --- | --- |
    | SFT | 0.32 | 0.112 | 0.249 | 
    | PPO | 0.298 | 0.101 | 0.231 |

    __Reward scores__
    Ground Truth is 3.158
    | Model | Average Reward | Reward $\Delta$ |
    | --- | --- | --- |
    | SFT | 3.234 | +0.076 |
    | PPO | 3.457 | +0.299 |


2. Examples of generated summaries can be found in 'rlhf/metadata/ppo_with_reward_scores_sample.csv' & 'rlhf/metadata/sft_with_reward_scores_sample.csv'

## References

1. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)", Neural Information Processing Systems, 2020.
