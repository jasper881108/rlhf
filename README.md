## RLHF

<br> 

### RLHF Task 
#### This example shows how to use `Colossal AI` & `trlx` to train a summarization model using human feedback
#### following the fine-tuning procedures described in Stiennon et al.'s, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)".

<br> 

#### Other two task need further update
1. [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/pdf/2112.09332.pdf)
2. [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)

<br> 

### RLHF pipeline
#### Currently suggest to use trlx ( 2023/4/12 )
1. [HPC AI TECH/Colossal AI](https://github.com/hpcaitech/ColossalAI)
2. [Carper AI/trlx](https://github.com/CarperAI/trlx)

<br> 

### Structure
1. colossalai_summarize_rlhf: Store Colossal AI method of summarization using rlhf
2. model: Only save Colossal AI's model ( Because it support [LoRA](https://github.com/microsoft/LoRA), saving a lot memory )
3. trlx_summarize_rlhf: Store trlx method of summarization using rlhf
4. openai_api: Helper script to call openai_api, you should put your personal api key in 'rlhf/openai_api/openai_api.txt'
5. metadata: Store summarization result