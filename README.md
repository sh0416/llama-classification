# Text classification using LLaMA

This repository provides a basic codebase for text classification using LLaMA.

## What system do I use for development?

* Device: Nvidia 1xV100 GPU
* Device Memory: 34G
* Host Memory: 252G

If you need other information about hardware, please open an issue.

## How to use

### Experimental setup

1. Get the checkpoint from official LLaMA repository from [here](https://github.com/facebookresearch/llama).  
    1-1. I assume that the checkpoint would be located in the project root direction and the contents would be arranged as follow.
    ```text
    checkpoints
    ├── llama
    │   ├── 7B
    │   │   ├── checklist.chk
    │   │   ├── consolidated.00.pth
    │   │   └── params.json
    │   └── tokenizer.model
    ```

2. Prepare your python environment. I recommend using anaconda to segregate your local machine CUDA version.
    ```bash
    conda create -y -n llama-classification python=3.8
    conda activate llama-classification
    conda install cudatoolkit=11.7 -y -c nvidia
    conda list cudatoolkit # to check what cuda version is installed (11.7)
    pip install -r requirements.txt
    ```

### Method: Direct

`Direct` is to compare the conditional probability `p(y|x)`.

1. Preprocess the data from huggingface datasets using the following scripts. From now on, we use the ag_news dataset.
    ```bash
    python run_preprocess_direct_ag_news.py
    python run_preprocess_direct_ag_news.py --sample=False # Use it for full evaluation
    python run
    ```

2. Inference to compute the conditional probability using LLaMA and predict class.
    ```bash
    torchrun --nproc_per_node 1 run_evaluate_direct_llama.py \
        --data_path samples/inputs_direct_ag_news.json \
        --output_path samples/outputs_direct_ag_news.json \
        --ckpt_dir checkpoints/llama/7B \
        --tokenizer_path checkpoints/llama/tokenizer.model
    ```

### Method: Channel

`Channel` is to compare the conditional probability `p(x|y)`.

1. Preprocess the data from huggingface datasets using the following scripts. From now on, we use the ag_news dataset.
    ```bash
    python run_preprocess_channel_ag_news.py
    python run_preprocess_channel_ag_news.py --sample=False # Use it for full evaluation
    python run
    ```

2. Inference to compute the conditional probability using LLaMA and predict class.
    ```bash
    torchrun --nproc_per_node 1 run_evaluate_channel_llama.py \
        --data_path samples/inputs_channel_ag_news.json \
        --output_path samples/outputs_channel_ag_news.json \
        --ckpt_dir checkpoints/llama/7B \
        --tokenizer_path checkpoints/llama/tokenizer.model
    ```

### Method: Pure generation

1. To evaluate using `generate` mode, you can use the preprocessed direct version.
    ```bash
    torchrun --nproc_per_node 1 run_evaluate_generate_llama.py \
        --data_path samples/inputs_direct_ag_news.json \
        --output_path samples/outputs_generate_ag_news.json \
        --ckpt_dir checkpoints/llama/7B \
        --tokenizer_path checkpoints/llama/tokenizer.model
    ```

## Todo list

- [x] Implement channel method
- [ ] Implement other calibration method
- [ ] Support other dataset inside the huggingface datasets
- [ ] Other evaluation metric to measure the different characteristic of foundation model (LLaMA)
- [ ] Experimental report

## Final remark

- I am really appreciate for the LLaMA project team to publish a checkpoint and their efficient inference code. Much of work in this repository is done based on [the official repository](https://github.com/facebookresearch/llama). 
- For the reader, don't hesitate to open issue or pull requests. You can give me..
  - Any issue about other feature requests
  - Any issue about the detailed implementation
  - Any discussion about the research direction

## Citation

It would be welcome citing my work if you use my codebase for your research.

```
@software{Lee_Simple_Text_Classification_2023,
    author = {Lee, Seonghyeon},
    month = {3},
    title = {{Simple Text Classification Codebase using LLaMA}},
    url = {https://github.com/github/sh0416/llama-classification},
    version = {1.0.0},
    year = {2023}
}
```