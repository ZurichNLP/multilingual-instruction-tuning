# ml-LLM

### Inference Environment Setup With Conda

```
conda create -n vllm_cuda118 -c conda-forge python=3.10 cudatoolkit=11.8 -y
conda activate vllm_cuda118
pip install vllm
pip install -r requirements.txt
```

```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

Download model used for language detection to `resources/lid/`
```
wget https://data.statmt.org/lid/lid201-model.bin.gz -P resources/lid/
gzip -d resources/lid/lid201-model.bin.gz 
```

install lm-eval harness
```
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
git reset --hard 3ccea2b2
pip install -e ".[multilingual]"
```
