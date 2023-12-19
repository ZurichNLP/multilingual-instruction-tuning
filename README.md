This repository contains the code and data for the paper "Turning English-centric LLMs into Polyglots: How Much Multilinguality is Needed?"

# Environment Setup

To setup the environment, we recommend using conda, e.g.:

```
conda create -n ml_llm -c conda-forge python=3.10 cudatoolkit=11.8 -y
conda activate ml_llm
pip install vllm
pip install -r requirements.txt
```

Download model used for language detection to `resources/lid/`

```
wget https://data.statmt.org/lid/lid201-model.bin.gz -P resources/lid/
gzip -d resources/lid/lid201-model.bin.gz 
```

For evaluations using [Eleuther AI's LM Evaluation Harness](<https://github.com/EleutherAI/lm-evaluation-harness>), run:

```
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
git reset --hard 3ccea2b2
pip install -e ".[multilingual]"
```

# API Keys

If running experiments with OpenAI's API-based models, create a file containing your API key, e.g.:

```
echo "OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'" > api_secrets.py
```
