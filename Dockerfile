FROM nvcr.io/nvidia/pytorch:22.12-py3

# Create a working directory
WORKDIR /workspace

RUN pip install --upgrade pip

# Install pip requirements
# RUN pip uninstall torch torchaudio torchvision -y

# RUN pip install vllm

# Copy your requirements file into the container
COPY requirements.txt /workspace/requirements.txt

# Install pip requirements
RUN pip install -r requirements.txt

RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

# interactive command line prompt when the container is run
CMD ["bash"]
