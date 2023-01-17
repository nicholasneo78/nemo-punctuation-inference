#use the base package for from official pytorch source 
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get update && apt-get install -y build-essential libsndfile1 git sox wget ffmpeg sudo \
&& rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Upgrade the pip
RUN pip install --upgrade pip

# Install pip requirements
RUN rm -rf $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/ruamel*

# upgrade the python to 3.8.10
RUN apt update -y && sudo apt upgrade -y && \
    apt-get install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    sudo wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz && \
    sudo tar xzf Python-3.8.10.tgz && \
    cd Python-3.8.10 && \
    sudo ./configure --enable-optimizations && \
    sudo make altinstall

# set python 3.8 as the default python
SHELL ["/bin/bash", "-c"]

# RUN sudo ln -s /usr/local/bin/python3.8 /usr/local/bin/python3
RUN printf "\n%s\n" "alias python3=python3.8" >> ~/.bashrc
RUN source ~/.bashrc

# # numpy issue
# RUN pip install numpy==1.21.6 --no-binary numpy

# build nemo toolkit from source
RUN python3.8 -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[nlp]

# get the dependencies and install torchaudio
ADD requirements.txt .
RUN python3 -m pip install torchaudio==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# installing jupyter lab inside
RUN pip install jupyterlab

# installing tensorboardX
RUN pip install tensorboard --no-cache-dir
RUN pip install tensorboardX --no-cache-dir

# declare port used by jupyterlab
EXPOSE 8888

# set default command for jupyterlab
CMD ["jupyter" ,"lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]

#docker container starts with bash
WORKDIR /nemo_punctuation_inference
RUN ["bash"]