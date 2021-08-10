# basic nvidia image of Ubuntu with CUDA and cuDNN
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

MAINTAINER Marchevskiy Vladislav comratvlad@yandex.ru
LABEL Deep Learning with Pytorch docker image

# timezone
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ==================== install common packages ==================
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  apt-utils \
  sudo \
  python3-pip \
  python3-setuptools \
  cifs-utils \
  nfs-common \
  git \
  subversion \
  htop \
  tmux \
  zsh \
  nano \
  wget \
  curl \
  python-dev \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# =========== add new working user ================
RUN useradd --home-dir /home/mainuser --create-home --shell /bin/zsh --gid root --groups sudo mainuser
RUN echo 'mainuser:userpass' | chpasswd

# ================ install additional python packages =====================
USER mainuser

RUN pip3 install --upgrade pip
COPY other-requirements.txt /home/mainuser/
RUN pip3 install -r /home/mainuser/other-requirements.txt
RUN pip3 install virtualenv setuptools tqdm pillow numpy scikit-learn torch torchvision \
  opencv-contrib-python-headless matplotlib scipy hydra clearml
RUN rm /home/mainuser/other-requirements.txt

# =========== configure zsh and tmux ===========
USER root
RUN apt-get update && apt-get -y install autojump
USER mainuser
RUN git clone https://github.com/zsh-users/antigen.git ~/.antigen

# copy file with zsh settings
COPY zshrc.mainuser /home/mainuser/.zshrc

# copy file with tmux settings
COPY tmux.conf /home/mainuser/.tmux.conf

# ========== install openssh server ===========
USER root
RUN apt-get update && apt-get -y install openssh-server

# ==== set working user and directory ====
USER mainuser

RUN mkdir /home/mainuser/data
RUN mkdir /home/mainuser/src
VOLUME /home/mainuser/data
VOLUME /home/mainuser/src

WORKDIR /home/mainuser
ENV PYTHONPATH=/home/mainuser/src

ENTRYPOINT sudo service ssh restart && zsh
