# basic nvidia image of Ubuntu with CUDA and cuDNN
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

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
  python-pip \
  python3-pip \
  python-setuptools \
  python3-setuptools \
  cifs-utils \
  nfs-common \
  git \
  subversion \
  tortoisehg \
  ipython \
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
RUN useradd --home-dir /home/researcher --create-home --shell /bin/zsh --gid root --groups sudo researcher
RUN echo 'researcher:deeppass' | chpasswd

# ================ install additional python packages =====================
USER researcher

RUN pip3 install --upgrade pip
COPY other-requirements.txt /home/researcher/
RUN pip3 install --file /home/researcher/other-requirements.txt
RUN pip3 install virtualenv setuptools tqdm pillow numpy scikit-learn torch torchvision \
  opencv-contrib-python-headless matplotlib scipy
RUN rm /home/researcher/other-requirements.txt

# =========== configure zsh and tmux ===========
USER root
RUN apt-get update && apt-get -y install autojump
USER researcher
RUN git clone https://github.com/zsh-users/antigen.git ~/.antigen

# copy file with zsh settings
COPY zshrc.researcher /home/researcher/.zshrc

# copy file with tmux settings
COPY tmux.conf /home/researcher/.tmux.conf

# ========== install openssh server ===========
USER root
RUN apt-get update && apt-get -y install openssh-server

# ==== set working user and directory ====
USER researcher

RUN mkdir /home/researcher/data
RUN mkdir /home/researcher/ssd_data
RUN mkdir /home/researcher/src
VOLUME /home/researcher/data
VOLUME /home/researcher/ssd_data
VOLUME /home/researcher/src

WORKDIR /home/researcher
ENV PYTHONPATH=/home/researcher/src

ENTRYPOINT sudo service ssh restart && zsh
