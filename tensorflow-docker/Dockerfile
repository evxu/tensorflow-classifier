FROM ubuntu:16.04

MAINTAINER Evelyn Xueqi Wang <evelyn.wang@image-data.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        wget \
	imagemagick \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install following prior to dlib and picpac
RUN apt-get -y update &&\
    apt-get install -y --fix-missing \
    build-essential \
    cmake \
    git \
    gfortran \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python-dev \
    python-numpy \
    python-protobuf\
    software-properties-common \
    python-setuptools \
    libgoogle-glog-dev \
    libopencv-dev \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

# Install dlib
RUN cd ~ && \
    mkdir -p dlib && \
    git clone https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python setup.py install --yes USE_AVX_INSTRUCTIONS

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        Flask \
        Jinja2 \
        tornado \
        Django \
        scikit-image \
        scikit-learn \
        opencv-python \
        simplejson  \
        requests \
        leveldb \
        redis 


RUN pip install -i https://testpypi.python.org/pypi picpac


# Install TensorFlow CPU version from central repo
RUN pip --no-cache-dir install tensorflow


# TensorBoard
EXPOSE 6006

# IDS API
EXPOSE 8000