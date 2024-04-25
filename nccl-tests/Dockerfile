FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-7 \
        git \
        curl \
        vim \
        wget \
        apt-utils \
        ca-certificates \
        iproute2 \
        iputils-ping \
        net-tools \
        ethtool \
        pciutils \
        libnl-route-3-200 \
        kmod \
        lsof \
        libcudnn8 \
        libnccl2 \
        libnccl-dev \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.3.tar.gz && \
    tar zxf openmpi-4.1.3.tar.gz && \
    cd openmpi-4.1.3 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# Install NCCL-Test step
WORKDIR /nccl_tests
RUN wget -q -O - https://github.com/NVIDIA/nccl-tests/archive/master.tar.gz | tar --strip-components=1 -xzf - \
    && make MPI=1

ENV LD_LIBRARY_PATH /usr/local/lib:/usr/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
