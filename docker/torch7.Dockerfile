ARG TORCH7_PLATFORM=linux/amd64
FROM --platform=$TORCH7_PLATFORM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential cmake curl wget sudo unzip pkg-config \
    gfortran libreadline-dev libjpeg-dev libpng-dev libncurses5-dev \
    python3 python3-pip python3-setuptools python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global url.https://github.com/.insteadOf git://github.com/ \
    && git clone --recursive https://github.com/torch/distro.git /opt/torch \
    && cd /opt/torch \
    && ./install.sh -b

ENV PATH=/opt/torch/install/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/torch/install/lib
ENV LUA_PATH=/opt/torch/install/share/lua/5.1/?.lua;/opt/torch/install/share/lua/5.1/?/init.lua;/opt/torch/install/share/lua/5.2/?.lua;/opt/torch/install/share/lua/5.2/?/init.lua;;
ENV LUA_CPATH=/opt/torch/install/lib/lua/5.1/?.so;/opt/torch/install/lib/lua/5.2/?.so;;
ENV TERM=xterm

RUN for attempt in 1 2 3; do \
        luarocks install dpnn && break; \
        if [ "$attempt" = 3 ]; then exit 1; fi; \
        sleep 5; \
    done

RUN python3 -m pip install --no-cache-dir \
    filelock==3.16.1 \
    fsspec==2025.3.0 \
    jinja2==3.1.6 \
    MarkupSafe==2.1.5 \
    mpmath==1.3.0 \
    networkx==3.1 \
    sympy==1.13.3 \
    typing_extensions==4.13.2 \
    biopython==1.83 \
    chanfig \
    numpy==1.24.4 \
    pyyaml==6.0.2 \
    safetensors==0.5.3

RUN python3 -m pip install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu

RUN ln -s /usr/bin/python3 /usr/local/bin/python

RUN luajit -e "require 'torch'; require 'nn'; require 'dpnn'; print('torch7 ok')" \
    && if luajit -e "require 'torch'; error('torch7 error smoke')"; then exit 1; else test "$?" -eq 1; fi
