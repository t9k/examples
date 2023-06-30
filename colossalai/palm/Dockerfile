FROM hpcaitech/colossalai:0.3.0

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --default-timeout=100 PaLM-pytorch pytest decorator

WORKDIR /workspace/ColossalAI/examples/language/palm/data
RUN wget http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip && rm enwik8.zip
WORKDIR /workspace/ColossalAI/examples/language/palm
