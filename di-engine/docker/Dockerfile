FROM opendilab/ding:nightly

RUN conda install -y x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge && \
    conda clean -ya
RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchtext==0.14.1
RUN pip install --no-cache-dir transformers gym_super_mario_bros slimevolleygym
COPY slimevolley.py /opt/conda/lib/python3.7/site-packages/slimevolleygym/slimevolley.py
COPY slime_volley_ppo_config.py /opt/conda/lib/python3.7/site-packages/dizoo/slime_volley/config
