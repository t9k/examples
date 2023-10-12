FROM opendilab/ding:nightly

RUN conda install -y x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge && \
    conda clean -ya
RUN pip install --no-cache-dir torch==1.12.1 torchvision==0.13.1
RUN pip install --no-cache-dir transformers gym_super_mario_bros
