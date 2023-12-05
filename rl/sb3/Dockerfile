FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -yq --no-install-recommends python-opengl xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    stable_baselines3 \
    gymnasium \
    gymnasium[atari] \
    gymnasium[accept-rom-license] \
    gymnasium[mujoco] \
    gymnasium[mujoco_py] \
    moviepy \
    pygame \
    stable-retro \
    opencv-python
