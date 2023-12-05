FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -yq --no-install-recommends python-opengl xvfb build-essential swig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    stable_baselines3 \
    gymnasium \
    gymnasium[atari] \
    gymnasium[accept-rom-license] \
    gymnasium[box2d] \
    gymnasium[mujoco] \
    moviepy \
    pygame \
    stable-retro \
    opencv-python \
    tensorboard
