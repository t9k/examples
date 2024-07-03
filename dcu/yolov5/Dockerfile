FROM image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.13.1-centos7.6-dtk-23.04.1-py38-latest

RUN source /opt/dtk/env.sh
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY Arial.ttf /root/.config/Ultralytics/Arial.ttf
