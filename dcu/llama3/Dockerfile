FROM image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.1.0-centos7.6-dtk24.04-py310

RUN pip uninstall -y flash-attn && \
  git clone -b v0.1.18 https://mirror.ghproxy.com/https://github.com/InternLM/xtuner.git && \
  cd xtuner && \
  sed -i '1,2d' requirements/runtime.txt && \
  pip install -e '.[all]' mmengine==0.10.3 --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple && \
  cd .. && \
  rm -rf xtuner
