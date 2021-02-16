FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git tmux libopencv-dev

RUN pip install tensorboard opencv-python
RUN conda install -y jupyter jupyterlab
RUN conda install -y pandas scikit-learn matplotlib seaborn
RUN pip install albumentations==0.5.2
RUN pip install timm==0.3.4
RUN pip install pytorch-gradcam==0.2.1
RUN pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git