# Classification Template

## environment

#### Docker build
```
cd docker
sudo docker build -t cls_template:v0 ./
```

#### Docker run
```
export DATA="/path/to/data"
sudo docker run -it --rm --name cls_0\
 --gpus all --shm-size=100g\
 -v $DATA:/data -v $PWD:/workspace\
 -p 8888:8888 -p 6006:6006 --ip=host\
 cls_template:v0
```


# Train

## Create DataFrame

DataFrame should have...

- img_path: str, absolute path to image
- target0~n: int, [0 or 1], your target, you can choose culumns in config yaml file
- cv: int, cross validation

## Augmentaiton

You can select augmentation in train.py.

Please check how the augmented image looks like in the following url;

https://albumentations-demo.herokuapp.com/

## Start training

```
pyton3 exp/exp000/train.py -y exp/exp000/config.yaml
```