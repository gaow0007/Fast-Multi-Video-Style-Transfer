# Fast-Multi-Video-Style-Transfer

## Installation
```
 pytorch >= 1.0
```

## Dataset
```
 wget -c http://vllab1.ucmerced.edu/~wlai24/video_consistency/data/videvo.zip # content data set
 unzip videvo.zip 
 wget -c https://1drv.ms/u/s!AjuMuu6vLknchWxFvJ6uYrNWGk1c # 21 style data set
 wget -c https://1drv.ms/u/s!AjuMuu6vLknchW31I51SXg4u-ygs # 120 style data set
```

## Pretrained Models
```
wget -c https://1drv.ms/u/s!AjuMuu6vLknchWuQAkzPxg9Rzhym # 21 styles + flow + 2lstm
wget -c https://1drv.ms/u/s!AjuMuu6vLknchWo8bw2ZH4kxpaaJ # 120 styles + flow + 2lstm
wget -c https://1drv.ms/u/s!AjuMuu6vLknchXcmTrwjt1CWlQOt # 21 styles + noflow + no2lstm
wget -c https://1drv.ms/u/s!AjuMuu6vLknchXZSfbQdR2OxP-Zy # 120 styles + noflow + no2lstm
download FlowNetS in https://drive.google.com/open?id=0B5EC7HMbyk3CbjFPb0RuODI3NmM  #flownetS
# more details are seen in https://github.com/ClementPinard/FlowNetPytorch

```

## Test
```
sh test_lstm.sh $CUDA_DEVICES
```

