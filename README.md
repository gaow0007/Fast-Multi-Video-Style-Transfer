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
 wget -c https://1drv.ms/u/s!AjuMuu6vLknchWxFvJ6uYrNWGk1c # 120 style data set
```

## Pretrained Models
```
wget -c https://1drv.ms/u/s!AjuMuu6vLknchWuQAkzPxg9Rzhym # 21 styles
wget -c https://1drv.ms/u/s!AjuMuu6vLknchWo8bw2ZH4kxpaaJ # 120 styles
```

## Test
```
sh test_lstm.sh $CUDA_DEVICES
```

