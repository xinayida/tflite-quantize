# 'tflite-quantize'
A demo of training tensorflow model with  quantization-aware training and convert to tflite format.

## Environment
python 3x
tensorflow 1.13.1

## Run

1. train model:  
`python train.py`
2. freeze model:  
`python freeze.py`
3. convert model:  
`cd out && ./convert.sh`
