# Single-Image-GAN

## Environment
Python 3.10
CUDA 11.8 and cudnn 8.2
pytorch 2.0.1 and torchvision 1.15.2

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
```
python -m pip install -r requirements.txt
```


## Train
To train model on your own image, put the desired training image under Input/Images, and run
```
python main_train.py --input_name wild_bush.jpg
```


## Harmonization
```
python harmonization.py --input_name <training_image_file_name> --ref_name <naively_pasted_reference_image_file_name> --harmonization_start_scale <scale to inject>
```


## Animation
```
python animation.py --input_name <input_file_name> 
```
