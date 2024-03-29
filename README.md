# TryYours - Virtual Try-On

Welcome to TryYours, a Virtual Try-On project that leverages cutting-edge machine learning techniques to enable users to try on clothes virtually. This guide will walk you through setting up and running the project in Google Colab, a free cloud service supported by Google that allows you to run your Jupyter notebooks.

## Getting Started

### 1. Setup

Before you begin, ensure your Colab environment is configured to use a GPU for better performance:

- Go to the `Runtime` menu.
- Click on `Change runtime type`.
- Set `Hardware Accelerator` to `GPU`.

Next, clone the repository and install the necessary dependencies by running the following commands in a new cell:


# Clone repository
!git clone https://github.com/lastdefiance20/TryYours-Virtual-Try-On

# Install dependencies
!pip install tensorboardX av torchgeometry flask flask-ngrok iglovikov_helper_functions cloths_segmentation albumentations
!pip install scipy==1.8.0

%cd TryYours-Virtual-Try-On

# Install detectron2
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

2. Download Pre-trained Models
Download the necessary pre-trained models for the project to function:

!pip install --upgrade --no-cache-dir gdown

# Download HR-VITON model
%cd HR-VITON-main
!gdown https://drive.google.com/u/0/uc?id=1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy&export=download
!gdown https://drive.google.com/u/0/uc?id=1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ&export=download

%cd ../

# Download Graphonomy model
%cd Graphonomy-master
!gdown https://drive.google.com/u/0/uc?id=1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9&export=download

%cd ../

3. Upload Cloth Images
Upload the cloth images you want to use. You can either use sample images provided or upload your own:

import os
import shutil
from google.colab import files


input_dir = 'static'
uploaded = files.upload()
for filename in uploaded.keys():
  input_path = os.path.join(input_dir, filename)
  shutil.move(filename, input_path)
os.remove(input_dir+'/cloth_web.jpg')
os.rename(input_path, input_dir+'/cloth_web.jpg')

4. Upload Person Images
Similarly, upload the person images onto which the clothes will be tried:


input_dir = 'static'
uploaded = files.upload()
for filename in uploaded.keys():
  input_path = os.path.join(input_dir, filename)
  shutil.move(filename, input_path)
os.remove(input_dir+'/origin_web.jpg')
os.rename(input_path, input_dir+'/origin_web.jpg')


import matplotlib.pyplot as plt
import cv2

original = cv2.cvtColor(cv2.imread("./static/origin_web.jpg"), cv2.COLOR_BGR2RGB)
cloth = cv2.cvtColor(cv2.imread("./static/cloth_web.jpg"), cv2.COLOR_BGR2RGB)

# Display Images
fig, axes = plt.subplots(nrows=1, ncols=2)
dpi = fig.get_dpi()
fig.set_size_inches(900 / dpi, 448 / dpi)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
axes[0].axis('off')
axes[1].axis('off')
axes[0].imshow(original)
axes[1].imshow(cloth)
plt.show()

!python main.py # Add --background False to remove background

7. View Results
After processing, view the result with the following code:

from PIL import Image
from IPython.display import Image

This README.md file is designed to guide users


# Sources and CP Viton Data Set
@InProceedings{Minar_CPP_2020_CVPR_Workshops,
	title={CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On},
	author={Minar, Matiur Rahman and Thai Thanh Tuan and Ahn, Heejune and Rosin, Paul and Lai, Yu-Kun},
	booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {June},
	year = {2020}
}


