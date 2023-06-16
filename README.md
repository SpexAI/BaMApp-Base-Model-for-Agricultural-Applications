# BaMApp Base Model for Agricultural Applications

The BaMApp Base Model for Agricultural Applications repository aims to provide the resources and guidance necessary for fine-tuning the DinoV2 model on a dataset specific to the AgTech sector. This model will be trained on RGB images of plants in various conditions: field, greenhouse, and indoor, as well as data from high throughput phenotyping facilities.

All data will be hosted on [DeepLake](https://github.com/activeloopai/deeplake), an open-source data lake for machine learning datasets. 

## Table of Contents

1. [Motivation](#motivation)
2. [Background](#background)
3. [Usage and Upload](#usage-and-upload)
4. [Datasets](#datasets)
5. [Trained Model](#trained-model)

## Motivation

The AgTech sector has unique data requirements and challenges that can greatly benefit from a foundation model specifically designed and fine-tuned for its use-cases. By providing this foundation model, we aim to advance research and development in the AgTech sector, facilitate phenotyping cases, and pave the way for more advanced, domain-specific models in the future. 

## Background

### DinoV2

The [DinoV2](https://github.com/facebookresearch/dinov2) model, developed by Facebook AI, is a computer vision model trained using self-supervised learning. This model represents a significant advancement in the field of computer vision and provides a strong foundation for further fine-tuning on domain-specific data. For more information, visit the [DinoV2 Blog Post](https://ai.facebook.com/blog/dino-v2-computer-vision-self-supervised-learning/).

### DeepLake

[DeepLake](https://github.com/activeloopai/deeplake) is an open-source project by ActiveLoop that provides a data lake for machine learning datasets. It allows users to store, share, and collaborate on large-scale datasets in an efficient and straightforward manner.

## Usage and Upload
1. Clone the repo.
2. Create a virtual environment and install the requirements
`pip install requirements.txt`
3. Login to DeepLake see [here](https://docs.activeloop.ai/getting-started/deep-learning/using-activeloop-storage) for instructions.
4. Upload the images to DeepLake using the upload.py script. The script takes in the following arguments:

```
usage: upload.py [-h] [--folder FOLDER] [--commit_message COMMIT_MESSAGE] [--json JSON]

Upload images to deeplake

options:
  -h, --help            show this help message and exit
  --folder FOLDER       Folder with images
  --commit_message COMMIT_MESSAGE
                        Commit message
  --json JSON           Json file formated Metadata eg. {"Origin": "Test"}

```

An example call would be:
```
python upload.py --folder ./images --commit_message "Test" --json '{"Origin": "Test", "Description": "Test"}'
```
5. Report the commit ID to the BaMApp team, to keep track of the data.

## Datasets

| Dataset Description | Commit ID | Origin of Data |
| ------------------- | --------- | -------------- |
| *Placeholder*       | *Placeholder* | *Placeholder* |

## Trained Model

We aim to train the BaMApp Base Model for Agricultural Applications and release it in Q3/4 2023. Please stay tuned for updates.

## Future Directions

At present, the BaMApp Base Model for Agricultural Applications only supports RGB images without any labels. In the future, we plan to include support for other modalities. 
