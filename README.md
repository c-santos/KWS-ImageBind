# Keyword Spotting using ImageBind on Google Speech Commands dataset (test split only)
---
Juan Carlo M. Santos

2018-00740

CoE 197Z Deep Learning

Project 2

---


In this project, we enable keyword spotting on the Speech Commands dataset using Meta's ImageBind model.

ImageBind is a model that enables multimodal capabalities (image, audio, text, depth, thermal, and IMU) by training solely on image-paired data to bind the modalities and learn a joint embedding space across the modalities. 

By not training the model on the Speech Commands dataset, we are able to spot keywords in a zero-shot manner.

KWS accuracy with ImageBind was found to be 59.4% with `n_datapoints=64`. Accuracy was computed with the 'multiclass' accuracy function from the `torchmetrics` library. 

Comparison with other KWS SOTA models is found at the end of the notebook.

References:
1. [roatienza/Deep-Learning-Experiments](https://github.com/roatienza/Deep-Learning-Experiments)
2. [facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind)

---
### How to run

This repo serves as a helper repo for the main notebook: `KWS-ImageBind.ipynb`.

To run the code, download **ONLY** the `KWS-ImageBind.ipynb`. The notebook will download this repository for the necessary files to run the program. Keep the `KWS-ImageBind` notebook and the repo in the same folder.

When the notebook is ran, the files will look like the following:


    folder/
    ├── KWS-ImageBind.ipynb             Main notebook
    ├── data                            Speech Commands dataset folder
    └── KWS-ImageBind/
        ├── .assets/
        ├── .checkpoints/
        ├── bpe/
        ├── models/
        ├── data.py
        ├── KWS-ImageBind.ipynb         Unused file
        └── README.md

---
### Comparison to SOTA models that use Speech Commands dataset


| Model Name       | Accuracy | Shot      | Supervision | Link                                                   |
|------------------|----------|-----------|-------------|--------------------------------------------------------|
| TripetLoss-res15 | 98.37%   | Not       | Supervised  | https://arxiv.org/ftp/arxiv/papers/2101/2101.04792.pdf |
| BC-ResNet        | 98.7%    | Not       | Supervised  | https://arxiv.org/pdf/2106.04140v3.pdf                 |
| ImageBind (ours) | 59.4%      | Zero-shot | Unsupervised (see notes below)   | https://arxiv.org/pdf/2305.05665.pdf  |

<sub>data from [paperswithcode.com](https://paperswithcode.com/sota/keyword-spotting-on-google-speech-commands)</sub>


Seen from the table, the large gap with SOTA models is noticeable. However, it is important to note that these models were trained on the Speech Commands dataset, unlike with ImageBind which is performing zero-shot KWS and learns from the joint embedding space (almost self-supervised).


Notes from ImageBind Paper:
- Foundation of ImageBind is a vision transformer.
- "ImageBind unlocks zero-shot classification for modalities without paired text data."
- Self-supervised learning is used in pairing audio, depth, thermal, and IMU with images
- Modalities are paired with images during training. Cross-modality alignment emerges from the embedding space.
- Supervised for image and text pairs (ViT-H from OpenCLIP)
- Supervised for audio and video pairs (AudioSet dataset)
- ImageBind is able to to do zero-shot audio and text classification from its cross modality capabilities i.e training image-text and image-audio pairs.
- For the task of **audio to text retrieval**, ImageBind is **self-supervised**, and ***emergent* zero-shot**
- It is self-supervised since the joint embedding space that it generates is a product of the supervised learning done on image-paired data.
