# Beyond-supervision-Harnessing-self-supervised-learning-in-unseen-plant-disease-recognition

## Abstract
[[Paper]](https://www.sciencedirect.com/science/article/pii/S0925231224013791)

Deep learning models have demonstrated great promise in plant disease identification. However, existing approaches often face challenges when dealing with unseen crop-disease pairs, limiting their practicality in real-world settings. This research addresses the gap between known and unknown (unseen) plant disease identification. Our study pioneers the exploration of the zero-shot setting within this domain, offering a new perspective to conceptualizing plant disease identification. Specifically, we introduce the novel Cross Learning Vision Transformer (CL-ViT) model, incorporating self-supervised learning, in contrast to the previous state-of-the-art, FF-ViT, which emphasizes conceptual feature disentanglement with a synthetic feature generation framework. Through comprehensive analyses, we demonstrate that our novel model outperforms state-of-the-art models in both accuracy performance and visualization analysis. This study establishes a new benchmark and marks a significant advancement in the field of plant disease identification, paving the way for more robust and efficient plant disease identification systems. The code will be made available upon publication.

## Contribution
1. We introduce a novel model called CL-ViT, featuring unique conceptual designs, setting a new benchmark in the field of unseen plant disease identification.
2. We improved previous FF-ViT model from [Pairwise Feature Learning for Unseen Plant Disease Recognition](https://ieeexplore.ieee.org/abstract/document/10222401/).
3. We demonstrate that the incorporation of a guided learning mechanism surpasses conventional approaches in the multi-plant disease identification benchmark. Furthermore, we show that the CL-ViT model, integrating a SSL approach, outperforms the FF-ViT model employing a purely supervisory learning scheme for unseen plant disease identification tasks.
4. In our qualitative analyses, we illustrate that CL-ViT learns a feature space capable of discriminating between different classes while minimizing the domain gap between seen and unseen data. This underscores the superiority of CL-ViT in implementing a more effective guided learning mechanism.

## Proposed model
1.  Cross Learning Vision Transformer (CL-ViT) model [[code]](model/CL-ViT.py)
    * Key feature: Incorporate self-supervised learning to supervised model using pre-text tasks.
<p align="center">
  <img src="Figure/CL-ViT.png" alt="CL-ViT" width="800">
  <br>
  <i>Proposed CL-ViT architecture.</i>
</p>

2. Improved Feature Fusion Vision Transformer (FF-ViT) model [[code]](model/FF-ViT.py)
    * Key feature: Generate embeddings of synthetic composition based on training data.
<p align="center">
  <img src="Figure/FF-ViT.png" alt="FF-ViT" width="800">
  <br>
  <i>Proposed FF-ViT architecture.</i>
</p>

## Results
![Acc Results](Figure/results.png)

## Preparation

* Dataset: PV dataset [[spMohanty Github]](https://github.com/spMohanty/PlantVillage-Dataset/tree/master)  
(You can group all images into single folder to directly use the csv file provided in this repo if you downloaded the original dataset.)

* Pretrained weight: [[ViT pretrained weight]](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) (From [[rwightman Github timm repo]](https://github.com/huggingface/pytorch-image-models))

## Implementations
* CL-ViT model [[code]](model/CL-ViT.py)
* FF-ViT [[code]](model/FF-ViT.py)

Notes
* The csv file (metadata of images) for CL-ViT are [here](dataset/csv_CLViT/)
* The csv file (metadata of images) for FF-ViT are [here](dataset/csv_FFViT/)

## See also
1. [Pairwise Feature Learning for Unseen Plant Disease Recognition](https://ieeexplore.ieee.org/abstract/document/10222401/): The first implementation of FF-ViT model with moving weighted sum. The current work improved and evaluated the performance of FF-ViT model on larger-scale dataset.
2. [Unveiling Robust Feature Spaces: Image vs. Embedding-Oriented Approaches for Plant Disease Identification](https://ieeexplore.ieee.org/abstract/document/10317550/): The analysis between image or embedding feature space for plant disease identifications.

## Dependencies
Pandas == 1.4.1  
Numpy == 1.22.2  
torch == 1.10.2  
timm == 0.5.4  
tqdm == 4.62.3  
torchvision == 0.11.3  
albumentations == 1.1.0  

## License

Creative Commons Attribution-Noncommercial-NoDerivative Works 4.0 International License (“the [CC BY-NC-ND License](https://creativecommons.org/licenses/by-nc-nd/4.0/)”)

## Citation

```bibtex
@article{chai2024beyond,
  title={Beyond supervision: Harnessing self-supervised learning in unseen plant disease recognition},
  author={Chai, Abel Yu Hao and Lee, Sue Han and Tay, Fei Siang and Bonnet, Pierre and Joly, Alexis},
  journal={Neurocomputing},
  pages={128608},
  year={2024},
  publisher={Elsevier}
}
