# ProLesA-Net
Official repository for the ProLesA-Net model, published in here Cell Patterns  
https://www.cell.com/action/showPdf?pii=S2666-3899%2824%2900107-7

# Requirements
```
pip install tensorflow==2.7.0
```

# Model Architecture
ProLesA-Net along with the multiscale attention mechanisms are presented below:
![ProLesA-Net](ModelMaterials/ProlesaNet.png)
![MultiScale attention mechanisms](ModelMaterials/components.png)

# Usage
```python
import tensorflow as tf
import ProlesaModule

msqa = ProlesaModule.ProLesA_Net.ProlesaNet()
msqa.build(input_shape = [1,24, 192,192,3])

msqa.summary()

Model: "prolesa_net"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
encoder_block (EncoderBlock) multiple                  10692     
_________________________________________________________________
encoder_block_1 (EncoderBloc multiple                  67912     
_________________________________________________________________
encoder_block_2 (EncoderBloc multiple                  711312    
_________________________________________________________________
encoder_block_3 (EncoderBloc multiple                  2839840   
_________________________________________________________________
bottleneck2 (Bottleneck2)    multiple                  4979840   
_________________________________________________________________
decoder_block (DecoderBlock) multiple                  5908737   
_________________________________________________________________
decoder_block_1 (DecoderBloc multiple                  1823873   
_________________________________________________________________
decoder_block_2 (DecoderBloc multiple                  161217    
_________________________________________________________________
decoder_block_3 (DecoderBloc multiple                  40673     
_________________________________________________________________
classifier (Classifier)      multiple                  33        
=================================================================
Total params: 16,544,129
Trainable params: 16,537,217
Non-trainable params: 6,912
```
# Citation
Please Cite our work if you find it usefull ;)

```
@article{ZARIDIS2024100992,
title = {ProLesA-Net: A multi-channel 3D architecture for prostate MRI lesion segmentation with multi-scale channel and spatial attentions},
journal = {Patterns},
pages = {100992},
year = {2024},
issn = {2666-3899},
doi = {https://doi.org/10.1016/j.patter.2024.100992},
url = {https://www.sciencedirect.com/science/article/pii/S2666389924001077},
author = {Dimitrios I. Zaridis and Eugenia Mylona and Nikos Tsiknakis and Nikolaos S. Tachos and George K. Matsopoulos and Kostas Marias and Manolis Tsiknakis and Dimitrios I. Fotiadis},
keywords = {deep learning, magnetic resonance imaging, prostate lesion segmentation, multi-scale attention, cancer detection, medical imaging},
abstract = {Summary
Prostate cancer diagnosis and treatment relies on precise MRI lesion segmentation, a challenge notably for small (<15 mm) and intermediate (15–30 mm) lesions. Our study introduces ProLesA-Net, a multi-channel 3D deep-learning architecture with multi-scale squeeze and excitation and attention gate mechanisms. Tested against six models across two datasets, ProLesA-Net significantly outperformed in key metrics: Dice score increased by 2.2%, and Hausdorff distance and average surface distance improved by 0.5 mm, with recall and precision also undergoing enhancements. Specifically, for lesions under 15 mm, our model showed a notable increase in five key metrics. In summary, ProLesA-Net consistently ranked at the top, demonstrating enhanced performance and stability. This advancement addresses crucial challenges in prostate lesion segmentation, enhancing clinical decision making and expediting treatment processes.}
}
```
