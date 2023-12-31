# ISsegATUNet
Implementation of the metwork for segmentation of the interventricular septum in MR black blood myocardium images.

# Description
The ISsegATUNet adopted a symmetrical "U" type encoding-decoding structure. The backbone of encoder was a five-layer DenseNet, while the decoder was composed of channel attention (CA), non-local Attention (NLA), dual-pathway spatial attention (DSA) and scale attention (SA).

# Dependencies
The code has been only tested in the environment as following
- Ubuntu 18.04.6 LTS
- Python 3.8.12
- Pytorch 1.1.0


# Reference
Zifeng Lian, Qiqi Lu, Bingquan Lin, Chen Lingjian, Peng Peng, Yanqiu Feng.
**"MRI Deep learning-based automatic segmentation of interventricular septum for black-blood myocardial T2star measurement in thalassemia."**
Journal of Magnetic Resonance Imaging. 2023 Nov 9. doi: 10.1002/jmri.29113. Epub ahead of print. PMID: 37941460.

# License
See [LICENSE](LICENSE)
