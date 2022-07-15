# SG-ShadowNet

#### PyTorch implementation of “Style-Guided Shadow Removal”, [ ECCV2022 ]( https://eccv2022.ecva.net/ ).

### Intruduction:
#### Abstract
Shadow removal is an important topic in image restoration,  and it can benefit many computer vision tasks. State-of-the-art shadowremoval  methods typically employ deep learning by minimizing a pixellevel  difference between the de-shadowed region and their corresponding (pseudo) shadow-free version. After shadow removal, the shadow and  non-shadow regions may exhibit inconsistent appearance, leading to a  visually disharmonious image. To address this problem, we propose a  style-guided shadow removal network (SG-ShadowNet) for better imagestyle  consistency after shadow removal. In SG-ShadowNet, we first learn  the style representation of the non-shadow region via a simple region  style estimator. Then we propose a novel effective normalization strategy  with the region-level style to adjust the coarsely re-covered shadow region  to be more harmonized with the rest of the image. Extensive experiments  show that our proposed SG-ShadowNet outperforms all the existing  competitive models and achieves a new state-of-the-art performance  on ISTD+, SRD, and Video Shadow Removal benchmark datasets.
### 1.  Requirement
Simpy run (Note that the different "scikit-image" versions will affect test results.):
```shell
conda install --yes --file requirements.txt
```

### 2. Datasets
[ISTD+](https://www3.cs.stonybrook.edu/~cvl/projects/SID/index.html)

SRD  (please email the [authors](http://vision.sia.cn/our%20team/JiandongTian/JiandongTian.html)).

[Video Shadow Removal (version 1)](https://www3.cs.stonybrook.edu/~cvl/projects/FSS2SR/index.html)


### 3. Evaluate the Pre-trained Models
We provide our shadow-removal results from [GoogleDrive](https://drive.google.com/drive/folders/1BtvVDRUe7HARGyJAwXf8CSVMbFKjBxDI?usp=sharing) generated on the ISTD+, SRD, and Video Shadow Removal benchmark dataset.

You can also evaluate the pretrained models. Place images and testing mask ([ISTD](https://github.com/hhqweasd/G2R-ShadowNet) and [SRD](https://github.com/vinthony/ghost-free-shadow-removal))  in `./input` directory and run the below script.

Before executing the script, please download the pretrained model from [GoogleDrive](https://drive.google.com/drive/folders/14cPEJMYSUFTLB4yaZ2jWJpx8kFre80oH?usp=sharing), place the models to `./pretrained`. 

```bash
python test.py
```

Note that, ISTD+  and Video Shadow Removal have to be evaluate using the model trained on the ISTD+ dataset while SRD via the model with SRD dataset.


### 4. Train Models

Run the following scripts to train models.


```bash
python train.py
```

### Acknowledgements

Thanks [Zhihao Liu](https://github.com/hhqweasd) for his useful discussion of this work.

Thanks to previous open-sourced repo: 
[G2R-ShadowNet](https://github.com/hhqweasd/G2R-ShadowNet)

