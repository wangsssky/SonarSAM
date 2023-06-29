# SonarSAM
This study presents the introduction of the Segment-Anything-Model (SAM) to sonar images. We conduct a comprehensive investigation into fine-tuning methods for SAM, including LoRA and visual prompt tuning. To facilitate comparison, we provide a framework that integrates these fine-tuning methods for SAM. If this project is helpful to your research, please consider citing our paper [PDF](https://arxiv.org/pdf/2306.14109.pdf).
```
@article{wang2023sonarsam,
  title={When SAM Meets Sonar Images},
  author={Wang, Lin and Ye, Xiufen and Zhu, Liqiang and Wu, Weijie and Zhang, Jianguo and Xing, Huiming and Hu, Chao},
  journal={arXiv preprint arXiv:2306.14109},
  year={2023}
}
```
# Update
- Add [Mobile SAM](https://github.com/ChaoningZhang/MobileSAM) backbone support for Image encoder fully-tuning.
# Dataset
The Marine Debris dataset is used in this work, which is available at [Forward-Looking Sonar Marine Debris Datasets](https://github.com/mvaldenegro/marine-debris-fls-datasets).

# Training
- Using box prompts
```
python train_SAM_box.py --config ./configs/sam_box.yaml
```

- Semantic segmentation
```
python train_SAM.py --config ./configs/sam.yaml
```
# Acknowledgment
This project was developed based on the following awesome codes.
- Segment Anything Model: [SAM](https://github.com/facebookresearch/segment-anything)
- Prompt layer & Custom segmentation head: [LearnablePromptSAM](https://github.com/Qsingle/LearnablePromptSAM/)
- LoRA: [SAMed](https://github.com/hitachinsk/SAMed/)
