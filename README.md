## Introduction
* One implementation of the paper __Learning from Bootstrapping and Stepwise Reinforcement Reward: A Semi-Supervised Framework for Text Style Transfer__ in NAACL 2022.
* This code and data are only for research use. Please cite the papers if they are helpful. <br>


## Package Requirements
The model training and inference scripts are tested on following libraries and versions:
1. pytorch==1.8.1
2. transformers==4.8.2

## Guideline
1. Run the script `classification_style_BERT.py` to obtain the style classification model, which will be used as the style discriminator when training the semi-supervised style transfer framework. Here data labeled with sentiment polairty are used (e.g., Yelp and Amazon corpora for sentiment style transfer).
2. Run the script `find_sentence_Multi_Thread_Yelp.py` to construct the pseudo parallel pairs, see the paper for details. We have provided the pseudo parallel pairs in this repo (`pseudo_paired_data_Yelp` and `pseudo_paired_data_Amazon`).
3. Run the script `main.py` to train and evaluate the semi-supervised style transfer framework. Before your run it, please check the training configurations in `global_config.py`.


## Model Generation
See the system outputs of Yelp, Amazon, and GYAFC test sets in `./model_outputs/`.

## Citation
If the work is helpful, please cite our papers in your publications, reports, slides, and thesis.
```
@inproceedings{liu-chen-2022-learning,
    title = "Learning from Bootstrapping and Stepwise Reinforcement Reward: A Semi-Supervised Framework for Text Style Transfer",
    author = "Liu, Zhengyuan  and
      Chen, Nancy",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.201",
    doi = "10.18653/v1/2022.findings-naacl.201",
    pages = "2633--2648",
}
```

