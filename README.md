# [Disentangled Source-Free Personalization for Facial Expression Recognition with Neutral Target Data (2025)](https://arxiv.org/pdf/2503.20771)


by
**Masoumeh Sharafi<sup>1</sup>,
Emma Ollivier<sup>1</sup>,
Muhammad Osama Zeeshan<sup>1</sup>,
Soufiane Belharbi<sup>1</sup>,
Marco Pedersoli<sup>1</sup>,
Alessandro Lameiras Koerich<sup>2</sup>,
Simon Bacon<sup>3,4</sup>,
Eric Granger<sup>1</sup>**

<sup>1</sup> LIVIA, LLS, Dept. of Systems Engineering, ÉTS, Montreal, Canada
<br/>
<sup>2</sup> LIVIA, Dept. of Software and IT Engineering, ÉTS, Montreal, Canada
<br/>
<sup>3</sup> Dept. of Health, Kinesiology \& Applied Physiology, Concordia University, Montreal, Canada
<br/>
<sup>4</sup> Montreal Behavioural Medicine Centre, Montreal, Canada

<p align="center"><img src="doc/promo.png" alt="outline" width="90%"></p>

[![arXiv](https://img.shields.io/badge/arXiv-2503.20771-b31b1b.svg)](https://arxiv.org/pdf/2503.20771)

## Abstract
Facial Expression Recognition (FER) from videos is a crucial task in various application areas, such as human-computer interaction and health monitoring (e.g., pain, depression, fatigue, and stress).
Beyond the challenges of recognizing subtle emotional or health states, the effectiveness of deep FER models is often hindered by the considerable variability of expressions among subjects. Source-free domain adaptation (SFDA) methods are employed to adapt a pre-trained source model using only unlabeled target domain data, thereby avoiding data privacy and storage issues. Typically, SFDA methods adapt to a target domain dataset corresponding to an entire population and assume it includes data from all recognition classes. However, collecting such comprehensive target data can be difficult or even impossible for FER in healthcare applications.
In many real-world scenarios, it may be feasible to collect a short neutral control video (displaying only neutral expressions) for target subjects before deployment. These videos can be used to adapt a model to better handle the variability of expressions among subjects. This paper introduces the Disentangled Source-Free Domain Adaptation (DSFDA) method to address the SFDA challenge posed by missing target expression data. DSFDA leverages data from a neutral target control video for end-to-end generation and adaptation of target data with missing non-neutral data. Our method learns to disentangle features related to expressions and identity while generating the missing non-neutral target data, thereby enhancing model accuracy. Additionally, our self-supervision strategy improves model adaptation by reconstructing target images that maintain the same identity and source expression.
Experimental results on the challenging BioVid and UNBC-McMaster pain datasets indicate that our DSFDA approach can outperform state-of-the-art adaptation methods.


## Citation:
```
@article{sharafi25,
  title={Disentangled Source-Free Personalization for Facial Expression Recognition with Neutral Target Data},
  author={Sharafi, M. and Ollivier, E. and Zeeshan, M.O. and Belharbi, S. and Pedersoli, M. and Koerich, A.L. and Bacon, S. and Granger, E.},
  journal={CoRR},
  volume={abs/2503.20771},
  year={2025}
}
```

## Installation of the environments
```bash
[TODO]
```


## BioVid database
```sh
Biovid datasets PartA can be downloaded from here: (https://www.nit.ovgu.de/BioVid.html#PubACII17)
```

## Train the model on source domain
```sh
python main_src.py --epoch 100 --batchsize 20 --lr 1e-5
```

## Adaptation to target domains (subjects)
```sh
python main_tar.py --epoch 25 --batchsize 32 --lr 1e-4 --biovid_annot_train $Path to the training data --biovid_annot_val $Path to the validation data --save_dir $Directory to save experiment results --img_dir Directory to save generated images --par_dir Directory to save the best parameters
```
## Test
```sh
python test.py
```
