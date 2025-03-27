# Disentangled Source-Free Personalisation for Facial Expression Recognition with Neutral Target Data
 
by Masoumeh Sharafi, Emma Ollivier, Muhammad Osama Zeeshan, Soufiane Belharbi, Marco Pedersoli, Alessandro Lameiras Koerich, Simon Bacon, EricGranger
# The Proposed Setting

# Abstract
Facial Expression Recognition (FER) from videos is a crucial task in various application areas, such as human-computer interaction and health monitoring (e.g., pain, depression, fatigue, and stress). Beyond the challenges of recognizing subtle emotional or health states, the effectiveness of deep FER models is often hindered by the considerable variability of expressions among subjects. Source-free domain adaptation (SFDA) methods are employed to adapt a pre-trained source model using only unlabeled target domain data, thereby avoiding data privacy and storage issues. Typically, SFDA methods adapt to a target domain dataset corresponding to an entire population and assume it includes data from all recognition classes. However, collecting such comprehensive target data can be difficult or even impossible for FER in healthcare applications. In many real-world scenarios, it may be feasible to collect a short neutral control video (displaying only neutral expressions) for target subjects before deployment. These videos can be used to adapt a model to better handle the variability of expressions among subjects. This paper introduces the Disentangled Source-Free Domain Adaptation (DSFDA) method to address the SFDA challenge posed by missing target expression data. DSFDA leverages data from a neutral target control video for end-to-end generation and adaptation of target data with missing non-neutral data. Our method learns to disentangle features related to expressions and identity while generating the missing non-neutral target data, thereby enhancing model accuracy. Additionally, our self-supervision strategy improves model adaptation by reconstructing target images that maintain the same identity and source expression.




# BioVid Database
```sh
Biovid datasets PartA can be downloaded from here: (https://www.nit.ovgu.de/BioVid.html#PubACII17)
```

# Train the Model on Source Domain
```sh
python main_src.py --epoch 100 --batchsize 20 --lr 1e-5
```

# Adaptation to Target Domains (Subjects)
```sh
python main_tar.py --epoch 25 --batchsize 32 --lr 1e-4 --biovid_annot_train $Path to the training data --biovid_annot_val $Path to the validation data --save_dir $Directory to save experiment results --img_dir Directory to save generated images --par_dir Directory to save the best parameters
```
# Test
```sh
python test.py
```

# Citation
```sh
@misc{sharafi2025disentangledsourcefreepersonalizationfacial,
      title={Disentangled Source-Free Personalization for Facial Expression Recognition with Neutral Target Data}, 
      author={Masoumeh Sharafi and Emma Ollivier and Muhammad Osama Zeeshan and Soufiane Belharbi and Marco Pedersoli and Alessandro Lameiras Koerich and Simon Bacon and Eric~Granger},
      year={2025},
      eprint={2503.20771},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.20771}, 
}
```
