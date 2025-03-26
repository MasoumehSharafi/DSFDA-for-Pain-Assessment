# Disentangled Source-Free Personalisation for Facial Expression Recognition with Neutral Target Data
This work is inspired by the disentanglement approach for facial expression recognition proposed by Xie et al. (2020) 

# The Proposed Setting

# Architecture






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

XIE, Siyue, HU, Haifeng, et CHEN, Yizhen. Facial expression recognition with two-branch disentangled generative adversarial network. IEEE Transactions on Circuits and Systems for Video Technology, 2020, vol. 31, no 6, p. 2359-2371.
