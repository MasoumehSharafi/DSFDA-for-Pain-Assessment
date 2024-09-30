# Subject-Based Adaptation for Facial Expression Recognition Using a Neutral Control Video
This work is inspired by the disentanglement approach for facial expression recognition proposed by Xie et al. (2020) 

# The Proposed Setting
![proposed_setting](https://github.com/user-attachments/assets/183278e1-a398-4a85-8797-a5a3e2d717d8)
# Architecture

![Dis-SFDA_arc](https://github.com/user-attachments/assets/4992826a-bbe5-4f95-a0da-34a24f1d7d32)




# BioVid Database



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
```bibtex
@article{xie2020facial,
  title={Facial expression recognition with two-branch disentangled generative adversarial network},
  author={Xie, Siyue and Hu, Haifeng and Chen, Yizhen},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={31},
  number={6},
  pages={2359--2371},
  year={2020},
  publisher={IEEE}
}
