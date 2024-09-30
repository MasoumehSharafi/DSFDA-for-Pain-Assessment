
import numpy as np
import os
import train_tar
#import train_src
import argparse
import torch
import util
import random


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"



def run(args):
    # construct the hyper-parameter dictionary
    hpar_dict = {
        'biovid_annot_train': args.biovid_annot_train,
        'biovid_annot_val': args.biovid_annot_val,
        'save_dir': args.save_dir,
        'img_dir': args.img_dir,
        'par_dir': args.par_dir,
        
        'RESOLUTION': 128,
        'NUM_AUGMENTATIONS' : 22,
        'FOLD': 8,
        # noise dimension
        'Nz': 256,
        # steps for discriminator update (F: Face, E: Expression)
        'D_GAP_FR': 1, 
        'D_GAP_ER': 1, 
        # steps for saving images
        'IMG_SAVE_GAP': 100,
        # gaps of epochs to save the parameters
        'PAR_SAVE_GAP': 50,
        # validation gap
        'VAL_GAP': 1,
        # batch size
        'BS': args.batchsize,
        # training epochs
        'epoch': args.epoch,
        # class number for face recognition (the first K_f persons)
        'FR_cls_num': 1, # 42 for km70  # 78 for km1(for each subject) # 78 for well_val # 62 for db25
        'AR_cls_num': 1,
        # learning rate
        'LR_D_FR': args.lr,  # face discriminator
        'LR_D_ER': args.lr,  # expression discriminator
        'LR_G_FR': args.lr,  # face encoder
        'LR_G_ER': args.lr,  # expression encoder
        # coefficients to balance the loss of generator or discriminator
        'H_G_FR_f': 0.2,  # lambda_G_f
        'H_G_ER_f': 0.8,  # lambda_G_e
        'H_G_FR_PER': 1,  # lambda_per_f
        'H_G_ER_PER': 0,  # lambda_per_e
        'H_G_CON_FR': 5,  # lambda_DIC
        'H_G_CON_ER': 5,  # lambda_DIC
        'H_D_FR_r': 1,
        'H_D_FR_f': 1,
        'H_D_ER_r': 1,
        'H_D_ER_f': 0,
        # flags to indicate whether to generate grayscale images
        'FLAG_GEN_GRAYIMG': False,
        # working mode
        'train': args.train,
        'device': "cuda",
    }
  

    save_dir = hpar_dict['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hpar_dict['save_dir'] = save_dir

    print('---- START RUNNING ----')
    print('WORKING MODE: {}\n'.format('TRAIN' if hpar_dict['train'] else 'VALIDATION'))

    train_tar.train(hpar_dict)


  
    print('--- END OF RUNNING ----')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dis-SFDA model')
    parser.add_argument('--train', default=True, action='store_true', help='flag to indicate working mode: default False (validation mode)')
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs for adaptation, default: 25')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size, defualt: 32')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default: 1e-4')
    parser.add_argument('--gpu', type=int, default=1, help='set gpu device, default: -1 (cpu)')
    
    parser.add_argument('--biovid_annot_train', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--biovid_annot_val', type=str, required=True, help='Path to the validation CSV file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save experiment results')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--par_dir', type=str, required=True, help='Directory to save the best parameters')


    args = parser.parse_args()
    run(args)
