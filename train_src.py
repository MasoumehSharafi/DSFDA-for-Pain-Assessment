'''
the trainer function
'''

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import time
import torch
import torch
torch.cuda.set_device(0)  # Assuming you want to use the first GPU
from torch import nn, optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import models as model
import numpy as np
import pandas as pd
import data_src
import util

import PIL.Image as Image
from tqdm import tqdm
import random
import copy
import learn2learn

from sklearn.model_selection import KFold

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_deterministic(seed=0):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

 
def train(hpar_dict):

    seed=0

    make_deterministic(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    

    # region: hyper-parameters for the model
    Resolution=hpar_dict["RESOLUTION"]
    Fold = 5
    # noise channel
    Nz = hpar_dict['Nz']

    # steps gap to update discriminator
    D_GAP_FR = hpar_dict['D_GAP_FR']
    D_GAP_ER = hpar_dict['D_GAP_ER']
    # steps gap for saving images
    IMG_SAVE_GAP = hpar_dict['IMG_SAVE_GAP']
    # gaps to save parameters (epochs)
    PAR_SAVE_GAP = hpar_dict['PAR_SAVE_GAP']
    # validation gap
    VAL_GAP = hpar_dict['VAL_GAP']
    # batch size
    BS = hpar_dict['BS']
    # training epochs
    epoch = hpar_dict['epoch']
    # face recognition class number
    FR_cls_num = hpar_dict['FR_cls_num']

    # learning rate
    LR_D_FR = hpar_dict['LR_D_FR']
    LR_D_ER = hpar_dict['LR_D_ER']
    LR_G_FR = hpar_dict['LR_G_FR']
    LR_G_ER = hpar_dict['LR_G_ER']
    # weights to balance the loss of generator or discriminator
    H_G_FR_f = hpar_dict['H_G_FR_f']
    H_G_ER_f = hpar_dict['H_G_ER_f']
    H_G_FR_PER = hpar_dict['H_G_FR_PER']
    H_G_ER_PER = hpar_dict['H_G_ER_PER']
    H_G_CON_FR = hpar_dict['H_G_CON_FR']
    H_G_CON_ER = hpar_dict['H_G_CON_ER']

    H_D_FR_r = hpar_dict['H_D_FR_r']
    H_D_FR_f = hpar_dict['H_D_FR_f']
    H_D_ER_r = hpar_dict['H_D_ER_r']
    H_D_ER_f = hpar_dict['H_D_ER_f']
    # flag to indicate whether to generate grayscale images
    FLAG_GEN_GRAYIMG = hpar_dict['FLAG_GEN_GRAYIMG']

    save_dir = hpar_dict['save_dir']

                    
    dic_log = {'loss_G_FR':[], 'loss_G_ER':[], 'loss_CON_FR':[], 'loss_CON_ER':[],'loss_PER_FR':[], 'loss_PER_ER':[], 'loss_TOT':[], 'accuracy':[]}
    dic_log_val = {'acc_val':[],'acc_val_dis':[], 'loss_val':[]}


    Biovid_img_all = '/state/share1/datasets/Biovid/sub_red_classes_img/'


    tr = util.data_augm(Resolution)
    tr_test = util.data_adapt(Resolution)
    tr_size = transforms.Resize((Resolution,Resolution),antialias=True)

    ER_cls_num = 2
    # patience = 10
    # min_delta = 0.001
    # epochs_no_improve = 0
    # best_val_acc = 0
    # early_stop = False


    tr = util.data_augm(Resolution)
    tr_test = util.data_adapt(Resolution)
    tr_size = transforms.Resize((Resolution,Resolution),antialias=True)

  
    biovid_annot_train="/home/ens/AT46120/GAN_Models/disentanglement_tdgan/dataset/training/train.csv"
    biovid_annot_val = "/home/ens/AT46120/GAN_Models/disentanglement_tdgan/dataset/training/valid.csv"
    dataset_train = data_src.DualTrainDataset(Biovid_img_all,biovid_annot_train,biovid_annot_train,transform = tr.transform,IDs = None,nb_image = None,preload=False)
    dataset_val = data_src.Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_val,transform = tr_test.transform,IDs = None,nb_image = None ,preload=False)

    print("data loaded")

    save_dir = "/home/ens/AT46120/GAN_Models/disentanglement_tdgan/src_train/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_dir = "/home/ens/AT46120/GAN_Models/disentanglement_tdgan/src_train/img/"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    par_dir = os.path.join(save_dir, 'best_parameters')
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)

    # directory to save parameters
    save_log_name='log_lr'+str(LR_G_ER)+'_batchsize'+str(BS)+'_gan_train.csv'
    save_log_name_val='log_lr'+str(LR_G_ER)+'_batchsize'+str(BS)+'_gan_test.csv'


    # dataloader for training, which contains two datasets
    #train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BS, sampler=torch.utils.data.SubsetRandomSampler(train_idx), num_workers=4)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=20, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=20, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
   
    # dataloader for testing, which only contains the expression dataset
    #expr_tt_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BS, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    # instantiate Generator
    Gen = model.Gen(clsn_ER=ER_cls_num, Nz=Nz, GRAY=FLAG_GEN_GRAYIMG, Nb=6)

    # instantiate face discriminator
    Dis_FR = model.Dis(GRAY=FLAG_GEN_GRAYIMG, cls_num=FR_cls_num + 1)
    # instantiate expression discriminator
    Dis_ER = model.Dis(GRAY=FLAG_GEN_GRAYIMG, cls_num=ER_cls_num)

    # instantiate Expression Clssification Module (M_ER)
    Dis_ER_val = model.Dis()
    Dis_ER_val.enc = Gen.enc_ER
    Dis_ER_val.fc = Gen.fc_ER

    Dis_merge = model.Dis()

    Gen.to(hpar_dict['device'])
    Dis_FR.to(hpar_dict['device'])
    Dis_ER.to(hpar_dict['device'])
    Dis_ER_val.to(hpar_dict['device'])
    Dis_merge.to(hpar_dict['device'])


    # parameters of the generator
    par_list_G_joint = [{'params': Gen.dec.parameters(), 'lr': LR_G_ER},
                        {'params': Gen.enc_FR.parameters(), 'lr': LR_G_FR},
                        {'params': Gen.enc_ER.parameters(), 'lr': LR_G_ER}
                        ]
    # parameters of the Expression Recognition Module
    par_list_G_ER_fc = [{'params': Gen.fc_ER.parameters(), 'lr': LR_D_ER},
                        ]
    # parameters of the two discriminators
    par_list_D_FR = [{'params': Dis_FR.parameters(), 'lr': LR_D_FR},
                    ]
    par_list_D_ER = [{'params': Dis_ER.parameters(), 'lr': LR_D_ER},
                    ]
        
    # parameters of the generator

    optG_joint = optim.Adam(par_list_G_joint)
    optG_ER_fc = optim.Adam(par_list_G_ER_fc)
    optD_FR = optim.Adam(par_list_D_FR)
    optD_ER = optim.Adam(par_list_D_ER)


    # criterion for loss
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    L1_loss = nn.L1Loss()


        
    # buffer to store validation accuracy (Expression Classification Module)
    tt_acc_mat = []
    tt_ce_mat = []
    # buffer to store validation accuracy (Expression discriminator)
    tt_acc_mat_ExpDis = []
    tt_ce_mat_ExpDis = []
    acc_max=0

    acc_max_ExpDis=0
 
    for e in range(1, epoch + 1):
        # if early_stop:
        #     print("Early stopping")
        #     break

        print('---- training ----')
        # the number of steps that an epoch goes
        step_total = train_loader.__len__()
        t_start = time.time()
        print('the %d-th training epoch' % (e))

        # set training mode
        Gen.train()
        Dis_ER.train()
        Dis_FR.train()

        pre_list = []
        pre_list_dis = []
        GT_list = []

        loop_train = tqdm(train_loader ,colour='BLUE')
        loss_tot = 0
        elem_sum = 0
        for step, (batch_FR_x_r, batch_FR_y_r, batch_FR_y_pain_r, batch_ER_x_r, batch_ER_y_r, batch_ER_y_ID_r) in enumerate(loop_train):      
                
            batch_FR_y_f = FR_cls_num * torch.ones(len(batch_FR_y_r)).long()

            # convert all tensors to the form of torch.Variables
            batch_FR_x_r = Variable(batch_FR_x_r).to(hpar_dict['device'])
            batch_FR_y_r = Variable(batch_FR_y_r).long().to(hpar_dict['device'])

            batch_ER_x_r = Variable(batch_ER_x_r).to(hpar_dict['device'])
            batch_ER_y_r = Variable(batch_ER_y_r).long().to(hpar_dict['device'])

            batch_FR_y_f = Variable(batch_FR_y_f).long().to(hpar_dict['device'])

            elem_sum += batch_ER_x_r .shape[0]

            # go through the discriminators
            batch_FR_Dfea_r, batch_FR_Dp_r = Dis_FR(batch_FR_x_r)
            batch_FR_Dfea_r = Variable(batch_FR_Dfea_r.data, requires_grad=False)
            batch_ER_Dfea_r, batch_ER_Dp_r = Dis_ER(batch_ER_x_r)
            batch_ER_Dfea_r = Variable(batch_ER_Dfea_r.data, requires_grad=False)

            # loss of face discriminator (with respect to real samples)
            loss_D_FR_r = CE(batch_FR_Dp_r, batch_FR_y_r)
            # loss of expression discriminator (with respect to real samples)
            loss_D_ER_r = CE(batch_ER_Dp_r, batch_ER_y_r)

            # generat images
                
            batch_x_f = Gen.gen_img(batch_FR_x_r, batch_ER_x_r, device=hpar_dict['device'])
            batch_ER_Gfea_r = Variable(Gen.fea_ER.data, requires_grad=False)

            optG_ER_fc.zero_grad()
            err_G_ER_r = CE(Gen.result_ER, batch_ER_y_r)
            err_G_ER_r.backward(retain_graph=True)
            #print(err_G_ER_r)
            optG_ER_fc.step()
                
            optD_FR.zero_grad()
            optD_ER.zero_grad()
                
            if step % D_GAP_FR == 0:
                batch_FR_Dfea_f, batch_FR_Dp_f = Dis_FR(batch_x_f.detach())
                loss_D_FR_f = CE(batch_FR_Dp_f, batch_FR_y_f)

                loss_D_FR = H_D_FR_r * loss_D_FR_r + H_D_FR_f * loss_D_FR_f

                loss_D_FR.backward()
                optD_FR.step()
                
            if step % D_GAP_ER == 0:

                loss_D_ER = H_D_ER_r * loss_D_ER_r

                loss_D_ER.backward()
                optD_ER.step()

                
            optG_joint.zero_grad()

            # get the predicted results on fake samples
            batch_FR_Dfea_f, batch_FR_Dp_f = Dis_FR(batch_x_f)
            batch_ER_Dfea_f, batch_ER_Dp_f = Dis_ER(batch_x_f)

            err_G_FR_f = CE(batch_FR_Dp_f, batch_FR_y_r) # Equ.6 first part
            err_G_ER_f = CE(batch_ER_Dp_f, batch_ER_y_r) # Equ.6 second part
            err_G_FR_PER = MSE(batch_FR_Dfea_f, batch_FR_Dfea_r) # Equ.8

            # consistency loss (Fig.3 upper part): face branch input: the generated image, expression branch input: the original face image, expected output: same as the original face image
            batch_x_f_FR = Gen.gen_img(batch_x_f, batch_FR_x_r, device=hpar_dict['device'])
            # consistency loss (Fig.3 lower part): face branch input: the original expression image, expression branch: the generated image, expected output: same as the original expression image
            batch_x_f_ER = Gen.gen_img(batch_ER_x_r, batch_x_f, device=hpar_dict['device'])

            # expression perceptual error (unused)
            batch_ER_Gfea_f = Variable(Gen.fea_ER.data).to(hpar_dict['device'])
            err_G_ER_PER = MSE(batch_ER_Gfea_f, batch_ER_Gfea_r)
            #print(err_G_ER_PER)

            err_G_con_FR = L1_loss(batch_x_f_FR, batch_FR_x_r)
            err_G_con_ER = L1_loss(batch_x_f_ER, batch_ER_x_r)
            err_G_con = H_G_CON_FR * err_G_con_FR + H_G_CON_ER * err_G_con_ER # Equ.7
            loss_G = H_G_FR_f * err_G_FR_f + H_G_ER_f * err_G_ER_f + \
                    H_G_FR_PER * err_G_FR_PER + H_G_ER_PER * err_G_ER_PER + err_G_con
            loss_G.backward()
            optG_joint.step()
            GT_list = np.hstack((GT_list, batch_ER_y_r.cpu().numpy()))
                
            batch_result = Gen.result_ER.cpu().data.numpy().argmax(axis=1)
            pre_list = np.hstack((pre_list, batch_result))

            val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
                
            batch_result_dis = batch_ER_Dp_r.cpu().data.numpy().argmax(axis=1)
            pre_list_dis = np.hstack((pre_list_dis, batch_result_dis))

            val_acc_dis = (np.sum((GT_list == pre_list_dis).astype(float))) / len(GT_list)
                

            # save the generated images
                
            if step % IMG_SAVE_GAP == 0:
                # combine five images of real face, real expression and fake images
                comb_img = util.combinefig_dualcon(batch_FR_x_r.cpu().data.numpy(),
                                                batch_ER_x_r.cpu().data.numpy(),
                                                batch_x_f.cpu().data.numpy(),
                                                batch_x_f_FR.cpu().data.numpy(),
                                                batch_x_f_ER.cpu().data.numpy())
                # save figures
                comb_img = Image.fromarray((comb_img * 255).astype(np.uint8))
                comb_img.save(os.path.join(img_dir, str(e) + '_' + str(step) + '.jpg'))
                f = open(os.path.join(img_dir, str(e) + '_' + str(step) + '.txt'), "a")
                f.write("pain : " +str(batch_FR_y_pain_r))
                f.write("\n")
                f.write("pain 2 : " +str(batch_ER_y_r))
                f.write("\n")
                f.write("\n")
                f.close()
                
                
            loop_train.set_description(f"Epoch [{e}/{epoch}] training")

            loop_train.set_postfix(accuracy_pain=val_acc*100, accuracy_pain_dis=val_acc_dis*100, loss_dis=loss_tot/elem_sum)
                
        
        dic_log['loss_G_FR'].append(err_G_FR_f.cpu().data)
        dic_log['loss_G_ER'].append(err_G_ER_f.cpu().data)
        dic_log['loss_CON_FR'].append(err_G_con_FR.cpu().data)
        dic_log['loss_CON_ER'].append(err_G_con_ER.cpu().data)
        dic_log['loss_PER_FR'].append(err_G_FR_PER.cpu().data)
        dic_log['loss_PER_ER'].append(err_G_ER_PER.cpu().data)
        dic_log['loss_TOT'].append(loss_G.cpu().data)
        dic_log['accuracy'].append(val_acc*100)
            
            
        dataframe = pd.DataFrame(dic_log)
        dataframe.to_csv(save_dir+save_log_name)
            
        t_end = time.time()
        print('an epoch last for %f seconds\n' % (t_end - t_start))

        # region validation
           
        if e % VAL_GAP == 0:

            with torch.no_grad():

                Dis_ER_val.eval()
                tt_acc, tt_ce = util.Val_acc(val_loader, Dis_ER_val, CE, hpar_dict['device'], e, epoch)
                tt_acc_mat.append(tt_acc)
                tt_ce_mat.append(tt_ce)
                if tt_acc > acc_max:

                        
                    acc_max = tt_acc
                    torch.save(Gen.enc_ER.state_dict(), os.path.join(par_dir, 'Enc_ER_G.pkl'))
                    torch.save(Gen.enc_FR.state_dict(), os.path.join(par_dir, 'Enc_FR_G.pkl'))
                    torch.save(Gen.fc_ER.state_dict(), os.path.join(par_dir, 'fc_ER_G.pkl'))
                    torch.save(Gen.dec.state_dict(), os.path.join(par_dir, 'dec.pkl'))
                    torch.save(Dis_FR.state_dict(), os.path.join(par_dir, 'Dis_FR.pkl'))
                    torch.save(Dis_ER.state_dict(), os.path.join(par_dir, 'Dis_ER.pkl'))
                        
                print('\n')
                print('the %d-th epoch' % (e))
                print('accuracy is : %f' % (tt_acc))
                print('validation cross enntropy is : %f' % (tt_ce))
                print('now the best accuracy is %f\n' % (np.max(tt_acc_mat)))


                Dis_ER.eval()
                tt_acc_ExpDis, tt_ce_ExpDis = util.Val_acc(val_loader, Dis_ER, CE, hpar_dict['device'], e, epoch)
                tt_acc_mat_ExpDis.append(tt_acc_ExpDis)
                tt_ce_mat_ExpDis.append(tt_ce_ExpDis)
                if tt_acc_ExpDis > acc_max_ExpDis :

                    acc_max_ExpDis = tt_acc_ExpDis
                    torch.save(Gen.enc_ER.state_dict(), os.path.join(par_dir, 'dis_val_Enc_ER_G.pkl'))
                    torch.save(Gen.enc_FR.state_dict(), os.path.join(par_dir, 'dis_val_Enc_FR_G.pkl'))
                    torch.save(Gen.fc_ER.state_dict(), os.path.join(par_dir, 'dis_val_fc_ER_G.pkl'))
                    torch.save(Gen.dec.state_dict(), os.path.join(par_dir, 'dis_val_dec.pkl'))
                    torch.save(Dis_FR.state_dict(), os.path.join(par_dir, 'dis_val_Dis_FR.pkl'))
                    torch.save(Dis_ER.state_dict(), os.path.join(par_dir, 'dis_val_Dis_ER.pkl'))

                # if e==5 :
                #     torch.save(Gen.enc_ER.state_dict(), os.path.join(par_dir, 'epoch5_Enc_ER_G.pkl'))
                #     torch.save(Gen.enc_FR.state_dict(), os.path.join(par_dir, 'epoch5_Enc_FR_G.pkl'))
                #     torch.save(Gen.fc_ER.state_dict(), os.path.join(par_dir, 'epoch5_fc_ER_G.pkl'))
                #     torch.save(Gen.dec.state_dict(), os.path.join(par_dir, 'epoch5_dec.pkl'))
                #     torch.save(Dis_FR.state_dict(), os.path.join(par_dir, 'epoch5_Dis_FR.pkl'))
                #     torch.save(Dis_ER.state_dict(), os.path.join(par_dir, 'epoch5_Dis_ER.pkl'))
                print('testing using discriminator:')
                print('accuracy is : %f' % (tt_acc_ExpDis))
                print('testing cross enntropy is : %f' % (tt_ce_ExpDis))
                print('now the best accuracy is %f\n' % (np.max(tt_acc_mat_ExpDis)))
                    
            dic_log_val['acc_val_dis'].append(tt_acc_ExpDis)
            dic_log_val['acc_val'].append(tt_acc)
            dic_log_val['loss_val'].append(tt_ce)

            dataframe_val = pd.DataFrame(dic_log_val)
            dataframe_val.to_csv(save_dir+save_log_name_val)
            '''
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                epochs_no_improve = 0
                # Save the best model
                torch.save(Gen.state_dict(), os.path.join(par_dir, 'Gen_best.pth'))
                torch.save(Dis_FR.state_dict(), os.path.join(par_dir, 'Dis_FR_best.pth'))
                torch.save(Dis_ER.state_dict(), os.path.join(par_dir, 'Dis_ER_best.pth'))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {e}")
                early_stop = True
            break
            '''
        
    print('end')
    
