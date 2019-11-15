##########################

This is the source code for training online action recogniton on UCF101

1. Setting:
  --dataroot: the root dir of the extracted feature (given in SVR11:"/home/sheqi/lei/ZIP/materials/online_action(ucf)/data_feature/")
  --ngh: the dim of the latent channel (defaulated 4096)
  --use_gan: Setting "True" to use GAN generating some unseen feature (should copy "/home/sheqi/lei/ZIP/materials/online_action(ucf)/netG.tar" into this folder)
  --use_od: Setting "True" to use OD selector to select the seen/unseen feature (should copy "/home/sheqi/lei/ZIP/materials/online_action(ucf)/netD.tar" into this folder)
  --use_train: Setting "True" to use train feature rather than the semantic feature to initalize the register
  
2. Run:
   sh run_ucf101.sh

   #GZSL-OD: the structure of 18CVPR-OD
   #MY Atction: Implimentation for the structure we discussed