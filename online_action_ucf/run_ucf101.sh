NEPOCH=50
LAMBDA1=10
COSEM_WEIGHT=0.1
RECONS_WEIGHT=0.01
#LR=0.0001
LR=0.00001
BATCH_SIZE=64 
SYN_NUM=600
RESSZ=8192
NDH=4096
NGH=4096
ATTSZ=300
NZ=300

# GZSL-OD
#python clswgan_action.py --gzsl_od --nclass_all 101 --dataroot ../data_feature --manualSeed 806 --ngh $NGH --ndh $NDH --lambda1 $LAMBDA1 --critic_iter 5 \
#--cosem_weight $COSEM_WEIGHT --recons_weight $RECONS_WEIGHT --syn_num 50 --preprocessing --cuda --batch_size $BATCH_SIZE --nz $NZ --attSize $ATTSZ --resSize $RESSZ --lr $LR \
#--action_embedding i3d --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch $NEPOCH --dataset ucf101 --split 1

# GZSL
#python clswgan_action.py --gzsl --nclass_all 101 --dataroot ../data_feature --manualSeed 806 --ngh $NGH --ndh $NDH --lambda1 $LAMBDA1 --critic_iter 5 \
#--cosem_weight $COSEM_WEIGHT --recons_weight $RECONS_WEIGHT --syn_num $SYN_NUM --preprocessing --cuda --batch_size $BATCH_SIZE --nz $NZ --attSize $ATTSZ --resSize $RESSZ --lr $LR \
#--action_embedding i3d --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch $NEPOCH --dataset ucf101 --split 1

# ZSL
#python clswgan_action.py --nclass_all 101 --dataroot ../data_feature --manualSeed 806 --ngh $NGH --ndh $NDH --lambda1 $LAMBDA1 --critic_iter 5 \
#--cosem_weight $COSEM_WEIGHT --recons_weight $RECONS_WEIGHT --syn_num $SYN_NUM --preprocessing --cuda --batch_size $BATCH_SIZE --nz $NZ --attSize $ATTSZ --resSize $RESSZ --lr $LR \
#--action_embedding i3d --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch $NEPOCH --dataset ucf101 --split 1


# My Action
CUDA_VISIBLE_DEVICES=1 python my_action_gan_od.py --nclass_all 101 --dataroot ../data_feature --manualSeed 806 --ngh $NGH --ndh $NDH --lambda1 $LAMBDA1 --critic_iter 5 \
--cosem_weight $COSEM_WEIGHT --recons_weight $RECONS_WEIGHT --syn_num $SYN_NUM --preprocessing --cuda --batch_size $BATCH_SIZE --nz $NZ --attSize $ATTSZ --resSize $RESSZ --lr $LR \
--action_embedding i3d --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch $NEPOCH --dataset ucf101 --split 1 \
--use_gan False --use_od False --use_train False
