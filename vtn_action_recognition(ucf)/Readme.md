##########################

This is the source code for training our VTN/VTN_SNL on UCF101.

1. Train
  1-1 Setting for "sh" file
     --pretrain-path: the pretrained path of the model on Kinetics (copy from SVR-11: "/home/sheqi/lei/ZIP/materials/vtn_action_recognition(ucf)/pretrain/resnet_34_vtn_rgb_kinetics.pth" into "pretrained") 
     --video-path: the path of UCF101 rgb image
     --annotation-path: the annotation file for the splits (already generating in the dir "split")

  1-2 Run 
     Training org VTN: sh train_UCF101_org.sh

     Training VTN + SNL: train_UCF101_snl.sh
    
2. Test
  2-1 Setting for "sh" file
     --pretrain-path: the trained model on UCF101 (the model in log/xxxx/checkpoints) 
     --video-path: the path of UCF101 rgb image
     --annotation-path: the annotation file for the splits ("Using same spilt file for train and test")


  2-2 Run 
     Training org VTN: sh validation_UCF101.sh

     Training VTN + SNL: validation_UCF101_snl.sh

