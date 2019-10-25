##########################

This is the source code for training YOWO Action Detection on JHMDB.

1. Train
  1-1 Setting the dataset path in jhmdb21.data
      base: the path of JHMDB21 dataset
      train: the train list of JHMDB21 dataset (already given in "configs/lists/trainlist.txt")
      valid: the test list of JHMDB21 dataset (already given in "configs/lists/testlist.txt")
      names: the name of action in JHMDB21 (already given in "configs/data/jhmdb21.names")

  1-2 Setting for "sh" file
     --data_cfg: the config file of the dataset (already given in "configs/cfg/jhmdb21.data")
     --cfg_file: the config file for the train setting (already given in "configs/cfg/jhmdb21.cfg")
     --backbone_3d_weights: the pretrained model of 3d backbone (download in SVR11: "/home/sheqi/lei/ZIP/materials/yowo_action_detection(hmdb)/pretrained/resnext-101-kinetics-hmdb51_split1.pth"  into "pretrained")
     --backbone_2d_weights: the pretrained model of 2d backbone (download in "/home/sheqi/lei/ZIP/materials/yowo_action_detection(hmdb)/pretrained/yolo.weights" into "pretrained")
     --check_name: saving name of the train log

  1-3 Run 
     sh run_jhmdb-21.sh
    
2. Test
  2-1 Setting for "sh" file
     --resume_path: the trained model on jhmdb-21 (the trained model in logs/xxxx/backup/) 
     --test_video_list: the video list for test (already given in "configs/lists/test_video.txt")
     --is_vis: if True, output the results bounding box


  2-2 Run 
     sh run_video_mAP_jhmdb.sh

     Training VTN + SNL: validation_UCF101_snl.sh
