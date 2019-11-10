python eval_video.py --dataset jhmdb-21 \
		--data_cfg configs/cfg/jhmdb21.data \
		--cfg_file configs/cfg/jhmdb21.cfg \
		--n_classes 21 \
		--backbone_3d resnext101 \
		--backbone_2d darknet \
                --resume_path logs/train_logs/jhmdb-21/darknet_resnext101_two_branch_2/backup/yowo_jhmdb-21_16f_checkpoint.pth \
                --check_name visual_two_branch \
                --test_video_list configs/lists/debug_video.txt \
                --is_vis True
