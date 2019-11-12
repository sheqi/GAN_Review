python train.py --dataset jhmdb-21 \
		--data_cfg configs/cfg/jhmdb21.data \
		--cfg_file configs/cfg/jhmdb21.cfg \
		--n_classes 21 \
		--backbone_3d resnext101 \
		--backbone_3d_weights pretrained/resnext-101-kinetics-hmdb51_split1.pth \
		--freeze_backbone_3d \
		--backbone_2d darknet \
		--backbone_2d_weights pretrained/yolo.weights \
		--freeze_backbone_2d \
                --check_name org


