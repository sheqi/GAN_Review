##########################

This is the source code for training our SNL on Imagenet.

########Requirements######
PyTorch >= 0.4.1 or 1.0
Python >= 3.5
torchvision >= 0.2.1
termcolor >= 1.1.0
tensorboardX >= 1.9
opencv >= 3.4

########Train SNL#########
To run train code:
1. install the conda environment using "env.yml"
2. Copying the pretrained ResNet50 model from SVR11:"/home/sheqi/lei/ZIP/materials/snl_image_classification(imagenet)/pretrained/resnet50-19c8e357.pth" and put it into the folder "pretrained". 
3. setting --data_dir as the root directory of the dataset in "train_snl.shs"
4. run the code by: "sh train_snl.sh"
5. the training log and checkpoint are saving in "save_model"
(We give the trained model on imagenet on google drive: "https://drive.google.com/drive/folders/1Iy6Wxe0qQSOc9ayHszEOyxya34xzuFM8?usp=sharing") 

#####Evaluate SNL#########
To run evaluate code:
1. install the conda environment using "env.yml"
2. Copy the trained SNL model from SVR11:"/home/sheqi/lei/ZIP/materials/snl_image_classification(imagenet)/imagenet_snl.tar"
3. Setting --check_path as the path of the downloaded SNL model in "eval_snl.sh".
4. Setting --data_dir as the root directory of the dataset in "eval_snl.sh".
5. run the code by: "sh eval_snl.sh"
 