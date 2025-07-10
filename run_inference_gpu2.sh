cd /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/2.coding_and_inference
# 激活 conda 环境
conda init
source activate
conda activate MVDet_NEXT

# # TOCOM-TEM 9.42 KB 
# cp /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/1.feature_extraction/logs_feature_extraction/2024-09-09_14-58-44/MultiviewDetector.pth /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector2.pth
 
# CUDA_VISIBLE_DEVICES=2,3 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector2.pth" --epochs 30 --tau_1 2 --tau_2 2 --drop_prob 0.7

# sleep 10

# cp /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/1.feature_extraction/logs_feature_extraction/2024-09-09_14-58-44/MultiviewDetector.pth /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector2.pth
 
# CUDA_VISIBLE_DEVICES=2,3 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector2.pth" --epochs 30 --tau_1 2 --tau_2 2 --drop_prob 0.9

# # --error_camera 3,6,4
# cp /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/1.feature_extraction/logs_feature_extraction/2024-09-09_14-58-44/MultiviewDetector.pth /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth
 
# CUDA_VISIBLE_DEVICES=0,1 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth" --epochs 10 --tau_1 2 --tau_2 2 --drop_prob 0 --translation_error 0.0 --rotation_error 0.1 --error_camera 3,6,4

# cp /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/1.feature_extraction/logs_feature_extraction/2024-09-09_14-58-44/MultiviewDetector.pth /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth

# CUDA_VISIBLE_DEVICES=0,1 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth" --epochs 10 --tau_1 2 --tau_2 2 --drop_prob 0 --translation_error 0.0 --rotation_error 0.2 --error_camera 3,6,4

# cp /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/1.feature_extraction/logs_feature_extraction/2024-09-09_14-58-44/MultiviewDetector.pth /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth

# CUDA_VISIBLE_DEVICES=0,1 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth" --epochs 10 --tau_1 2 --tau_2 2 --drop_prob 0 --translation_error 0.0 --rotation_error 0.3 --error_camera 3,6,4

# cp /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/1.feature_extraction/logs_feature_extraction/2024-09-09_14-58-44/MultiviewDetector.pth /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth

# CUDA_VISIBLE_DEVICES=0,1 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth" --epochs 10 --tau_1 2 --tau_2 2 --drop_prob 0 --translation_error 0.0 --rotation_error 0.4 --error_camera 3,6,4

# cp /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/1.feature_extraction/logs_feature_extraction/2024-09-09_14-58-44/MultiviewDetector.pth /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth

# CUDA_VISIBLE_DEVICES=0,1 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth" --epochs 10 --tau_1 2 --tau_2 2 --drop_prob 0 --translation_error 0.0 --rotation_error 0.5 --error_camera 3,6,4

# CUDA_VISIBLE_DEVICES=0,1 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth" --epochs 10 --tau_1 2 --tau_2 2 --drop_prob 0 --translation_error 0.0 --rotation_error 0.5 --error_camera 3,5,6
 
cp /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/1.feature_extraction/logs_feature_extraction/2024-09-09_14-58-44/MultiviewDetector.pth /data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth

CUDA_VISIBLE_DEVICES=2,3 python main_coding_and_inference.py --dataset_path "/data1/fangzr/Research/PIB/Data/Wildtrack" --model_path "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS-2/2.coding_and_inference/logs/model_temp/MultiviewDetector.pth" --epochs 30 --tau_1 2 --tau_2 2 --drop_prob 0 
