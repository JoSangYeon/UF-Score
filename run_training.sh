# python make_ablation.py 

# #Train img
# CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/ablation/valor/va_train_1_50 --train_file ablation_data/valor/va_train_1_50.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/ablation/valor/va_train_1_75 --train_file ablation_data/valor/va_train_1_75.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/ablation/valor/va_train_1_90 --train_file ablation_data/valor/va_train_1_90.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/ablation/valor/va_train_9_50 --train_file ablation_data/valor/va_train_9_50.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/ablation/valor/va_train_9_75 --train_file ablation_data/valor/va_train_9_75.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/ablation/valor/va_train_9_90 --train_file ablation_data/valor/va_train_9_90.data --data_dir data --rank 1 --world_size 1

# #Train audio
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/vggsound/va_train_1_50 --train_file ablation_data/vggsound/va_train_1_50.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/vggsound/va_train_1_75 --train_file ablation_data/vggsound/va_train_1_75.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/vggsound/va_train_1_90 --train_file ablation_data/vggsound/va_train_1_90.data --data_dir data --rank 1 --world_size 1
# # CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/vggsound/va_train_9_50 --train_file ablation_data/vggsound/va_train_9_50.data --data_dir data --rank 1 --world_size 1
# # CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/vggsound/va_train_9_75 --train_file ablation_data/vggsound/va_train_9_75.data --data_dir data --rank 1 --world_size 1
# # CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/vggsound/va_train_9_90 --train_file ablation_data/vggsound/va_train_9_90.data --data_dir data --rank 1 --world_size 1


# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/valor/audio/va_train_1_50 --train_file ablation_data/valor/va_train_1_50.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/valor/audio/va_train_1_75 --train_file ablation_data/valor/va_train_1_75.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/valor/audio/va_train_1_90 --train_file ablation_data/valor/va_train_1_90.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/valor/audio/va_train_9_50 --train_file ablation_data/valor/va_train_9_50.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/ablation/valor/audio/va_train_9_75 --train_file ablation_data/valor/va_train_9_75.data --data_dir data --rank 1 --world_size 1
# 

# #* Test Audio Valor
# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/audio/va_train_1_50/checkpoint-3630/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/audio/va_train_1_50/metric/3630
# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/audio/va_train_1_75/checkpoint-5445/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/audio/va_train_1_75/metric/5445
# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/audio/va_train_1_90/checkpoint-6540/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/audio/va_train_1_50/metric/6540

# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/audio/va_train_9_50/checkpoint-3630/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/audio/va_train_9_50/metric/3630
# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/audio/va_train_9_75/checkpoint-5445/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/audio/va_train_9_75/metric/5445
# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/audio/va_train_9_90/checkpoint-6540/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/audio/va_train_9_50/metric/6540



# #* Test Img Valor

# CUDA_VISIBLE_DEVICES=6 python test_img.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/img/va_train_1_50/checkpoint-2420/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/img/va_train_1_50/metric/2420
# CUDA_VISIBLE_DEVICES=6 python test_img.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/img/va_train_1_75/checkpoint-3630/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/img/va_train_1_50/metric/3630
# CUDA_VISIBLE_DEVICES=6 python test_img.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/img/va_train_1_90/checkpoint-4360/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/img/va_train_1_50/metric/4360

# CUDA_VISIBLE_DEVICES=6 python test_img.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/img/va_train_9_50/checkpoint-2420/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/img/va_train_1_50/metric/2420
# CUDA_VISIBLE_DEVICES=6 python test_img.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/img/va_train_9_75/checkpoint-3630/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/img/va_train_1_50/metric/3630
# CUDA_VISIBLE_DEVICES=6 python test_img.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/valor/img/va_train_9_90/checkpoint-4360/pytorch_model.bin --test_data data/va_test.data --result_dir result/ablation/valor/img/va_train_1_50/metric/4360


# #* Test audio VGG Sound
# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/vggsound/va_train_1_50/checkpoint-26700/pytorch_model.bin --test_data data/vs_test.data --result_dir result/ablation/valor/audio/va_train_9_50/metric/26700
# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/vggsound/va_train_1_75/checkpoint-40050/pytorch_model.bin --test_data data/vs_test.data --result_dir result/ablation/valor/audio/va_train_9_75/metric/40050
# CUDA_VISIBLE_DEVICES=6 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/ablation/vggsound/va_train_1_90/checkpoint-48060/pytorch_model.bin --test_data data/vs_test.data --result_dir result/ablation/valor/audio/va_train_9_50/metric/48060


# #* Train VGGSound 90
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/aud/vggsound_90 --train_file vs_train_90.data --data_dir data --rank 1 --world_size 2
# CUDA_VISIBLE_DEVICES=7 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/aud/vggsound_90 --train_file vs_train_90.data --data_dir data --rank 2 --world_size 2


#* Train BLIP Valor
#CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/blip/va_train_90 --train_file blip_data/va_train_90.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/blip/va_train_75 --train_file blip_data/va_train_75.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_img.py --config_path conf/finetune_clo.yaml --out_dir result/blip/va_train_50 --train_file blip_data/va_train_50.data --data_dir data --rank 1 --world_size 1


# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/blap/valor/vs_train_90 --train_file blap_data/valor/va_train_90.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/blap/valor/vs_train_75 --train_file blap_data/valor/va_train_75.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/blap/valor/vs_train_50 --train_file blap_data/valor/va_train_50.data --data_dir data --rank 1 --world_size 1

# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/blap/vggsound/vs_train_90 --train_file blap_data/vggsound/vs_train_90.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/blap/vggsound/vs_train_75 --train_file blap_data/vggsound/vs_train_75.data --data_dir data --rank 1 --world_size 1
# CUDA_VISIBLE_DEVICES=6 python train_audio.py --config_path conf/finetune_ac.yaml --out_dir result/blap/vggsound/vs_train_50 --train_file blap_data/vggsound/vs_train_50.data --data_dir data --rank 1 --world_size 1


# CUDA_VISIBLE_DEVICES=5 python test_audio.py --config_path conf/finetune_ac.yaml --checkpoint result/blap/valor/vs_train_75/checkpoint-5445/pytorch_model.bin --test_data data/va_test.data --result_dir result/blap/valor/vs_train_75/metric/5445
# CUDA_VISIBLE_DEVICES=5 python test_img.py --config_path conf/finetune_clo.yaml --checkpoint result/blip/va_train_75/checkpoint-3630/pytorch_model.bin --test_data data/va_test.data --result_dir result/blip/va_train_75/metric/3630

CUDA_VISIBLE_DEVICES=5 python utils/compute_metrics.py --test_data data/va_test.data --predict_dir result/blap/valor/vs_train_75/metric/5445
CUDA_VISIBLE_DEVICES=5 python utils/compute_metrics.py --test_data data/va_test.data --predict_dir result/blip/va_train_75/metric/3630