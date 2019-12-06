
# Building bts compute_depth layer
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/lib/nvidia-cuda-toolkit ..
cmake -D CUDA_TOOLKIT_ROOT_DIR=/home/burakov/anaconda3/envs/cv36 ..

# Testing model
python bts_test.py @arguments_test_nyu.txt --data_path /media/burakov/Data1/Data/nyu_depth_v2/nyu_depth_v2_labeled_vis/test/

