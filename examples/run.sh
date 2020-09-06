
#python main_params.py --model_name lr --dataset_path /mnt/disk1/best_flow/git/pytorch-fm/examples/criteo/train.txt --device cpu --save_dir /mnt/disk1/best_flow/git/pytorch-fm/examples/save_dir --batch_size 12

rm -rf .criteo/
python main_params.py --model_name lr --dataset_path /mnt/disk1/best_flow/git/pytorch-fm/examples/train.txt --device cpu --save_dir /mnt/disk1/best_flow/git/pytorch-fm/examples/save_dir --batch_size 1024