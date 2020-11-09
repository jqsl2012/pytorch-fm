ps -ef|grep main_params

# v1.1.0
#nohup python -u main_params.py --model_name xdfm --dataset_path /home/eduapp/best_flow/release-1.1.0/train_data/202010/dnn_part_train.csv --device cpu --save_dir /home/eduapp/pytorch-fm/examples/model/ --batch_size 2048 --epoch 10   > run2.log.xdfm 2>&1 &
nohup python -u main.py --model_name dfm --dataset_path /home/eduapp/best_flow/release-1.1.0/train_data/202010/dnn_part_train.csv --device cpu --save_dir /home/eduapp/pytorch-fm/examples/model/ --batch_size 2048 --epoch 10   > run2.log.dfm 2>&1 &

tail -f run2.log.dfm