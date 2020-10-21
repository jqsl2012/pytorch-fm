
#nohup python -u  main_params.py --model_name ipnn --dataset_path /home/eduapp/best_flow/20200907_more2more/all_features.train.fe_output.csv --device cpu --save_dir /home/eduapp/pytorch-fm/examples_prod/ --batch_size 2048 --epoch 2000   > run2.log.ipnn 2>&1 &
#tail -f run2.log.ipnn

# for best_flow: release/release-1.1.0
nohup python -u main_params.py --model_name opnn --dataset_path /home/eduapp/best_flow/20200907_more2more/a_9month_train.csv --device cpu --save_dir /home/eduapp/pytorch-fm/examples/model/ --batch_size 2048 --epoch 20   > run2.log.opnn 2>&1 &

tail -f run2.log.opnn