
nohup python -u  main_params.py --model_name ipnn --dataset_path /home/eduapp/best_flow/20200907_more2more/all_features.train.fe_output.csv --device cpu --save_dir /home/eduapp/pytorch-fm/examples_prod/ --batch_size 2048 --epoch 2000   > run2.log.ipnn 2>&1 &
tail -f run2.log.ipnn