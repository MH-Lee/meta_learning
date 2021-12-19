python ./train_maml.py --trainK 1 --testK 1 --cuda_device cuda:0 > ./log/train_maml_train_1shot_test1shot.log &
python ./train_maml.py --trainK 5 --testK 1 --cuda_device cuda:1 > ./log/train_maml_train_5shot_test1shot.log
sleep 300s
sudo kill -9 `ps -ef|grep train_maml|awk '{print $2}'`