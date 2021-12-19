python ./train_maml.py --trainK 1 --testK 5 --cuda_device cuda:0 > ./log/train_maml_train_${i}shot_test${i}shot.log &
python ./train_maml.py --trainK 5 --testK 5 --cuda_device cuda:1 > ./log/train_maml_train_${i}shot_test${i}shot.log
sudo kill -9 `ps -ef|grep train_maml|awk '{print $2}'`