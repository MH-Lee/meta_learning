for i in 5 10 20
do
   echo "train $i ways"
   python ./train_protonet.py --trainN $i --trainK 1 --cuda_device cuda:0 > ./log/train_${i}ways_1shot.log &
   python ./train_protonet.py --trainN $i --trainK 5 --cuda_device cuda:1 > ./log/train_${i}ways_5shot.log
   sleep 180s
done
sudo kill -9 `ps -ef|grep train_proto|awk '{print $2}'`