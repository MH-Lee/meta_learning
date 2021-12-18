import numpy as np
import torch
import random
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataloader import MiniImagenet
from proto.protonet import ConvNet, distance, accuracy, protonet_train
import argparse

import neptune.new as neptune

run = neptune.init(
    project="lmhoon012/meta-learning",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMTg4ODIyNC0xMDU0LTQzNzctOGU0Ni1kY2VkNWYwMTA1YzYifQ==",
)  # your credentials

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default='./datasets/miniimagenet/pkl_file/', type=str, help="root path")
    parser.add_argument("--trainN", default=5, type=int, help="trainN way")
    parser.add_argument("--trainK", default=1, type=int, help="trainK shot")
    parser.add_argument("--testN", default=5, type=int, help="testN way")
    parser.add_argument("--testK", default=1, type=int, help="testK shot")
    parser.add_argument("--Q", default=15, type=int, help="Num of query per class")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--train_iter", default=60000, type=int, help="num of iters in training")
    parser.add_argument("--val_iter", default=5000, type=int, help="num of iters in validation")
    parser.add_argument("--test_iter", default=10000, type=int, help="num of iters in testing")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--seed", default=777, type=int, help="random seed")
    parser.add_argument("--cuda_device", default="cuda:0", type=str, help="cuda device no.")
    parser.add_argument("--dist_mode", default='euclidean', type=str, help="distance metric")
    args = parser.parse_args()
    
    params = {'trainN': args.trainN, 'trainK':args.trainK, 'Q':args.Q, \
              'testN' : args.testN, 'testK': args.testK, 'train_iter':args.train_iter,\
              'val_iter':args.val_iter, 'test_iter':args.test_iter, 'dist_mode':args.dist_mode,\
              'batch_size':args.batch_size, 'lr':args.lr, 'seed': args.seed}
    
    run["parameters"] = params
    
    device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')

    root_path = args.root_path
    trainN = args.trainN
    trainK = args.trainK
    testN = args.testN
    testK = args.testK
    Q = args.Q
    batch_size = args.batch_size
    set_seed(args.seed)
    
    train_dataset = MiniImagenet(path=root_path, N=trainN, K=trainK, Q=Q, \
                                 mode='train', total_iter=args.train_iter)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,\
                            shuffle=True, num_workers=1)
    val_dataset = MiniImagenet(path=root_path, N=testN, K=testK, Q=Q,\
                               mode='validation', total_iter=args.val_iter)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,\
                            shuffle=True, num_workers=1)
    test_dataset = MiniImagenet(path=root_path, N=testN, K=testK, Q=Q,\
                                mode='test', total_iter=args.test_iter)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,\
                            shuffle=True, num_workers=1)
    
    model = ConvNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler =  torch.optim.lr_scheduler.StepLR(
                                optimizer, step_size=20, gamma=0.5
                            )
    
    iteration = 0
    for task_batch in train_loader:
        torch.cuda.empty_cache()
        model.train()
        meta_train_error = []
        meta_train_accuracy = []
        for batch in zip(*task_batch):
            loss, acc = protonet_train(batch, model, metric=distance, \
                                      criterion=criterion,\
                                      ways=trainN, shots=trainK,\
                                      device=device, dist_mode=args.dist_mode)
            meta_train_error.append(loss.item())
            meta_train_accuracy.append(acc.item())
            run["train/loss"].log(loss.item())
            run["train/accuracy"].log(acc.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
        lr_scheduler.step()


        if iteration % 1000 == 0:
            print('Iteration', iteration)
            print('Meta Train Error : ', round(np.mean(meta_train_error),4),\
                  ' std : ', round(np.std(meta_train_error), 4))
            print('Meta Train Accuracy : ', round(np.mean(meta_train_accuracy), 4),\
                  ' std : ', round(np.std(meta_train_accuracy), 4))
        
        if iteration % 5000 == 0:
            # Compute meta-validation loss
            meta_valid_error = []
            meta_valid_accuracy = []
            model.eval()
            for task_batch in val_loader:
                for batch in zip(*task_batch):
                    val_loss, val_acc = protonet_train(batch, model, metric=distance, \
                                                      criterion=criterion,\
                                                      ways=testN, shots=testK,\
                                                      device=device, dist_mode=args.dist_mode)
                    meta_valid_error.append(val_loss.item())
                    meta_valid_accuracy.append(val_acc.item())
            run["eval/loss"].log(round(np.mean(meta_valid_error), 4))
            run["eval/accuracy"].log(round(np.mean(meta_valid_accuracy), 4))
            run["eval/Accuracy_mean"] = round(np.mean(meta_valid_accuracy), 4)
            run["eval/Accuracy_std"] = round(np.std(meta_valid_accuracy), 4)
            print('Valid Result')
            print('Meta Valid Error : ', round(np.mean(meta_valid_error),4),\
                  ' std : ', round(np.std(meta_valid_error), 4))
            print('Meta Valid Accuracy : ', round(np.mean(meta_valid_accuracy), 4),\
                  ' std : ', round(np.std(meta_valid_accuracy), 4))
    
    meta_test_error = []
    meta_test_accuracy = []
    model.eval()
    for task_batch in test_loader:
        for batch in zip(*task_batch):
            test_loss, test_acc = protonet_train(batch, model, metric=distance,\
                                      criterion=criterion,\
                                      ways=testN, shots=testK,\
                                      device=device, dist_mode=args.dist_mode)
            meta_test_error.append(test_loss.item())
            meta_test_accuracy.append(test_acc.item())
    run["test/Accuracy_mean"] = round(np.mean(meta_test_accuracy), 4)
    run["test/Accuracy_std"] = round(np.std(meta_test_accuracy), 4)            
    print('Meta Test Result')
    print('Meta test Error : ', round(np.mean(meta_test_error),4),\
            " std : ", round(np.std(meta_test_error), 4))
    print('Meta test Accuracy : ', round(np.mean(meta_test_accuracy), 4),\
            " std : ", round(np.std(meta_test_accuracy), 4))

if __name__ == '__main__':
    main()
