from logging import log
import numpy as np
import torch
import random
import pandas as pd
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataloader import MiniImagenet
from proto.protonet import ConvNet, pairwise_distances_logits, protonet_train
import argparse
import os
import neptune.new as neptune
from credentials.api_key import API_KEY, PROJECT

run = neptune.init(
    project=PROJECT,
    api_token=API_KEY,
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
    parser.add_argument("--testK", default=5, type=int, help="testK shot")
    parser.add_argument("--Q", default=15, type=int, help="Num of query per class")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--train_iter", default=60000, type=int, help="num of iters in training")
    parser.add_argument("--val_iter", default=1000, type=int, help="num of iters in validation")
    parser.add_argument("--test_iter", default=600, type=int, help="num of iters in testing")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--seed", default=777, type=int, help="random seed")
    parser.add_argument("--cuda_device", default="cuda:0", type=str, help="cuda device no.")
    args = parser.parse_args()
    
    params = {'trainN': args.trainN, 'trainK':args.trainK, 'Q':args.Q, \
              'testN' : args.testN, 'testK': args.testK, 'train_iter':args.train_iter,\
              'val_iter':args.val_iter, 'test_iter':args.test_iter,\
              'batch_size':args.batch_size, 'lr':args.lr, 'seed': args.seed}
    
    run["parameters"] = params
    
    device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')
    os.makedirs('./log/', exist_ok=True)
    root_path = args.root_path
    trainN = args.trainN
    trainK = args.trainK
    testN = args.testN
    testK = args.testK
    Q = args.Q
    batch_size = args.batch_size
    set_seed(args.seed)
    print(device)
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
    lr_scheduler =  torch.optim.lr_scheduler.StepLR(
                                optimizer, step_size=20, gamma=0.5
                            )
    log_df = pd.DataFrame(columns=['iteration', 'mean_acc', 'std_acc', \
                                    'mean_loss', 'std_loss', 'mode'])
    for iterations, task_batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        model.train()
        meta_train_error = []
        meta_train_accuracy = []
        sx, sy, qx, qy  = task_batch
        data = torch.cat((sx, qx), dim=1)
        labels = torch.cat((sy, qy), dim=1)
        batch = (data, labels)
        loss, acc = protonet_train(model,
                                   batch,
                                   trainN,
                                   trainK,
                                   Q,
                                   metric=pairwise_distances_logits,
                                   device=device)

        meta_train_error.append(loss.item())
        meta_train_accuracy.append(acc.item())
        run["train/loss"].log(loss.item())
        run["train/accuracy"].log(acc.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if iterations % 500 == 0:
            lr_scheduler.step()
            print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
                                    iterations, np.mean(meta_train_error),\
                                    np.mean(meta_train_accuracy)))
            model.eval()
            log_df = log_df.append({'iteration':iterations,\
                                    'mean_acc': np.mean(meta_train_accuracy),\
                                    'std_acc':np.std(meta_train_accuracy), \
                                    'mean_loss': np.mean(meta_train_error),\
                                    'std_loss': np.std(meta_train_error),\
                                    'mode':'train'}, ignore_index=True)
            meta_valid_error = []
            meta_valid_accuracy = []
            for batch in val_loader:
                sx, sy, qx, qy  = batch
                data = torch.cat((sx, qx), dim=1)
                labels = torch.cat((sy, qy), dim=1)
                batch = (data, labels)
                val_loss, val_acc = protonet_train(model,
                                                   batch,
                                                   testN,
                                                   testK,
                                                   Q,
                                                   metric=pairwise_distances_logits,
                                                   device=device)
                meta_valid_error.append(val_loss.item())
                meta_valid_accuracy.append(val_acc.item())
            run["eval/loss"].log(round(np.mean(meta_valid_error), 4))
            run["eval/accuracy"].log(round(np.mean(meta_valid_accuracy), 4))
            run["eval/Accuracy_mean"] = round(np.mean(meta_valid_accuracy), 4)
            run["eval/Accuracy_std"] = round(np.std(meta_valid_accuracy), 4)
            log_df = log_df.append({'iteration':iterations,\
                                    'mean_acc': np.mean(meta_valid_accuracy),\
                                    'std_acc':np.std(meta_valid_accuracy), \
                                    'mean_loss': np.mean(meta_valid_error),\
                                    'std_loss': np.std(meta_valid_error),\
                                    'mode':'validation'}, ignore_index=True)
            print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
                  iterations, np.mean(meta_valid_error),\
                  np.mean(meta_valid_accuracy)))
    meta_test_error = []
    meta_test_accuracy = []
    for batch in test_loader:
        sx, sy, qx, qy  = batch
        data = torch.cat((sx, qx), dim=1)
        labels = torch.cat((sy, qy), dim=1)
        batch = (data, labels)
        test_loss, test_acc = protonet_train(model,
                                   batch,
                                   testN,
                                   testK,
                                   Q,
                                   metric=pairwise_distances_logits,
                                   device=device)
        meta_test_error.append(test_loss.item())
        meta_test_accuracy.append(test_acc.item())
    run["test/loss_mean"] = round(np.mean(meta_test_error), 4)
    run["test/loss_std"] = round(np.std(meta_test_error), 4)            
    run["test/Accuracy_mean"] = round(np.mean(meta_test_accuracy), 4)
    run["test/Accuracy_std"] = round(np.std(meta_test_accuracy), 4)            
    print('Meta Test Result')
    print('epoch {}, test, loss={:.4f} acc={:.4f}'.format(
                  iterations, np.mean(meta_test_error),\
                  np.mean(meta_test_accuracy)))
    log_df = log_df.append({'iteration':iterations,\
                            'mean_acc': np.mean(meta_test_accuracy),\
                            'std_acc':np.std(meta_test_accuracy), \
                            'mean_loss': np.mean(meta_test_error),\
                            'std_loss': np.std(meta_test_error),\
                            'mode':'test'}, ignore_index=True)
    print('epoch {}, test, loss={:.4f} acc={:.4f}'.format(
         iterations, np.mean(meta_test_error),\
         np.mean(meta_test_accuracy)))
    log_df.to_csv('./log/protonet_train_log_{}ways_{}shot.csv'.format(trainN, trainK), index=False)
    
if __name__ == '__main__':
    main()
