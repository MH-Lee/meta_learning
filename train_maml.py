import numpy as np
import torch
import random
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataloader import MiniImagenet
from common.cnn_models import FewshotClassifier
from maml.models import MAML
from maml.utils import fast_adapt
from tqdm import tqdm
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
    parser.add_argument("--N", default=5, type=int, help="N way")
    parser.add_argument("--K", default=1, type=int, help="K shot")
    parser.add_argument("--Q", default=15, type=int, help="Num of query per class")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--train_iter", default=60000, type=int, help="num of iters in training")
    parser.add_argument("--val_iter", default=10000, type=int, help="num of iters in validation")
    parser.add_argument("--test_iter", default=60000, type=int, help="num of iters in testing")
    parser.add_argument("--adaptation_steps", default=1, type=int, help="num of adatation steps")
    parser.add_argument("--seed", default=777, type=int, help="random seed")
    parser.add_argument("--meta_lr", default=0.003, type=float, help="meta learner learning rate")
    parser.add_argument("--fast_lr", default=0.5, type=float, help="fast adaptation learning rate")
    parser.add_argument("--cuda_device", default="cuda:0", type=str, help="cuda device no.")
    parser.add_argument("--first_order", action="store_true", help="first order")
    args = parser.parse_args()
    
    params = {'N': args.N, 'K':args.K, 'Q':args.Q, 'train_iter':args.train_iter, \
              'val_iter':args.val_iter, 'test_iter':args.test_iter, \
              'adaptation_steps':args.adaptation_steps, 'batch_size':args.batch_size,\
              'first_order':args.first_order,'meta_lr':args.meta_lr, 'fast_lr':args.fast_lr,\
              'seed': args.seed}
    
    run["parameters"] = params
    
    device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')

    root_path = args.root_path
    N = args.N
    Q = args.Q
    K = args.K
    batch_size = args.batch_size
    set_seed(args.seed)
    
    train_dataset = MiniImagenet(path=root_path, N=N, K=K, Q=Q, \
                                 mode='train', total_iter=args.train_iter)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,\
                            shuffle=True, num_workers=1)
    val_dataset = MiniImagenet(path=root_path, N=N, K=K, Q=Q,\
                               mode='validation', total_iter=args.val_iter)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,\
                            shuffle=True, num_workers=1)
    test_dataset = MiniImagenet(path=root_path, N=N, K=K, Q=Q,\
                                mode='test', total_iter=args.test_iter)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,\
                            shuffle=True, num_workers=1)
    
    model = FewshotClassifier(output_size=N)
    model.to(device)
    maml = MAML(model, lr=args.fast_lr, first_order=False)
    maml.to(device)
    opt = optim.Adam(maml.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    
    iteration = 0
    for task_batch in train_loader:
        torch.cuda.empty_cache()
        opt.zero_grad()
        meta_train_error = []
        meta_train_accuracy = []
        for batch in zip(*task_batch):
            learner = maml.clone()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adaptation_steps,
                                                               device)
            evaluation_error.backward()
            meta_train_error.append(evaluation_error.item())
            meta_train_accuracy.append(evaluation_accuracy.item())
            run["train/loss"].log(evaluation_error.item())
            run["train/accuracy"].log(evaluation_accuracy.item())
            iteration += 1 
        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / len(task_batch))
        opt.step()

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
            for task_batch in val_loader:
                for batch in zip(*task_batch):
                    learner = maml.clone()
                    evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                    learner,
                                                                    loss,
                                                                    args.adaptation_steps,
                                                                    device)
                    meta_valid_error.append(evaluation_error.item())
                    meta_valid_accuracy.append(evaluation_accuracy.item())
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
    for task_batch in test_loader:
        for batch in zip(*task_batch):
            learner = maml.clone()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                            learner,
                                                            loss,
                                                            args.adaptation_steps,
                                                            device)
            meta_test_error.append(evaluation_error.item())
            meta_test_accuracy.append(evaluation_accuracy.item())
            torch.cuda.empty_cache()
    run["test/Accuracy_mean"] = round(np.mean(meta_test_accuracy), 4)
    run["test/Accuracy_std"] = round(np.std(meta_test_accuracy), 4)            
    print('Meta Test Result')
    print('Meta test Error : ', round(np.mean(meta_test_error),4),\
            " std : ", round(np.std(meta_test_error), 4))
    print('Meta test Accuracy : ', round(np.mean(meta_test_accuracy), 4),\
            " std : ", round(np.std(meta_test_accuracy), 4))

if __name__ == '__main__':
    main()
