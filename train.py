from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import configparser
import math
import random

from lib import utils
from lib.utils import log_string, load_data, count_parameters
from model.msp import marketSharePrediction

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)
parser.add_argument('--cuda', type = int, default = config['train']['cuda'],
                    help = 'choose GPU')
parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'],
                    help = 'epoch to run')
parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'],
                    help = 'batch size')
parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'],
                    help = 'initial learning rate')
parser.add_argument('--seed', type = int, default = config['train']['seed'],
                    help = 'random seed')


parser.add_argument('--d', type = int, default = config['param']['d'], # 08 128
                    help = 'dims of model')

parser.add_argument('--train_file', default = config['file']['train_file'],
                    help = 'train file')
parser.add_argument('--val_file', default = config['file']['val_file'],
                    help = 'validation file')
parser.add_argument('--test_file', default = config['file']['test_file'],
                    help = 'tset file')
parser.add_argument('--route_file', default = config['file']['route_file'],
                    help = 'top k route file')
parser.add_argument('--model_file', default = config['file']['model_file'],
                    help = 'file to save pretrained parameters')
parser.add_argument('--log_file', default = config['file']['log_file'],
                    help = 'log file')
parser.add_argument('--prior_file', default = config['file']['prior_file'],
                    help = 'file contain prior knowledge')

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def res(model, valX, valY):
    model.eval()
    
    num_val = valX.shape[0]
    pred = []
    label = []
    
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                y_hat = model(X)

                pred.append(y_hat.cpu().numpy())
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)
    
    val_loss = np.mean(np.abs(pred-label))
    rmse = np.sqrt(np.mean(np.square(pred-label)))
    
    vx = label - np.mean(label)
    vy = pred - np.mean(pred)
    cc = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    
    r2 = 1 - np.sum((label-pred)**2) / np.sum((label-label.mean())**2)
    
    return val_loss, rmse, cc, r2

def train(model, trainX, trainY, valX, valY):
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    
    e = 0
    
    for epoch in tqdm(range(1,args.max_epoch+1)):
        model.train()
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainY = trainY[permutation]
        
        num_batch = math.ceil(num_train / args.batch_size)
        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(trainX[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)

                optimizer.zero_grad()

                y_hat = model(X)
                loss = torch.mean(torch.abs(y-y_hat))
                
                loss.backward()
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
                pbar.update(1)
                
        log_string(log, 'epoch %d, lr %.6f, train loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        
        val_loss, _, _, _ = res(model, valX, valY)
        log_string(log, 'validation loss: %.4f' % (val_loss))
        if val_loss < min_loss:
            e = epoch
            min_loss = val_loss
            torch.save(model.state_dict(), args.model_file)
    print('best epoch:', e)

def test(model, valX, valY):
    model.load_state_dict(torch.load(args.model_file))
    mae, rmse, cc, r2 = res(model, valX, valY)
    log_string(log, 'cc: %.4f, r2: %.4f, mae: %.4f, rmse: %.4f' % (cc, r2, mae, rmse))

if __name__ == '__main__':
    log_string(log, "loading data....")
    train_data, train_label, val_data, val_label, test_data, test_label, adj = load_data(args)
    log_string(log, "loading end....\n")

    log_string(log, "model constructed begin....")
    model = marketSharePrediction(6, args.d, 1, adj, device).to(device)
    print(count_parameters(model))
    log_string(log, "model constructed end....\n")
    
    log_string(log, "train begin....")
    train(model, train_data, train_label, val_data, val_label)
    log_string(log, "train end....\n")
    
    test(model, test_data, test_label)
