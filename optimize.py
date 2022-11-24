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
from lib.utils import log_string, load_optimize_data, count_parameters
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

parser.add_argument('--d', type = int, default = config['param']['d'], 
                    help = 'dimentions of model')
parser.add_argument('--optimize_carry', type = int, default = config['param']['optimize_carry'],
                    help = 'airlines of optimizting')

parser.add_argument('--optimize_file', default = config['file']['optimize_file'],
                    help = 'optimize file')
parser.add_argument('--cost_demand_file', default = config['file']['cost_demand_file'],
                    help = 'cost and demand file')
parser.add_argument('--mx_file', default = config['file']['mx_file'],
                    help = 'max file')
parser.add_argument('--prior_file', default = config['file']['prior_file'],
                    help = 'file save prior topo knowledge')
parser.add_argument('--route_file', default = config['file']['route_file'],
                    help = 'top k route file')
parser.add_argument('--model_file', default = config['file']['model_file'],
                    help = 'file to save pretrained parameters')
parser.add_argument('--log_file', default = config['file']['log_file'],
                    help = 'log file')

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def train(model, optimize_data, cost, demand, mx):
    best_market = 0
    budget = torch.from_numpy(np.array([(cost * optimize_data[args.optimize_carry,:,-2]).sum()])).to(device)
    cost = torch.from_numpy(cost).to(device)
    demand = torch.from_numpy(demand).to(device)
    mx = torch.from_numpy(mx).float().to(device)
    e = 0
    
    learned_lambda = nn.Parameter(torch.zeros(1).to(device))
    learned_freq = nn.Parameter(torch.from_numpy(optimize_data[args.optimize_carry,:,-2]).to(device))
    
    lambda_optimizer = torch.optim.Adam([learned_lambda], lr=0.01)
    theta_optimizer = torch.optim.Adam([learned_freq], lr=1)
    
    model.eval()
        
    for epoch in range(1,args.max_epoch+1):
        if epoch == 1:
            with torch.no_grad():
                optimize_data_torch = torch.from_numpy(optimize_data).float().to(device)
                optimize_data_torch[args.optimize_carry,:,-2] = learned_freq
                market = model(optimize_data_torch.unsqueeze(0)).squeeze(0)[args.optimize_carry]
                ori_market = (market*demand).sum().cpu().detach().numpy()
        
        optimize_data_torch = torch.from_numpy(optimize_data).float().to(device)
        optimize_data_torch[args.optimize_carry,:,-2] = learned_freq
        theta_optimizer.zero_grad()
        market = model(optimize_data_torch.unsqueeze(0)).squeeze(0)
        theta_loss = -((market[args.optimize_carry] * demand).sum() - learned_lambda * ((cost * learned_freq).sum() - budget))
        theta_loss.backward()
        theta_optimizer.step()
        
        optimize_data_torch = torch.from_numpy(optimize_data).float().to(device)
        optimize_data_torch[args.optimize_carry,:,-2] = learned_freq
        lambda_optimizer.zero_grad()
        market = model(optimize_data_torch.unsqueeze(0)).squeeze(0)
        lambda_loss = (market[args.optimize_carry] * demand).sum() - learned_lambda * ((cost * learned_freq).sum() - budget)
        lambda_loss.backward()
        lambda_optimizer.step()
        
        learned_freq = nn.Parameter(torch.where(torch.from_numpy(learned_freq.cpu().detach().numpy()).float().to(device)>0, torch.from_numpy(learned_freq.cpu().detach().numpy()).float().to(device), torch.zeros(1).float().to(device)[0]))
        learned_freq = nn.Parameter(torch.where(torch.from_numpy(learned_freq.cpu().detach().numpy()).float().to(device)<mx, torch.from_numpy(learned_freq.cpu().detach().numpy()).float().to(device), mx))
        theta_optimizer = torch.optim.Adam([learned_freq], lr=1)
        
        
        with torch.no_grad():
            freq_i = learned_freq.cpu().detach().numpy()
            test_data = torch.from_numpy(optimize_data).float().to(device)
            test_data[args.optimize_carry,:,-2] = torch.from_numpy(freq_i).to(device)
            
            market = model(test_data.unsqueeze(0)).squeeze(0)[args.optimize_carry]
            if epoch % 100 == 0:
                log_string(log, 'epoch %d, loss %.4f, learned lambda %.4f, learned passengers %.2f'
                      % (epoch, theta_loss, learned_lambda, (market*demand).sum()))
            
            if (cost * learned_freq).sum() - budget < 0 and (market*demand).sum().cpu().detach().numpy() > best_market:
                best_market = (market*demand).sum().cpu().detach().numpy()
                e = epoch
    
    print('best epoch:', e)
    print('original passengers', ori_market)
    print('optimized passengers', best_market)

if __name__ == '__main__':
    st = time.time()
    
    log_string(log, "loading data....")
    optimize_data, adj, cost, demand, mx = load_optimize_data(args)
    log_string(log, "loading end....\n")

    log_string(log, "model constructed begin....")
    model = marketSharePrediction(6, args.d, 1, adj, device).to(device)
    log_string(log, "model constructed end....\n")
    model.load_state_dict(torch.load(args.model_file))
    print(count_parameters(model))
    
    log_string(log, "optimize begin....")
    train(model, optimize_data, cost, demand, mx)
    log_string(log, "optimize end....\n")
    
    print('time: ',time.time()-st)
    