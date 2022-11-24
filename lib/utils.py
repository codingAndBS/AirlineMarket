import numpy as np

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data(args):
    route = np.load(args.route_file)
    
    train = np.load(args.train_file).reshape(-1,3600,8,7)[:,route,:,:].transpose(0,2,1,3) # [time, carries, routes, features]
    val = np.load(args.val_file).reshape(-1,3600,8,7)[:,route,:,:].transpose(0,2,1,3)
    test = np.load(args.test_file).reshape(-1,3600,8,7)[:,route,:,:].transpose(0,2,1,3)
    
    train = np.stack([train[:,0,:,:],train[:,3,:,:],train[:,5,:,:],train[:,7,:,:]], 1)
    val = np.stack([val[:,0,:,:],val[:,3,:,:],val[:,5,:,:],val[:,7,:,:]], 1)
    test = np.stack([test[:,0,:,:],test[:,3,:,:],test[:,5,:,:],test[:,7,:,:]], 1)
    
    train_data, train_label = train[...,:-1], train[...,-1]
    val_data, val_label = val[...,:-1], val[...,-1]
    test_data, test_label = test[...,:-1], test[...,-1]
    
    print('train data:', train_data.shape, train_label.shape)
    print('val data:', val_data.shape, val_label.shape)
    print('test data:', test_data.shape, test_label.shape)
    
    adj = np.load(args.prior_file)
    print('prior adj:', adj.shape)
    
    return train_data, train_label, val_data, val_label, test_data, test_label, adj

def load_optimize_data(args):
    route = np.load(args.route_file)
    mx = np.load(args.mx_file)
    
    data = np.load(args.optimize_file).reshape(3600,8,7)[route,:,:].transpose(1,0,2) # [carries, routes, features]
    data = np.stack([data[0,:,:],data[3,:,:],data[5,:,:],data[7,:,:]], 0)
    optimize_data = data[...,:-1]
    
    print('optimize data:', optimize_data.shape)
    
    adj = np.load(args.prior_file)
    print('prior adj:', adj.shape)
    
    cost_demand = np.load(args.cost_demand_file)
    cost = cost_demand['cost'][args.optimize_carry]
    demand = cost_demand['demand'].reshape(3600)[route]
        
    return optimize_data, adj, cost, demand, mx