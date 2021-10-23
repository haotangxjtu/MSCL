import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import os
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

def early_stopping(log_value, best_value, stopping_step, expected_order='recall', flag_step=100):
    # early stopping strategy:
    #assert expected_order in ['recall', 'ndcg']

    if (expected_order == 'recall' and log_value >= best_value)   :
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

 
#####
stopping_step = 0
should_stop = False
cur_best_pre_0=0.0
# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

save_path = join(world.BOARD_PATH,world.config['dataset'] +world.config['methods'] + world.config['info']+"-T" + str(world.config['temperature'])+"L"+str(world.config['lightGCN_n_layers'])+time.strftime("%m-%d-%Hh%Mm%Ss-") +".txt") 

d = os.path.dirname(save_path)
if not os.path.exists(d):
    os.makedirs(d) 
lossinfo=[]
epochtime=[]
try:
    for epoch in range(world.TRAIN_epochs):
        f = open(save_path, 'a') 
        start = time.time()

        if epoch ==0  : 
            f.write("Info :   ")
            for k,v in world.config.items():
                f.write( str(k)+","+str(v)+"\n")
            print("start")  


 
            
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        epochtime.append(time.time()-start)
       # print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}',world.config['dataset'] + world.config['info']+str(np.mean(epochtime)))
        
        if epoch > 0:
            cprint("[TEST]")
            outs=Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

            f.write("\n"+str(epoch)+":   ")
            for k,v in outs.items():
                f.write( str(k)+","+str(v))  
            cur_best_pre_0, stopping_step, should_stop = early_stopping(outs['recall'][0], cur_best_pre_0,
                                                                                stopping_step, expected_order='recall', flag_step=20)


        lossinfo.append(output_information)
        torch.save(Recmodel.state_dict(), weight_file)
        if epoch == range(world.TRAIN_epochs)[-1]:
            f.write( 'finish ,best recall '+str(cur_best_pre_0) )
            f.close()  

        if should_stop == True:
            f.write( 'early stop ,best recall '+str(cur_best_pre_0)+str(epoch) )
            f.close()
            break        
 
    f = open(save_path, 'a')
    for i, val in enumerate(lossinfo):
        f.write("\n"+ str(i+1)+","+str(val)) 
    f.close()

    print(f'last-time',world.config['dataset'] + world.config['info']+str(np.mean(epochtime)))
        


finally: 
    if world.tensorboard:
        w.close()