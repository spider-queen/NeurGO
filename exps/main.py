import os
import sys
import pickle
import numpy as np
import torch
import argparse
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm

from NeurGO.problem import Problem
from NeurGO.model import NeurGO  
from train_sets import TRAIN_FUNCTIONS, gen_train_offset, get_train_fitness
from Benchmark.cecfunctions import FUNCTIONS as F
from Benchmark.bbobfunctions import FUNCTIONS as BBOBF
from Benchmark.utils import getFitness, genOffset, setOffset, getOffset

os.environ['CUDA_LAUNCH_BLOCKING']='1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class myProblem(Problem):
    def __init__(self, fun=None, repaire=True, dim=None):
        super().__init__()
        self.fun = fun
        self.useRepaire = repaire
        self.dim = dim
    
    def repaire(self, x):
        """Clamp the population within the search boundaries."""
        x = torch.clamp(x, self.fun['xlb'], self.fun['xub'])
        return x

    def calfitness(self, x):
        if self.useRepaire:
            x1 = self.repaire(x)
        else:
            x1 = x
            
        if self.fun['fid'].startswith('lf'):
            r = get_train_fitness(x1, self.fun)
            if r.dim() == 1:
                r = r.unsqueeze(-1) # (b,) -> (b, 1)
        else:
            r = getFitness(x1, self.fun)
            
        return r
    
    def genRandomPop(self, batchShape):
        lb = self.fun['xlb'] 
        ub = self.fun['xub']
        return torch.rand(batchShape, device=DEVICE) * (ub - lb) + lb

    def reoffset(self):
        genOffset(self.dim, self.fun)
        
    def setOffset(self, offset):
        for key in offset.keys():
            self.fun[key] = offset[key]

    def lossFunc(self, father, all_kcand, lamda=0.3):
        """NeurGO Quality-Diversity (QD) Loss"""
        father = self.repaire(father)
        all_kcand = self.repaire(all_kcand)
        fit_father = self.calfitness(father)            
        fit_kcand = self.calfitness(all_kcand)          
        tau = 1e-8

        # Z-score normalization
        fit_father = (fit_father - fit_father.mean(dim=1, keepdim=True)) / (fit_father.std(dim=1, keepdim=True) + tau)
        fit_kcand = (fit_kcand - fit_kcand.mean(dim=1, keepdim=True)) / (fit_kcand.std(dim=1, keepdim=True) + tau)

        # 1. Quality Loss
        best_father, _ = torch.min(fit_father, dim=1)
        best_kcand, _ = torch.min(fit_kcand, dim=1)
        r = best_kcand - best_father
        loss_quality = torch.mean(r)

        # 2. Diversity Loss
        b, K, d = all_kcand.shape
        cand_dist = torch.cdist(all_kcand, all_kcand, p=2)            
        scale = (self.fun['xub'] - self.fun['xlb']) * np.sqrt(d)      
        mean_dist = cand_dist.mean(dim=(1, 2))                        
        loss_diversity = -torch.mean(mean_dist / (scale + tau))       

        loss = loss_quality + lamda * loss_diversity
        return loss

    def getfunname(self):
        return self.fun['fid']
    
    def setfun(self, fun):
        self.fun = fun

class bbobProblem(Problem):
    def __init__(self, fun=None, repaire=True, dim=None):
        super().__init__()
        self.fun = fun
        self.useRepaire = repaire
        self.dim = dim
    
    def repaire(self, x):
        x = torch.clamp(x, self.fun['xlb'], self.fun['xub'])
        return x
 
    def calfitness(self, x):
        if self.useRepaire:
            x1 = self.repaire(x)
        else:
            x1 = x
        b, n, d = x.shape
        x1 = x1.reshape(-1, d)
        r = getFitness(x1, self.fun)
        r = torch.unsqueeze(r, -1)
        r = r.view((b, n))
        return r
    
    def genRandomPop(self, batchShape):
        lb = self.fun['xlb'] 
        ub = self.fun['xub']
        return torch.rand(batchShape, device=DEVICE) * (ub - lb) + lb

    def reoffset(self):
        genOffset(self.dim, self.fun)
        
    def setOffset(self, offset):
        for key in offset.keys():
            self.fun[key] = offset[key]
    
    def lossFunc(self, father, all_kcand, lamda=0.3):
        father = self.repaire(father)
        all_kcand = self.repaire(all_kcand)
        fit_father = self.calfitness(father)
        fit_kcand = self.calfitness(all_kcand)
        tau = 1e-8
        fit_father = (fit_father - fit_father.mean(dim=1, keepdim=True)) / (fit_father.std(dim=1, keepdim=True) + tau)
        fit_kcand = (fit_kcand - fit_kcand.mean(dim=1, keepdim=True)) / (fit_kcand.std(dim=1, keepdim=True) + tau)
        best_father, _ = torch.min(fit_father, dim=1)
        best_kcand, _ = torch.min(fit_kcand, dim=1)
        r = best_kcand - best_father
        loss_quality = torch.mean(r)
        b, K, d = all_kcand.shape
        cand_dist = torch.cdist(all_kcand, all_kcand, p=2)            
        scale = (self.fun['xub'] - self.fun['xlb']) * np.sqrt(d)      
        mean_dist = cand_dist.mean(dim=(1, 2))                        
        loss_diversity = -torch.mean(mean_dist / (scale + tau))       
        loss = loss_quality + lamda * loss_diversity
        return loss

    def getfunname(self):
        return self.fun['fid']
    
    def setfun(self, fun):
        self.fun = fun


# === Training Function ===
def train(expname=None, dim=None, hiddendim=None, popsize=100, problem=None,
          maxepoch=None, lr=None, batchsize=None, T=10, funset=None, needsave=True):
    
    # Initialization
    model = NeurGO(dim=dim, hidden_dim=hiddendim, popSize=popsize).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batchShape = (batchsize, popsize, dim)
    bar = tqdm(range(maxepoch), ncols=120)
    losslist = []
    minloss = None
    
    os.makedirs('./imgs/trainloss', exist_ok=True)
    os.makedirs('./ckpt', exist_ok=True)

    for epoch in bar:
        lamda = 0.5 * (1 - epoch / maxepoch) 
        if (epoch + 1) % 100 == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.9

        if epoch == 0 or (epoch + 1) % T == 0:
            for fun in funset:
                gen_train_offset(dim, fun)
            if len(losslist) > 0:
                plt.figure(figsize=(12, 9))
                plt.plot(losslist)
                plt.savefig(f'./imgs/trainloss/{expname}_d{dim}.png')
                plt.close()

        totalloss = None
        for fun in funset:
            problem.setfun(fun)
            pop = problem.genRandomPop(batchShape)
            offpop, trail, evalnums, all_candidates = model(pop, problem)
            loss = problem.lossFunc(pop, all_candidates, lamda)
            
            if totalloss is None:
                totalloss = loss
            else:
                totalloss += loss
        
        totalloss /= len(funset)
        opt.zero_grad()
        totalloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
        opt.step()
        losslist.append(totalloss.item())

        if (minloss is None or totalloss.item() < minloss) and needsave:
            minloss = totalloss.item()
            torch.save(model.state_dict(), f'./ckpt/{expname}_d{dim}.pth')
            
        bar.set_description(f"NeurGO({expname})_dim({dim})_loss:{totalloss.item():.6f}|min:{minloss:.6f}")

def test(expname=None, dim=None, hiddendim=None, popsize=100, problem=None,
       batchsize=None, runs=1):
    
    model = NeurGO(dim=dim, hidden_dim=hiddendim, popSize=popsize).to(DEVICE)
    
    ckpt_path = f'./ckpt/{expname}_d{dim}.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f'NeurGO checkpoint {ckpt_path} loaded!')
    else:
        print(f'Checkpoint not found: {ckpt_path}')
        return

    model.eval()
    batchShape = (batchsize, popsize, dim)
    bar = tqdm(range(runs))
    final_fits = []
    
    if problem.getfunname() == 'cecf1':
        offset = {'bias': torch.zeros(dim, device=DEVICE)}
    else:
        offset = {'bias': torch.zeros(dim, device=DEVICE)}
    problem.setOffset(offset)

    with torch.no_grad():
        for run in bar:
            pop = problem.genRandomPop(batchShape)
            offpop, trail, evalnums, all_candidates = model(pop, problem)
            batch_final_fit = trail[:, -1]
            final_fits.append(batch_final_fit)

    all_final_fits = torch.cat(final_fits, dim=0)
    print(f"NeurGO-{problem.getfunname()} | mean:{torch.mean(all_final_fits):.2E}({torch.std(all_final_fits):.2E})")
    
    totaltrail = {
        'final_fits': all_final_fits.detach().cpu().numpy(),
        'evalnums': evalnums,
        'mean': np.mean(all_final_fits.detach().cpu().numpy()),
        'std': np.std(all_final_fits.detach().cpu().numpy())
    }
    
    os.makedirs('./trails', exist_ok=True)
    with open(f'./trails/exp({expname})_f({problem.getfunname()})_dim({dim}).pkl', 'wb') as f:
        pickle.dump(totaltrail, f)

def expForFunction(dim=5, expname='neurgo_base',
                   popsize=100, maxepoch=1000, lr=0.001):
    print('Start training NeurGO...')
    funset = TRAIN_FUNCTIONS
    problem = myProblem(fun=TRAIN_FUNCTIONS[0], dim=dim, repaire=True)
    hiddendim = 200
    batchsize = 64
    T = 20
    train(expname=expname, dim=dim, hiddendim=hiddendim, popsize=popsize, 
          problem=problem, maxepoch=maxepoch, lr=lr, batchsize=batchsize, T=T, funset=funset)

def testSysFuns(dim=5, expname='neurgo_base', popsize=100):
    problem = myProblem(fun=None, dim=dim, repaire=True) 
    hiddendim = 200
    batchsize = 10
    testfunset = [F[f'cecf{i}'] for i in range(1, 7)]
    for fun in testfunset:
        problem.setfun(fun)
        test(expname=expname, dim=dim, hiddendim=hiddendim, popsize=popsize, problem=problem, batchsize=batchsize, runs=1)

def testBBOBFuns(expname=None, dim=None, hiddendim=200, popsize=100, problem=None,
       batchsize=10, runs=1):
    model = NeurGO(dim=dim, hidden_dim=hiddendim, popSize=popsize).to(DEVICE)
    ckpt_path = f'./ckpt/{expname}_d{dim}.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return
    model.eval()
    batchShape = (batchsize, popsize, dim)
    bar = tqdm(range(runs))
    final_fits = []
    with torch.no_grad():
        for run in bar:
            pop = problem.genRandomPop(batchShape)
            offpop, trail, evalnums, all_candidates = model(pop, problem)
            batch_final_fit = trail[:, -1]
            final_fits.append(batch_final_fit)
    all_final_fits = torch.cat(final_fits, dim=0)
    print(f"NeurGO-{problem.getfunname()} | mean:{torch.mean(all_final_fits):.2E}({torch.std(all_final_fits):.2E})")
    totaltrail = {
        'final_fits': all_final_fits.detach().cpu().numpy(),
        'evalnums': evalnums,
        'mean': np.mean(all_final_fits.detach().cpu().numpy()),
        'std': np.std(all_final_fits.detach().cpu().numpy())
    }
    with open(f'./trails/exp({expname})_f({problem.getfunname()})_dim({dim}).pkl', 'wb') as f:
        pickle.dump(totaltrail, f)
    return (problem.getfunname(), 'NeurGO', f'{totaltrail["mean"]:.2E}({totaltrail["std"]:.2E})')

def genBBOBoffset(dim=10):
    offsets = dict()
    bar = tqdm(range(1, 25))
    for fid in bar:
        f = BBOBF[fid]
        genOffset(dim, f)
        if not fid in [5, 24]:
            f['xopt'] = torch.zeros((dim,)).cuda()
        f['fopt'] = 0
        offsets[fid] = getOffset(f)
    with open(f'bbobOffsets_dim{dim}.pkl', 'wb') as f:
        pickle.dump(offsets, f)

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', '-d', required=True, type=int, help='a integer stands for the dimension')
    parser.add_argument('--expname', '-expname', required=True, type=str, default='neurgo_test')
    parser.add_argument('--popsize', '-popsize', required=True, type=int, default=100)
    parser.add_argument('--maxepoch', '-maxepoch', required=False, type=int, default=1000)
    parser.add_argument('--lr', '-lr', required=False, type=float, default=0.001)
    parser.add_argument('--mode', '-mode', required=True, type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--target', '-target', required=False, type=str, default='bbob', choices=['sys', 'bbob'])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseargs()
    dim = args.dim
    expname = args.expname
    popsize = args.popsize
    maxepoch = args.maxepoch
    lr = args.lr
    mode = args.mode
    target = args.target

    if mode == 'train':
        print(f'Ready to train NeurGO (dim:{dim}), expname:{expname}')
        expForFunction(dim=dim, expname=expname, popsize=popsize, maxepoch=maxepoch, lr=lr)
    else:
        if target == 'sys':
            print('Ready to test on CEC functions (F1-F6)')
            testSysFuns(dim=dim, expname=expname, popsize=popsize)
        
        if target == 'bbob':
            print('Ready to test on bbob functions')
            testfunset = [i for i in range(1, 25)]
            problem = bbobProblem(fun=BBOBF[1], dim=dim, repaire=True)
            if not os.path.exists(f'bbobOffsets_dim{dim}.pkl'):
                genBBOBoffset(dim)
            with open(f'bbobOffsets_dim{dim}.pkl', 'rb') as f:
                offsets = pickle.load(f)
                for fid in testfunset:
                    fun = BBOBF[fid]
                    fun['xlb'] = -5; fun['xub'] = 5
                    offset = offsets[fun['fid']]
                    setOffset(fun, offset)
                    problem.setfun(fun)
                    try:
                        testBBOBFuns(dim=dim, expname=expname, popsize=popsize, problem=problem)
                    except Exception as e:
                        print(f"Error testing function {fid}: {e}")