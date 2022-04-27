import argparse
import os
# from tabnanny import check
# from turtle import update
from typing import *
import pennylane as qml
import pennylane.numpy as qnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.stats import truncnorm, beta
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
import json


#### utils #### 
def checkmkdirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def load_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

##### Data loading #### 

def load_iris(src: str):
    data = np.loadtxt(src)
    dims = data.shape[1]
    X = data[:, 0:dims-1]
    y = data[:, -1]
    print(f"Num samples in Iris: {len(X)}")
    return X, y

def load_wine(src: str):
    data = np.load(src)
    dims = data.shape[1]
    X = data[:, 0:dims-1]
    y = data[:, -1]
    print(f"Num samples in Wine: {len(X)}")
    return X, y

def save_ckpt(fname, obj):
    with open(fname, 'wb') as ofile:
        pickle.dump(obj, ofile)



#### circuits #### 

# 1. McClean et. al's random circuit
def rand_ckt(params, gate_seq=None, nqubits=None):
    for i in range(nqubits):
        qml.RY(qnp.pi/4, wires=i)
    # print(gate_seq)
    # random parametrized gate sequence
    for i in range(nqubits):
        gate_seq[i](params[i], wires=i)
    
    # CZ gate
    for i in range(nqubits-1):
        qml.CZ(wires=[i, i+1])
        
    
    H = np.zeros((2**nqubits, 2**nqubits)) # expensive
    H[0, 0] = 1
    wlist = [i for i in range(nqubits)]
    return qml.expval(qml.Hermitian(H, wlist))

# 2. pennylane default tutorial circuit 
def statepreparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)

def layer(params):
    # params: [nqubits x rot-params]
    # nqubits = nfeatures
    nfeats = params.shape[0]
    for i in range(nfeats):
        qml.Rot(params[i, 0], params[i,1], params[i, 2], wires=i)
    
    for i in range(nfeats - 1):
        qml.CNOT(wires=[i, i+1])
        
    # qml.Rot(params[1,0], params[1, 1], params[1, 2], wires=1)
    

def pnl_ckt(params, data):
    """Pennylane circuit 1: Directly lifted from
    pennylane's tutorial, this circuit accepts data
    that has already been angle embedded and then prepares
    an initial state from that angle embedded data.
    The params are randomly initialized angles for the rotation
    angles. 
    
    Params shape: (N-layers x nqubits x 3)
    data shape: (N-samples x n-dim)
    """
    statepreparation(data)
    for param in params:
        layer(param)
    return qml.expval(qml.PauliZ(0))

# 3. pennylane angle embedding template

def pnl_ckt_v2(params, data):
    """ Pennylane circuit 2: This circuit uses
    classical data and internally angle embeds the data. The params
    are initialized as before.
    
    LIMITATION: The number of qubits(n) must be equal to or greater than
    the number of dimensions of the data(N). (N <= n)
    """
    assert len(params.shape) == 3, f"Expected param shape len: 3, got {len(params.shape)}"
    nqubits = params.shape[1]
    qml.AngleEmbedding(data, wires=range(nqubits), rotation="X")
    for param in params:
        layer(param)
    
    return qml.expval(qml.PauliZ(0))

### initializations ### 

def init_beta(a, b, shape):
    sz = np.prod(shape)
    return qnp.random.beta(a, b, size=sz).reshape(shape)

def init_trunc_normal(l, h, shape):
    sz = np.prod(shape)
    r = truncnorm.rvs(l, h, size=sz).reshape(shape)
    return qnp.array(r, requires_grad=True)

def init_uniform(l, h, shape):
    sz = np.prod(shape)
    return qnp.random.uniform(l, h, size=sz).reshape(shape)


def init_beta_ebayes(data, shape):
    data_r = data.reshape(data.size)
    sz = np.prod(shape)
    dmin, dmax = np.min(data_r), np.max(data_r)
    for i in range(len(data_r)):
        data_r[i] = (data_r[i] - dmin) / (dmax - dmin)
    zmask = data_r <= 0
    omask = data_r >= 1
    data_r[zmask] = data_r[zmask] + 1e-8
    data_r[omask] = data_r[omask] - 1e-8 
    a, b, _, _ = beta.fit(data_r, floc=0, fscale=1)
    print(f"Found alpha:{a}, beta:{b}")
    return qnp.random.beta(a=a, b=b, size=sz).reshape(shape)

def init_uniform_norm(data, shape):
    dr = data.reshape(data.size)
    sz = np.prod(shape)
    dmin, dmax = np.min(data), np.max(data)
    for i in range(len(data)):
        dr[i] = (dr[i] - dmin)/(dmax - dmin)
    zmask = dr <= 0
    omask = dr >= 1
    dr[zmask] = dr[zmask] + 1e-8
    dr[omask] = dr[omask] - 1e-8
    l, h = ss.uniform.fit(dr)
    print(f"Determined range: [{l},{h}]")
    return qnp.random.uniform(l, h, size=sz).reshape(shape)

PARAM_INIT_DICT = {
    'uniform': init_uniform_norm,
    'beta': init_beta,
    'truncnorm': init_trunc_normal,
    'beta_bayes': init_beta_ebayes
}

#### Utils #### 

def angle_embed(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

def get_features_iris(data, embed_fn, pad=False):
    if pad:
        padding = 0.3 * np.ones((len(data), 1))
        dpad = np.c_[np.c_[data, padding], np.zeros((len(data), 1))]
        norm = np.sqrt(np.sum(dpad ** 2, -1))
        dnorm = (dpad.T/norm).T
    else:
        normf = np.sqrt(np.sum(data**2, -1))
        dnorm = (data.T / normf).T
    
    feats = qnp.array([embed_fn(x) for x in dnorm], requires_grad=False)
    return feats
        
        

def mse_loss(label, pred):
    loss = 0
    for l, p in zip(label, pred):
        loss = loss + (l - p) ** 2

    loss = loss / len(label)
    return loss

def get_ab(data):
    dr = data.reshape(data.size)
    a, b,_, _ = beta.fit(dr)
    return a, b

def get_preds(ckt, params, data):
    preds = [qnp.sign(ckt(params, d)) for d in data]
    return preds

def accuracy(label, preds):
    corr = 0
    for l,p in zip(label, preds):
        if abs(l-p) < 1e-5:
            corr += 1
    acc = corr / len(label)
    return acc


def get_grad_stats(grad_fn, params, data):
    gvals = qnp.array([grad_fn(params, d) for d in data])
    return gvals

def update_params(gmean, gvar):
    alpha = gmean*gvar / (gmean - (gmean**2 + gvar))
    beta = gvar* ( 1 - gmean) / (gmean - (gmean**2 + gvar))
    print(alpha, beta)
    return alpha, beta

### Experiment ####

# TODO: integrate this with data measuring circuit.
def measure_var_rand_ckt(rand_ckt:Callable, 
                         param_init:List[str], 
                         gate_seq:List, ntrials:int=200):
    """
    Measure effect of different initialization strategies
    on the gradient of a randomly parametrized circuit. 
    """
    def _run_expt(qubit, init):
        grad_vals = []
        dev = qml.device("default.qubit", wires=qubit)
        qckt = qml.QNode(rand_ckt, device=dev)
        ifn = PARAM_INIT_DICT[init]
        for i in range(ntrials):
            params = ifn(qubits)
            gseq = {i: qnp.random.choice(gate_seq)
                    for i in range(qubits)}
            grad = qml.grad(qckt, argnum=0)
            gval = grad(params, gate_seq=gate_seq, nqubits=qubit)
            grad_vals.append(gval)
        return grad_vals
    
    qubits = [2, 3, 4, 5, 6, 8, 10]
    variances = {}
    figname = "_".join(param_init)
    
    for init in param_init:
        for qubit in qubits:
            grad_vals = _run_expt(qubit, init)
            if init not in variances:
                variances[init] = [np.var(grad_vals)]
            else:
                variances[init].append(np.var(grad_vals))
    
    for k in variances.keys():
        plt.semilogy(qubits, variances[k], '-^', label=f"{k}")
    plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
    plt.xlabel("Qubits")
    plt.legend(loc='best')
    plt.savefig("figures/{figname}.png")


##### Circuit running ####
def _run_ckt(ckt, init_fn, cfg):  
    def cost(params, data, label):
      val = [ckt(params, d) for d in data]
      return mse_loss(label, val)

    qubits = len(ckt.device.wires)
    pinit = init_fn(cfg['X'], (cfg['layers'], qubits, 3))
    params = pinit
    opt = cfg['opt'](cfg['lr'])
    X, y = cfg['X'], cfg['y']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8)
    print(X_tr.shape)
    grad = qml.grad(ckt, argnum=0)
    grad_var = []
    eta = 1.0
    iters = cfg['iters']
    perturb = cfg['perturb']
    gvals = []
    for i in range(iters): 
      params, _, _, = opt.step(cost, params, X_tr, y_tr)
      grad_vals = qnp.array([grad(params, d) for d in X_tr])
      gvals.append(grad_vals)
      gvar = np.var(grad_vals)
      gvals.append(grad_vals)
      if perturb:
        sc = eta/((1+i)**(0.55+gvar))
        params = params + qnp.random.normal(loc=0.0, scale=sc, 
                                            size=(cfg['layers']*qubits*3)).reshape(cfg['layers'],qubits, 3)
      if i != 0 and i % 5 == 0:
        gv = np.var(np.array(gvals)[:, -1])
        grad_var.append(gv)
        gvals = []

      tcost = cost(params, X_tr, y_tr)
      tr_preds = get_preds(ckt, params, X_tr)
      te_preds = get_preds(ckt, params, X_te)
      tr_acc = accuracy(y_tr, tr_preds)
      te_acc = accuracy(y_te, te_preds)

      print(f"{i}| Cost: {tcost:.2f} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")
    return grad_var



def run_qubit_expt(qckt, cfg, istr='uniform'):
    init_fn = PARAM_INIT_DICT[istr]
    qubits = [int(q) for q in cfg['qubits']]
    devs = [qml.device('default.qubit', wires=qub) for qub in qubits]
    ckts = [qml.QNode(qckt, dev) for dev in devs]

    gvars_qubits = {}

    for ckt in ckts:
        _qub = len(ckt.device.wires)
        print(f"Training with {_qub} qubits")
        gvars = _run_ckt(ckt, init_fn, cfg)
        gvars_qubits[_qub] = gvars

    return gvars_qubits

def run_layer_expt(qckt, cfg, istr='uniform'):
    init_fn = PARAM_INIT_DICT[istr]
    layers = [int(l) for l in cfg['layers']]
    # qubit = cfg['qubits']
    cfg_layers = []
    for l in layers:
        _cf = {'X': cfg['X'], 'y': cfg['y'],
        'opt': cfg['opt'], 'lr':cfg['lr'],
        'layers': l, 'iters': cfg['perturb'], 'perturb': cfg['perturb']}
        cfg_layers.append(_cf)

    gvars_layers = {}
    for _cfg in cfg_layers:
        _qub = len(qckt.device.wires)
        print(f"Training with {_qub} qubits")
        gvars = _run_ckt(qckt, init_fn, _cfg)
        gvars_layers[_qub] = gvars

    return gvars_layers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ctype', type=int, default=2,
     help='Type of circuit to use')
    parser.add_argument('-t', '--type', type=str, default='qubits',
    help='Type of experiment')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config dict')

    args = parser.parse_args()

    cfg = load_json(args.cfg)

    if cfg['dataset'] == 'iris':
        X, y = load_iris(cfg['data_path'])
        cfg['X'] = X
        cfg['y'] = y
    elif cfg['dataset'] == 'wine':
        X, y = load_wine(cfg['data_path'])
        pca = PCA(n_components=2)
        Xw = pca.fit_transform(X)
        cfg['X'] = Xw
        cfg['y'] = y

    
    print("Starting training...")
    print("==== Hyperparams ====")
    print("Dataset: ", cfg['dataset'])
    print("Iterations: ", cfg['iters'])
    print("Circuit Type: ", args.ctype)
    print("Initialization: ", cfg['init'])
    print('=='*9)
    
    CKT_MAP = {1: pnl_ckt, 2: pnl_ckt_v2}
    OPT_MAP = {
    'adam': qml.AdamOptimizer,
    'nesterov': qml.NesterovMomentumOptimizer,
    'sgd': qml.GradientDescentOptimizer
    }
    qckt = CKT_MAP[args.ctype]

    assert 'X' in cfg, 'No data found in config!'
    assert 'y' in cfg, 'No labels found in config!'

    checkmkdirs('ckpts')
    cfg['opt'] = OPT_MAP[cfg['optimizer']]

    if args.type == 'qubit':
        print(f"Running optimization over qubits...")   
        gvars_qubits = run_qubit_expt(qckt, cfg, istr=cfg['init'])
        ofname = f"{cfg['dataset']}_qubits_{cfg['init']}.pkl"
        save_ckpt(os.path.join('ckpts', ofname),gvars_qubits)

    elif args.type == 'layer':
        print(f"Running optimization over layers...")
        qubits = int(cfg['qubits'])
        print(f"Running with {qubits} qubits")
        dev = qml.device(wires=qubits)
        ckt = qml.QNode(qckt, device=dev)
        gvars_layers = run_layer_expt(ckt, cfg, istr=cfg['init'])
        ckpt_name = f"{cfg['dataset']}_layers_{cfg['init']}.pkl"
        save_ckpt(os.path.join('ckpts', ckpt_name), gvars_layers)
   
if __name__ == '__main__':
    main()