# A Unified Hard-Constraint Framework for<br>Solving Geometrically Complex PDEs

This repository is the official implementation of *A Unified Hard-Constraint Framework for Solving Geometrically Complex PDEs*. 

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## Directory

The following is a description of the contents of the directory

```bash
code/
│  .gitignore
│  README.md				# description of this repository
│  requirements.txt
│
├─data						# data used in this paper
│      case1_pack.txt		# ground truth for "Simulation of a 2D battery pack (Heat Equation)"
│      case2_airfoil.txt	# ground truth for "Simulation of an Airfoil (Navier-Stokes Equations)"
│      w1015.dat			# achor points of the airfoil
│
├─model/					# saved model weights (empty)
├─outs/						# outputs (empty)
└─src
    │  case1.py				# scripts for each experiment 
    │  ...
    │
    ├─configs				# hyper-parameters for each experiment 
    │  │
    │  ├─case1
    │  │      ...
    │  │
    │  ├─case2
    │  │      ...
    │  │
    │  └─case3
    │          ...
    │
    ├─FBPINN/				# implementations of each model 
    ├─HC/
    ├─PFNN/
    ├─xPINN/
    │
    └─utils/				# some utils
```

## Training & Evaluation

To train and evaluate the models in the paper, run this command:

```bash
DDEBACKEND=pytorch python -m src.caseX 
```

where X = 1, 2, 3, 4, corresponding to *Simulation of a 2D battery pack (Heat Equation)*, *Simulation of an Airfoil (Navier-Stokes Equations)*, *High-dimensional Heat Equation*, and *Ablation Study: Extra fields*.

If you want to test different models (i.e., the proposed model and baselines), please modify the global variables in `src/caseX.py`.

To run the experiment of *Ablation Study: Hyper-parameters of Hardness*, you can change the value of $\beta_s$ in `src/HC/l_functions.py` (line 31, default: $\beta_s=5$) and $\beta_t$ in `src/configs/case1/hc.py` (line 127, default: $\beta_t=10$, case 1) or `src/configs/case3/hc.py` (line 56, default: $\beta_t=10$, case 3).

## Possible Problems & Solutions

1. Scalar Type Error

   ```bash
   ...
     File "ENV_PATH/lib/python3.9/site-packages/deepxde/model.py", line 228, in outputs_losses
       outputs_ = self.net(self.net.inputs)
   ...
     File "ENV_PATH/lib/python3.9/site-packages/torch/nn/functional.py", line 1848, in linear
       return torch._C._nn.linear(input, weight, bias)
   RuntimeError: expected scalar type Float but found Double
   ```

   Please modify `ENV_PATH/lib/python3.9/site-packages/deepxde/model.py (line 228)`:

   ```python
   self.net.train(mode=training)
   self.net.inputs = torch.as_tensor(inputs)
   self.net.inputs.requires_grad_()
   outputs_ = self.net(self.net.inputs)
   ```

   to:

   ```python
   self.net.train(mode=training)
   self.net.inputs = torch.as_tensor(inputs)
   self.net.inputs.requires_grad_()
   outputs_ = self.net(self.net.inputs.float()) # add this
   ```


## Contributing

Authors of  *A Unified Hard-Constraint Framework for Solving Geometrically Complex PDEs*

