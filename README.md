## BEINIT - Avoiding Barren Plateaus Through Better Initialization

This code repository contains the code, configs and data used to perform experiments in the accompanying paper. The code is self contained
and can be easily extended. 


### Usage

The code can be run using the following command:

```bash
python train.py -c [circuit_type] -t [expt_type] --cfg path/to/config.json
```

Here:

1. `circuit_type` refers to two different types of quantum circuits. They differ in the way data is encoded into the quantum state. We used the circuit type `2` for all our experiments.

2. `expt_type` refers to the experiment type to be performed. Choices are `qubit` and `layer`. If the former is selected then the number of layers must be kept constant in the config file and for the latter case the number of qubits must be kept constant.



### Things to note:

- We use a parameter `eta` that is set to either 0.3 or 0.01. You may need to adjust this parameter for your own dataset.

- In our experiments we have found the default value of `0.55` to be a good choice for our set of datasets. If you're experimenting with your own dataset then a strategy may be to perform a grid search with constant qubits and layers.
