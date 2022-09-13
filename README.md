# Dual-Channel RNN
================================

A Recurrent Neural Network architecture designed to study the effect of mixed synaptic and diffusive signaling in a population of neurons. Released in conjuction with the publication "Adaptive robustness through incoherent signaling mechanisms in a regenerating brain".

### Model
This folder contains the minimal elements to implement and use this architecture within your work.
The file [_dualChannelRNN.py_](/Model/dualChannelRNN.py) contains the implementation of the custom Keras layer rnnCell_dualNet_. This is designed to be used as a recurrent neural network cell within keras (documentation). Schematic of the cells computation at each timestep shown in the lower half of the figure below.

The file [_model.py_](/Model/model.py) contains a constructor for the complete network model described in the paper and the top half of the diagram below. It also includes class definitions for addition custom layers used within the network.

![Schematic](/schematic.png)

### Publication
This folder contains the tools described above as well as scripts for training and evaluating models.

Key files:\
[_fit_script_stage123_](/Publication/fit_script_stage123.py): provides the complete model initialization and training protocol used within the paper\
[_evaluate_ablation_](/Publication/evaluate_ablation.py): Used to model ablation studies and evaluate output


### Requirements
Model is implemented in Keras for a TensorFlow 1.14 backend.


## Citation


## Authors

* **Samuel Bray**
* **Livia Wyss**
* **Bo Wang**

## License

This project is licensed under GNUV3
