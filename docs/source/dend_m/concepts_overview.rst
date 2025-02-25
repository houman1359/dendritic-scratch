.. _concept_overview:  
  
Concepts Overview
=================

The initial dendritic model is designed as follows: external input data (considered as excitatory input) is fed into an excitatory/inhibitory classifier (``EINetClassifier``). This classifier model contains a single excitatory/inhibitory network (``ExitationInhibitionNetwork``), which consists of one or more excitation-inhibition layers (``ExcitationInbitionLayer``). Each excitation-inhibition layer has two dendritic layer (``Dendrinet``). 

.. figure:: figures/pdf/EINetClassifier_dataflow.pdf
    :align: center
    :width: 100%
    :alt: EINetClassifier dataflow
    :name: EINetClassifier_dataflow
    
    : Dataflow diagram of the EINetClassifier model. The blue arrows represent the flow of excitatory data, and the red arrows represent the flow of inhibitory data. Please note that the first inhibitory dendritic layer is not recieving any inhibitory input.

Within each dendritic layer, there is a specialized feedforward multilayer perceptron.

.. figure:: figures/pdf/Dendrinet_1.pdf
    :align: center
    :width: 100%
    :alt: Dendrinet dataflow 1
    :name: Dendrinet_dataflow_1
    
    : Dataflow diagram of the Dendrinet model. The blue arrows represent the flow of excitatory data, and the red arrows represent the flow of inhibitory data. Each dendritic baranch layer (``DendriticBranchLayer``) accepts three types of inputs, excitatory, inhibitory, and data flow from upstream layers through branch factor. In the presented figure, the first dendritic layer branch factor is 2 and the second dendritic layer branch factor is 3. The outermost layer does not have any branch factor.

Inside the ``ExcitationInhibitionLayer``, the Dendrinet is categorized as either excitatory or inhibitory. Consequently, the output of the Dendrinet will be excitatory if it is defined as such, or inhibitory if it is defined as inhibitory. Additionally, note that the input data is fed to all dendritic branch layers. The final layer, known as the somatic layer, will receive direct input (both excitatory and inhibitory) if it is specified when compiling the model (``somatic_synapses = True``). 

The following figure shows the data flow to the second dendritic layer in the Dendrinet model. The inputs are excitatory, inhibitory, and data flow from the upstream layer. The branch factor for the first dendritic layer is 3.

.. figure:: figures/pdf/Dendrinet_2.pdf
    :align: center
    :width: 100%
    :alt: Dendrinet dataflow 2
    :name: Dendrinet_dataflow_2

    : Dataflow diagram of the Dendrinet model for the second layer. The blue arrows represent the flow of excitatory data, and the red arrows represent the flow of inhibitory data. Each dendritic baranch layer (``DendriticBranchLayer``) accepts three types of inputs, excitatory, inhibitory, and data flow from upstream layers through branch factor. 

The data flow aggregation from upper layers to lower layers is done by ``BlockLinear`` class. The ``BlockLinear`` class generates a mask to dropout the input data based on the branch factor. 

The linear transformation is performed by the ``TopKLinear`` class. The TopKLinear class is a linear transformation that is selects the top-k incoming weights and sets the rest to zero. Note that this process is just for selecting the weights to compute the activation. During the backpropagation, the gradients are computed for all weights and all weights will be updated. 






