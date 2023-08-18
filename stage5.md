---
title: Node Classification with Graph Neural Network
layout: default
nav_order: 4
---
# Node Classification with Graph Neural Network
{: .no_toc}
Trained three datasets: Cora, Citeseer, and Pubmed using different Graph Neural Network models and evaluate the model performance on the testing set.

## Table of contents
{: .no_toc .text-delta}

1. TOC
{:toc}

## Dataset
[Cora, Citeseer, and Pubmed dataset](https://drive.google.com/file/d/1DeFFj4iVWDX8El3iCUE4a_mJME485t0R/view?usp=sharing)

Each dataset represents one directed graph. All these datasets are composed of two files: node file, and link file.

### Cora
In cora dataset, the node file has 2,708 lines, each line denoting one node. For the nodes in cora, each line has 1,435 elements, which are organized as follows:
`<node_id> <1433 node_features> <node_class_label>`

The first element denotes the node index. (Each node has one unique index, which will also be used in the link file denoting the nodes). The last element (a string) denotes the node label, and the mid 1433 elements denote the features of the node.

The link file has 5,429 lines, each line denoting one directed link (A B, i.e., it denotes B pointing to A).

#### Dataset Organization
{: .no_toc}
Has 7 different classes `(Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory)`.

### Citeseer
In the citeseer dataset, the node file has 3,312 lines, each line has 3,705 elements denoting one node `(<node id> <3 703 node features> <node label>)`. The link file has 4,715 lines, each line denoting one directed link (A B, i.e., it denotes B pointing to A).

#### Dataset Organization
{: .no_toc}
Has 6 different classes `(AI, Agents, DB, HCI, IR, ML)`.

### Pubmed
In pubmed dataset, the node file has 19 717 lines, each line has 502 elements denoting one node `(<node id> <500 node features> <node label>)`. The link file has 44,324 lines, each line denoting one directed link (A B, i.e., it denotes B pointing to A).

#### Dataset Organization
{: .no_toc}
Has 3 different classes `(0, 1, 2)`.

## Abstract
Implemented a node classification with graph neural network model, passed data through convolution layers.

## Model
The node classification model uses Transductive Learning to study the node classification in each of the 3 datasets. The model uses a combination of Pytorch’ Geometry Graph Convolution Layer and Linear Layer with ReLu activation layer to produce results. I used KFold Cross Validation in our training to prevent overfitting especially for the Citeseer dataset. I didn't use mini batches for our model since we did not run into memory issues like in the previous Text Classification task.

### Cora Model
![Tensorboard plot for the classification model used for Cora dataset](./assets/stage5/Cora%20architecture.png "2 GConv layers and then classification layer")

### Citeseer Model
![Tensorboard plot for the classification model used for Citeseer dataset](./assets/stage5/citeseer%20architecture.png "2 GConv layers then classification layer")

### Pubmed Model
![Tensorboard plot for the classification model used for the Pubmed dataset](./assets/stage5/pubmed%20architecture.png "3 GConv layers then classification layer")

## Experiment
We have three datasets for this stage. The dataset for all contains nodes and links data, with each node representing a document and each link representing the citation. Each dataset contains different number of nodes and links, however, each nodes in a dataset are also classified into different class. Since the class of each nodes are imbalanced, we split the training and testing using the project description and give both our training and testing data with number of entries in a class equal to each other in the same dataset. Since sampling data point for each run takes time, we decided to save the sampled data into pickle file and use them when hypertunning the model. Since we use transductive learning for our graph, we use the entire dataset and masked out the testing label when calculating loss. When testing the model, we masked out all the label we used during training except the unseen testing label.

### Cora & Citeseer Setup
- Cora

Randomly sample a training set with 140 nodes (20 node instances per class), and evaluate the learned model on a randomly sampled testing set with 1050 nodes (150 node instances per class).
- Citeseer

Randomly sample a training set with 120 nodes (20 node instances per class), and evaluate the learned model on a randomly sampled testing set with 1200 nodes (200 node instances per class).

#### Detailed Experiment Setups for Cora & Citeseer
{: .no_toc}
```
--network status--
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Method_Classification                    [2708, 7]                 --
├─Sequential_c81df2: 1-1                 [2708, 7]                 --
│    └─GConv: 2-1                      [2708, 50]                50
│    │    └─Linear: 3-1                  [2708, 50]                71,650
│    │    └─SumAggregation: 3-2          [2708, 50]                --
│    └─ReLU: 2-2                         [2708, 50]                --
│    └─GConv: 2-3                      [2708, 7]                 7
│    │    └─Linear: 3-3                  [2708, 7]                 350
│    │    └─SumAggregation: 3-4          [2708, 7]                 --
│    └─Sigmoid: 2-4                      [2708, 7]                 --
==========================================================================================
Total params: 72,057
Trainable params: 72,057
Non-trainable params: 0
Total mult-adds (M): 194.98
==========================================================================================
Input size (MB): 15.61
Forward/backward pass size (MB): 1.23
Params size (MB): 0.29
Estimated Total Size (MB): 17.13
==========================================================================================
```

### Pubmed Setup
Randomly sample a training set with 60 nodes (20 node instances per class), and evaluate the learned model on a randomly sampled testing set with 600 nodes (200 node instances per class).

#### Detailed Experiment Setups
{: .no_toc}
```
--network status--
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Method_Classification                    [19717, 3]                --
├─Sequential_87497e: 1-1                 [19717, 3]                --
│    └─GConv: 2-1                      [19717, 100]              100
│    │    └─Linear: 3-1                  [19717, 100]              50,000
│    │    └─SumAggregation: 3-2          [19717, 100]              --
│    └─ReLU: 2-2                         [19717, 100]              --
│    └─GConv: 2-3                      [19717, 50]               50
│    │    └─Linear: 3-3                  [19717, 50]               5,000
│    │    └─SumAggregation: 3-4          [19717, 50]               --
│    └─ReLU: 2-4                         [19717, 50]               --
│    └─GConv: 2-5                      [19717, 16]               16
│    │    └─Linear: 3-5                  [19717, 16]               800
│    │    └─SumAggregation: 3-6          [19717, 16]               --
│    └─ReLU: 2-6                         [19717, 16]               --
│    └─Linear: 2-7                       [19717, 3]                51
│    └─Sigmoid: 2-8                      [19717, 3]                --
==========================================================================================
Total params: 56,017
Trainable params: 56,017
Non-trainable params: 0
Total mult-adds (G): 1.10
==========================================================================================
Input size (MB): 40.14
Forward/backward pass size (MB): 26.66
Params size (MB): 0.22
Estimated Total Size (MB): 67.02
==========================================================================================
```

With learning a new Deep Learning model, we proceed to use most of the concrete experimental setups that we have been using for the last 4 stages. This time we meet many challenges in achieving high testing accuracy with the 3 datasets using our Graph Convolution Network architecture. Our training doesn’t take too long for each datasets, however the imbalance in each of the class in all 3 datasets cause us a few trouble in getting good accuracy. To resolve this, we use the numpy.random.choice library function to build our training data to be balance across different classes .Aside from the imbalanced dataset, we use a learning rate of 1e-4 and momentum of 0.9. We use simple GConv layer follow with a Linear Layer at the end.

## Evaluation Metrics
We use scikit-learn library’s `classification_report()` function to report F1, Accuracy, precision and recall when using our test dataset.

### Cora Training Result
![Tensorboard graph showing training loss on Cora dataset](./assets/stage5/train%20loss%20cora.png "Training loss in 2000 epochs")
Cora Model Performace
```
--start training...
Epoch: 0 Accuracy: 0.10782865583456426 Loss: 1.9447355270385742
Epoch: 100 Accuracy: 0.5258493353028065 Loss: 1.8176501989364624
Epoch: 200 Accuracy: 0.6824224519940916 Loss: 1.679567575454712
Epoch: 300 Accuracy: 0.7211964549483013 Loss: 1.5665239095687866
Epoch: 400 Accuracy: 0.7370753323485968 Loss: 1.4733980894088745
Epoch: 500 Accuracy: 0.741506646971935 Loss: 1.4004459381103516
Epoch: 600 Accuracy: 0.7426144756277696 Loss: 1.3460379838943481
Epoch: 700 Accuracy: 0.741506646971935 Loss: 1.3061797618865967
Epoch: 800 Accuracy: 0.740029542097489 Loss: 1.2769527435302734
Epoch: 900 Accuracy: 0.742245199409158 Loss: 1.2552289962768555
Epoch: 1000 Accuracy: 0.7403988183161004 Loss: 1.2388395071029663
Epoch: 1100 Accuracy: 0.740029542097489 Loss: 1.2262749671936035
Epoch: 1200 Accuracy: 0.7392909896602659 Loss: 1.21648108959198
Epoch: 1300 Accuracy: 0.7374446085672083 Loss: 1.2087384462356567
Epoch: 1400 Accuracy: 0.7385524372230429 Loss: 1.202528953552246
Epoch: 1500 Accuracy: 0.7385524372230429 Loss: 1.1974860429763794
Epoch: 1600 Accuracy: 0.7378138847858198 Loss: 1.1933459043502808
Epoch: 1700 Accuracy: 0.7374446085672083 Loss: 1.1899107694625854
Epoch: 1800 Accuracy: 0.7370753323485968 Loss: 1.187034010887146
Epoch: 1900 Accuracy: 0.7359675036927622 Loss: 1.1846070289611816
--start testing...
run performace metrics: 
              precision    recall  f1-score   support

           0       0.93      0.70      0.80       150
           1       0.85      0.79      0.82       150
           2       0.60      0.84      0.70       150
           3       0.81      0.89      0.85       150
           4       0.73      0.63      0.67       150
           5       0.77      0.76      0.76       150
           6       0.67      0.64      0.66       150

    accuracy                           0.75      1050
   macro avg       0.76      0.75      0.75      1050
weighted avg       0.76      0.75      0.75      1050

saving models...
Accuracy is: 75.04761904761905%
```

### Citeseer Training Result
![Tensorboard plot showing training loss on Citeseer dataset](./assets/stage5/citeseer%20loss.png "Training loss on 2000 epochs")
Citeseer Model Performance
```
--start training...
Epoch: 0 Accuracy: 0.2086352657004831 Loss: 1.7926758527755737
Epoch: 100 Accuracy: 0.4821859903381642 Loss: 1.551129937171936
Epoch: 200 Accuracy: 0.5739734299516909 Loss: 1.3443034887313843
Epoch: 300 Accuracy: 0.5960144927536232 Loss: 1.2258328199386597
Epoch: 400 Accuracy: 0.5993357487922706 Loss: 1.1609569787979126
Epoch: 500 Accuracy: 0.6023550724637681 Loss: 1.1238117218017578
Epoch: 600 Accuracy: 0.6020531400966184 Loss: 1.1012934446334839
Epoch: 700 Accuracy: 0.5987318840579711 Loss: 1.0868868827819824
Epoch: 800 Accuracy: 0.5978260869565217 Loss: 1.077197790145874
Epoch: 900 Accuracy: 0.595108695652174 Loss: 1.070405125617981
Epoch: 1000 Accuracy: 0.595108695652174 Loss: 1.065474271774292
Epoch: 1100 Accuracy: 0.5939009661835749 Loss: 1.0618014335632324
Epoch: 1200 Accuracy: 0.5929951690821256 Loss: 1.0590002536773682
Epoch: 1300 Accuracy: 0.592391304347826 Loss: 1.0568196773529053
Epoch: 1400 Accuracy: 0.5932971014492754 Loss: 1.0550918579101562
Epoch: 1500 Accuracy: 0.592391304347826 Loss: 1.053702473640442
Epoch: 1600 Accuracy: 0.5926932367149759 Loss: 1.0525693893432617
Epoch: 1700 Accuracy: 0.5929951690821256 Loss: 1.051634669303894
Epoch: 1800 Accuracy: 0.5929951690821256 Loss: 1.050855278968811
Epoch: 1900 Accuracy: 0.5926932367149759 Loss: 1.0501995086669922
Epoch: 2000 Accuracy: 0.592391304347826 Loss: 1.0496429204940796
Epoch: 2100 Accuracy: 0.5926932367149759 Loss: 1.0491669178009033
Epoch: 2200 Accuracy: 0.5926932367149759 Loss: 1.0487573146820068
Epoch: 2300 Accuracy: 0.5926932367149759 Loss: 1.0484023094177246
Epoch: 2400 Accuracy: 0.5920893719806763 Loss: 1.0480931997299194
Epoch: 2500 Accuracy: 0.591183574879227 Loss: 1.04782235622406
Epoch: 2600 Accuracy: 0.591183574879227 Loss: 1.0475841760635376
Epoch: 2700 Accuracy: 0.591183574879227 Loss: 1.047373652458191
Epoch: 2800 Accuracy: 0.5905797101449275 Loss: 1.0471867322921753
Epoch: 2900 Accuracy: 0.5902777777777778 Loss: 1.047020435333252
--start testing...
run performace metrics: 
              precision    recall  f1-score   support

           0       0.66      0.58      0.62       200
           1       0.49      0.77      0.60       200
           2       0.68      0.65      0.67       200
           3       0.35      0.26      0.30       200
           4       0.59      0.68      0.63       200
           5       0.55      0.40      0.46       200

    accuracy                           0.56      1200
   macro avg       0.55      0.55      0.55      1200
weighted avg       0.55      0.56      0.55      1200

saving models...
Accuracy is: 55.50000000000001%
```

### Pubmed Training Result
![Tensorboard plot shows training loss on Citeseer dataset](./assets/stage5/pubmed%20loss.png "Training loss on 3000 epochs")
Pubmed Model Performance
```
--start training...
Epoch: 0 Accuracy: 0.20809453770857636 Loss: 1.0993536710739136
Epoch: 100 Accuracy: 0.20880458487599535 Loss: 1.08661949634552
Epoch: 200 Accuracy: 0.46766749505502864 Loss: 1.0361478328704834
Epoch: 300 Accuracy: 0.6188061064056398 Loss: 0.9371546506881714
Epoch: 400 Accuracy: 0.6496424405335497 Loss: 0.8396590352058411
Epoch: 500 Accuracy: 0.6780443272303088 Loss: 0.7546167373657227
Epoch: 600 Accuracy: 0.693310341329817 Loss: 0.6807986497879028
Epoch: 700 Accuracy: 0.7021859309225541 Loss: 0.63088059425354
Epoch: 800 Accuracy: 0.701780189684029 Loss: 0.6008398532867432
Epoch: 900 Accuracy: 0.6992443069432469 Loss: 0.5834053158760071
Epoch: 1000 Accuracy: 0.6957447887609677 Loss: 0.5730117559432983
Epoch: 1100 Accuracy: 0.6934624942942639 Loss: 0.5665971636772156
Epoch: 1200 Accuracy: 0.6916873763757164 Loss: 0.5624879598617554
Epoch: 1300 Accuracy: 0.6907237409342192 Loss: 0.5597530007362366
Epoch: 1400 Accuracy: 0.6898615408023533 Loss: 0.5578685402870178
Epoch: 1500 Accuracy: 0.6897093878379064 Loss: 0.5565271377563477
Epoch: 1600 Accuracy: 0.6897093878379064 Loss: 0.5555448532104492
Epoch: 1700 Accuracy: 0.6888979053608562 Loss: 0.5548080205917358
Epoch: 1800 Accuracy: 0.6884921641223309 Loss: 0.5542422533035278
Epoch: 1900 Accuracy: 0.6881371405386215 Loss: 0.5537994503974915
Epoch: 2000 Accuracy: 0.6882385758482528 Loss: 0.5534470081329346
Epoch: 2100 Accuracy: 0.6879342699193589 Loss: 0.5531625747680664
Epoch: 2200 Accuracy: 0.6879342699193589 Loss: 0.5529307126998901
Epoch: 2300 Accuracy: 0.6879342699193589 Loss: 0.5527392029762268
Epoch: 2400 Accuracy: 0.6880357052289902 Loss: 0.5525792241096497
Epoch: 2500 Accuracy: 0.6878835522645433 Loss: 0.5524444580078125
Epoch: 2600 Accuracy: 0.6875792463356495 Loss: 0.5523301362991333
Epoch: 2700 Accuracy: 0.6875285286808338 Loss: 0.5522323250770569
Epoch: 2800 Accuracy: 0.6875285286808338 Loss: 0.552148163318634
Epoch: 2900 Accuracy: 0.6875285286808338 Loss: 0.5520753264427185
--start testing...
run performace metrics:
              precision    recall  f1-score   support

           0       0.75      0.48      0.59       200
           1       0.73      0.85      0.79       200
           2       0.66      0.79      0.72       200

    accuracy                           0.71       600
   macro avg       0.71      0.71      0.70       600
weighted avg       0.71      0.71      0.70       600

saving models...
Accurarcy is: 70.83333333333334%
```
## Ablation Studies
Since we receive a low accuracy on Citeseer. We tried changing the learning rate and with different architectures, however it gives only little change in both training and testing accuracy. We needs more time to look into what is going on in the citeseer dataset.

## Source Code
GitHub repository page: [https://github.com/CyberExplosion/Deep-Learning-Projects/tree/P5](https://github.com/CyberExplosion/Deep-Learning-Projects/tree/P5)