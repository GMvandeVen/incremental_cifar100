# Class-incremental learning with Brain-Inspired Replay in the T&E framework

Besides Brain-Inspired Replay (BI-R), we have currently implemented simple fine-tuning (None), Synaptic Intelligence (SI) and Learning without Forgetting (LwF) as baseline methods.

To run each of these methods, use the following:
- None: ```python train_cifar100_incremental```
- LwF: ```python train_cifar100_incremental --lwf```
- SI: ```python train_cifar100_incremental --si```
- BI-R: ```python train_cifar100_incremental --bir```
- BI-R + SI: ```python train_cifar100_incremental --bir --si```

Log-files for evaluating performance should be produced by running any of the above commands, but we are still working on evaluating these log-files using the l2metrics-environment.