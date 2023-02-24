# Coverage Based Detection - a Simple Method for Detecting a Distribution Shift
This repository contains the official implementation of our paper: [DISTRIBUTION SHIFT DETECTION FOR DEEP NEURAL
NETWORKS](https://arxiv.org/pdf/2210.10897).

**TLDR**:
The method aims to detect deviations in the input distribution that could potentially harm the accuracy of the network's predictions.
The proposed method is based on selective prediction principles and involves continuously monitoring the network's operation over a test window and firing off an alarm when a deviation is detected.
The method outperforms the state-of-the-art approaches for this task on the CIFAR-10 and ImageNet datasets while being more efficient in time and space complexities.

![results](figures/demo_in_vs_out.pdf)

## Citation

If you find our paper/code helpful, please cite our paper:

    @article{bar2022distribution,
        title={Distribution Shift Detection for Deep Neural Networks},
        author={Bar-Shalom, Guy and Geifman, Yonatan and El-Yaniv, Ran},
        journal={arXiv preprint arXiv:2210.10897},
        year={2022}
    }


## Results

This code is designed to reproduce our results of the experiments on **ResNets** variants in **ImageNet**.

| Architecture | Params (M) | Inductive (standard) (%) | Transductive (ours) (%) | Improvement (%) |
|--------------|------------|--------------------------|-------------------------|-----------------|
| ResNet18     | 11.69      | 69.76                    | 73.36                   | +3.60           |
| ResNet34     | 21.80      | 73.29                    | 76.70                   | +3.41           |
| ResNet50     | 25.56      | 76.15                    | 79.03                   | +2.88           |
| ResNet101    | 44.55      | 77.37                    | 79.86                   | +2.49           |
| ResNet152    | 60.19      | 78.33                    | 80.64                   | +2.31           |

## Reproduce Results

Make sure you have downloaded the **ImageNet** dataset first.

To clone and install this repository run the following commands:

    git clone https://github.com/omerb01/TransBoost.git
    cd TransBoost
    pip install -r requirements.txt

## To run it on you own id-distribution and out-of-distribution data

    usage: transboost.py [-h] [--dev DEV] --gpus GPUS [--resume RESUME] --data-dir DATA_DIR [--num-workers NUM_WORKERS] [--wandb]
                     [--gpu-monitor] [--data DATA] --model MODEL [--seed SEED] [--max-epochs MAX_EPOCHS]
                     [--batch-size BATCH_SIZE] [--optimizer OPTIMIZER] [--learning-rate LEARNING_RATE] [--cosine]
                     [--weight-decay WEIGHT_DECAY] [--lamda LAMDA] [--test-only]

    optional arguments:
      -h, --help            show this help message and exit
      --dev DEV             debugging mode
      --gpus GPUS           number of gpus
      --resume RESUME       path
      --data-dir DATA_DIR   data dir
      --num-workers NUM_WORKERS
                            number of cpus per gpu
      --wandb               logging in wandb
      --gpu-monitor         monitors gpus. Note: slowing the training process
      --data DATA           dataset name
      --model MODEL         model name
      --seed SEED           seed
      --max-epochs MAX_EPOCHS
                            number of fine-tuning epochs
      --batch-size BATCH_SIZE
                            batchsize for each gpu, for each train/test. i.e.: actual batchsize = 128 x num_gpus x 2
      --optimizer OPTIMIZER
      --learning-rate LEARNING_RATE
      --cosine              apply cosine annealing lr scheduler
      --weight-decay WEIGHT_DECAY
      --lamda LAMDA         TransBoost loss hyperparameter
      --test-only           run testing only

For example, to reproduce our result on **ResNet50** in **ImageNet**, run:

    python transboost.py --gpus 4 --data-dir /path-to-data-folder/ImageNet --model resnet50

## Acknowledgments

![isf](images/isf.png)

This research was partially supported by the Israel Science Foundation, grant No. 710/18.
