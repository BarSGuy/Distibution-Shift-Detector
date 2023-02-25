# Coverage Based Detection - a Simple Method for Detecting a Distribution Shift
This repository contains the official implementation of our paper: [DISTRIBUTION SHIFT DETECTION FOR DEEP NEURAL
NETWORKS](https://arxiv.org/pdf/2210.10897).

**TLDR**:
The method aims to detect deviations in the input distribution that could potentially harm the accuracy of the network's predictions.
The proposed method is based on selective prediction principles and involves continuously monitoring the network's operation over a test window and firing off an alarm when a deviation is detected.
The method outperforms the state-of-the-art approaches for this task on the CIFAR-10 and ImageNet datasets while being more efficient in time and space complexities.

## Example Usage
### To clone and install this repository run the following commands:

    git clone https://github.com/BarSGuy/Distibution-Shift-Detector.git
    cd Distibution-Shift-Detector
    pip install -r requirements.txt

### Example Run
Copy the Detector file to your project.
Inference your in-distribution data through you classifier, and extract the $\kappa$ for each input. 

    from Detector import Shift_Detector as SH
    number_of_coverages = 10
    delta = 0.0001
    shift_detector = Shift_Detector(C_num=number_of_coverages, delta=delta)
    us_of_in_dist = 

    detector.fit_lower_bound(us_in_dist)
    under_confidence_score = detector.detect_lower_bound_deviation(us_window)
    detector.visualize_lower_bound()


## Demo
When using the CIFAR-10 dataset as in-distribution and the CIFAR-100 dataset as out-of-distribution, the lower bound is violated
(we use the Softmax Response (SR) of a ResNet18 as our confidence-rate function).

![results](figures/demo_in_vs_out.png)

When using the CIFAR-10 dataset both for in distribution data and out-of-distribution data, the bound holds
(we use the Softmax Response (SR) of a ResNet18 as our confidence-rate function).

![results](figures/demo_in_vs_in.png)

## Reproduce The Demo


## To run it on you own in-distribution and out-of-distribution data

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

## Citation

If you find our paper/code helpful, please cite our paper:

    @article{bar2022distribution,
        title={Distribution Shift Detection for Deep Neural Networks},
        author={Bar-Shalom, Guy and Geifman, Yonatan and El-Yaniv, Ran},
        journal={arXiv preprint arXiv:2210.10897},
        year={2022}
    }

