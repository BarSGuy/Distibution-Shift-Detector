import torch
import torchvision
import pytorch_lightning as pl
import yaml
from easydict import EasyDict as edict
from models import resnets
import math

# ============================== Confidence-rate functions ============================== #
def get_softmax_responses_inline(logits_list):
    """
    Args:
        logits_list (list): List of PyTorch tensors, each containing a vector of logits.

    Returns:
        List of PyTorch tensors, each containing a vector of softmax responses corresponding to the input logits.
    """
    SR_list = []
    for logits in logits_list:
        softmax = torch.softmax(logits, dim=-1)
        SR = torch.max(softmax).item()
        SR_list.append(SR)
    return SR_list

def get_entropy_responses_inline(logits_list, num_classes):
    """
    Calculates the entropy of each softmax vector in the list of logits vectors

    Args:
        logits_list: A list of PyTorch tensors containing logits
        num_classes: The number of classes for the classification problem

    Returns:
        A list of PyTorch tensors containing the entropy of each softmax vector
    """
    entropy_list = []
    for logits in logits_list:
        softmax = torch.nn.functional.softmax(logits, dim=0)
        entropy = -1 * torch.sum(softmax * torch.log2(softmax + 1e-20))
        entropy /= math.log2(num_classes)
        one_minus_entropy = 1 - entropy.item()
        entropy_list.append(one_minus_entropy)
    return entropy_list


# ============================== Data Modules ============================== #
class CIFAR10DataModule(pl.LightningDataModule):
    """A PyTorch Lightning data module for the CIFAR10 dataset."""

    def __init__(self, test_size=10000):
        super().__init__()
        self.test_size = test_size

    def prepare_data(self):
        """Downloads the CIFAR10 dataset if it is not already downloaded."""
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        """Sets up the test dataset."""
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])
        full_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
        self.test_set, _ = torch.utils.data.random_split(full_test_set,
                                                         [self.test_size, len(full_test_set) - self.test_size])

    def test_dataloader(self):
        """Returns a data loader for the test set."""
        return torch.utils.data.DataLoader(self.test_set, batch_size=64, num_workers=4)


class CIFAR100DataModule(pl.LightningDataModule):
    """A PyTorch Lightning data module for the CIFAR100 dataset."""

    def __init__(self, test_size=10000):
        super().__init__()
        self.test_size = test_size

    def prepare_data(self):
        """Downloads the CIFAR100 dataset if it is not already downloaded."""
        torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

    def setup(self, stage=None):
        """Sets up the test dataset."""
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])
        full_test_set = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform)
        self.test_set, _ = torch.utils.data.random_split(full_test_set,
                                                         [self.test_size, len(full_test_set) - self.test_size])

    def test_dataloader(self):
        """Returns a data loader for the test set."""
        return torch.utils.data.DataLoader(self.test_set, batch_size=64, num_workers=4)


# ============================== Models ============================== #
class Lightning_model(pl.LightningModule):
    """A PyTorch Lightning module for a pre-trained ResNet18 model on CIFAR10."""

    def __init__(self, model):
        """Creates a pre-trained ResNet18 model and sets it to evaluation mode."""
        super().__init__()
        if model == 'Resnet18':
            self.model = resnets.resnet18(pretrained=True)
        else:
            raise ValueError('Invalid OUT_OF_DISTRIBUTION_DATA value.')

    def forward(self, x):
        """Performs a forward pass on the model."""
        return self.model(x)

    def test_step(self, batch, batch_idx):
        """Computes the loss and accuracy of the model on the test set."""
        x, y = batch
        y_hat = self(x)
        # loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        # self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {
            # 'test_loss': loss,
            'logits': y_hat.detach()}

    def test_epoch_end(self, outputs):
        """Stacks all the logits in a torch array."""
        stacked_logits = torch.cat([x['logits'] for x in outputs], dim=0)
        self.logits = stacked_logits


# ============================== Inference ============================== #

def inference(data_module, model, num_gpus=1):
    """
    Runs inference on a PyTorch Lightning model using a test dataset.

    Args:
        data_module: A PyTorch Lightning DataModule that provides the test dataset.
        model: A PyTorch Lightning module that performs the inference.
        num_gpus: The number of GPUs to use for inference (default: 0).

    Returns:
        A PyTorch tensor of shape (num_test_samples, num_classes) containing the model's logits.
    """
    # Set up the trainer for inference
    trainer = pl.Trainer(gpus=num_gpus)

    # Run inference on the test dataset
    trainer.test(model, datamodule=data_module)

    return model.logits


# ============================== Detection experiment ============================== #

def detect_experiment(cfg):
    # setting the in-distribution data module
    in_distribution_data_module = CIFAR10DataModule(cfg.IN_DISTRIBUTION_SIZE)
    # setting the out-of-distribution data module
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar100':
        out_of_distribution_data_module = CIFAR100DataModule(test_size=cfg.OUT_OF_DISTRIBUTION_SIZE)
    elif cfg.OUT_OF_DISTRIBUTION == 'Cifar10':
        out_of_distribution_data_module = CIFAR10DataModule(cfg.OUT_OF_DISTRIBUTION_SIZE)
    else:
        raise ValueError('Invalid OUT_OF_DISTRIBUTION_DATA value.')
    # getting all logits
    light_model = Lightning_model(model=cfg.MODEL.NAME)
    in_distribution_logits = inference(data_module=in_distribution_data_module, model=light_model,
                                       num_gpus=cfg.NUM_GPUS)
    out_of_distribution_logits = inference(data_module=out_of_distribution_data_module, model=light_model,
                                           num_gpus=cfg.NUM_GPUS)

    # ---------------------------- In dist ---------------------------- #
    SR_list_in_dist = get_softmax_responses_inline(in_distribution_logits)

    # separating to test and validation sets for in distribution
    SR_list_in_dist_to_fit = SR_list_in_dist[:len(SR_list_in_dist) // 2]
    SR_list_in_dist_to_test = SR_list_in_dist[len(SR_list_in_dist) // 2:]

    # ---------------------------- Out dist ---------------------------- #
    SR_list_out_dist = get_softmax_responses_inline(out_of_distribution_logits)

    from Detector import Shift_Detector as SH
    C_num = math.log(cfg.IN_DISTRIBUTION_SIZE)
    delta = 0.0001
    detector = SH(C_num, delta)
    detector.fit_lower_bound(SR_list_in_dist_to_fit)
    detector.fit_upper_bound(SR_list_in_dist_to_fit)

    def get_Cifar10_vs_Cifar100_inline():

        under_confidence_score = detector.detect_lower_bound_deviation(SR_list_out_dist)
        print(f'The under confidence score is {under_confidence_score}')
        detector.visualize_lower_bound("lower bound; Cifar10 (in-dist) vs Cifar100 (out-dist)")

        over_confidence_score = detector.detect_upper_bound_deviation(SR_list_out_dist)
        print(f'The over confidence score is {over_confidence_score}')
        detector.visualize_upper_bound("upper bound; Cifar10 (in-dist) vs Cifar100 (out-dist)")

    get_Cifar10_vs_Cifar100_inline()

    def get_Cifar10_vs_Cifar10_inline():

        under_confidence_score = detector.detect_lower_bound_deviation(SR_list_in_dist_to_test)
        print(f'The under confidence score is {under_confidence_score}')
        detector.visualize_lower_bound("lower bound; Cifar10 (in-dist) vs Cifar10 (out-dist)")

        over_confidence_score = detector.detect_upper_bound_deviation(SR_list_in_dist_to_test)
        print(f'The over confidence score is {over_confidence_score}')
        detector.visualize_upper_bound("upper bound; Cifar10 (in-dist) vs Cifar10 (out-dist)")

    get_Cifar10_vs_Cifar10_inline()




if __name__ == '__main__':
    with open("config.yml", "r") as file:
        cfg = yaml.safe_load(file)
        cfg = edict(cfg)
    detect_experiment(cfg)
    exit()
