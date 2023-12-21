"""Setup constants, ymmv."""

PIN_MEMORY = True
NON_BLOCKING = False
BENCHMARK = True
MULTITHREAD_DATAPROCESSING = 4


cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
celeba_mean = [0.5061, 0.4254, 0.3828]
celeba_std = [0.2658, 0.2452, 0.2412]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
fashion_mnist_mean = (0.2859,)
fashion_mnist_std = (0.3530,)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

d_m = {'MNIST': mnist_mean, 'Fashion-MNIST': fashion_mnist_mean, 'CIFAR10': cifar10_mean, 'CelebA': celeba_mean, 'CIFAR100': cifar100_mean, 'ImageNet': imagenet_mean}
d_std = {'MNIST': mnist_std, 'Fashion-MNIST': fashion_mnist_std, 'CIFAR10': cifar10_std, 'CelebA': celeba_std, 'CIFAR100': cifar100_std, 'ImageNet': imagenet_std}

d_m_guess = {'MNIST': (0.5,), 'Fashion-MNIST': (0.5,), 'CIFAR10': imagenet_mean, 'CelebA': imagenet_mean, 'CIFAR100': cifar100_mean, 'ImageNet': imagenet_mean}
d_std_guess = {'MNIST': (0.5,), 'Fashion-MNIST': (0.5,), 'CIFAR10': imagenet_std, 'CelebA': imagenet_std, 'CIFAR100': cifar100_std, 'ImageNet': imagenet_std}

d_size = {'MNIST': (1, 28, 28), 'Fashion-MNIST': (1, 28, 28), 'CIFAR10': (3, 32, 32), 'CelebA': (3, 64, 64), 'CIFAR100': (3, 32, 32), 'ImageNet': (3, 224, 224)}