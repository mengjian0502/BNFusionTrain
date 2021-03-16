# BNFusionTrain
Low precision DNN model training with CIFAR-10 dataset. 

## Outline

- Low precision ResNet model training
  - 4-bit / 8-bit ResNet-20 & 4-bit / 8-bit ResNet-18
- Low-precision MobileNet model training
  - 4-bit Progressive quantization.



## Quantization - Aware training

### Symmetric quantization

The current quantization method are implemented based on the *symmetric* quantization scheme to avoid the zero-point computation. 

|                      |        4-bit        |        8-bit        |
| :------------------: | :-----------------: | :-----------------: |
| Floating point range |      [-a, +a]       |      [-a, +a]       |
|  Quantization range  |       [-7, 7]       |     [-127, 127]     |
|    Scaling factor    | $s = (2^{4-1}-1)/a$ | $s = (2^{8-1}-1)/a$ |

Please note that, the floating point values (weight or activation) range $[-a, a]$ should be clamped before performing the quantization operation. 

### Training low-precision model with CIFAR-10 dataset

Before the training, please make sure the follow requirements are satisfied: 



