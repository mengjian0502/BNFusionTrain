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

### Training low-precision ResNet model with CIFAR-10 dataset

Before the training, please make sure the follow requirements are satisfied: 

```
python 3.7.4
pytorch 1.5.0
```

To train the resnet model, run `resnet_cifar_lp.sh`. Currently, we support ResNet model series including ResNet-20 (0.27Million) and ResNet-18 (11.17Miilion) model. The architecture should be specified by setting the `model` variable inside the `.sh` script. 

The ResNet model series was quantized based on PACT and SAWB quantization algorithms. For detailed implementation, please check [here](https://github.com/mengjian0502/BNFusionTrain/blob/0adace354325366864f961224d9f668a62b77a46/models/modules.py#L425) 

More specifically, you can set the weight and activation precision by changing the `wbit`, `abit` variable inside the `.sh` script.

The 4-bit and 8-bit quantization scheme support both layer-wise and channel-wise quantization scheme, you can specify this by setting `channel_wise` to 0 or 1, respectively.

### Training low-precision MobileNet model with CIFAR-10 dataset

To train the mobilenet model, run `mobilenet_cifar_lp.sh`, the detailed settings and input argument are similar to the ResNet training. 



## Reference

```latex
@inproceedings{choi2019accurate,
  title={Accurate and efficient 2-bit quantized neural networks},
  author={Choi, Jungwook and Venkataramani, Swagath and Srinivasan, Vijayalakshmi and Gopalakrishnan, Kailash and Wang, Zhuo and Chuang, Pierce},
  booktitle={Proceedings of the 2nd SysML Conference},
  volume={2019},
  year={2019}
}
```

```latex
@inproceedings{park2020profit,
  title={Profit: A novel training method for sub-4-bit mobilenet models},
  author={Park, Eunhyeok and Yoo, Sungjoo},
  booktitle={European Conference on Computer Vision},
  pages={430--446},
  year={2020},
  organization={Springer}
}
```

