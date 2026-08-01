# This structure is from PyTorch:
1. dataset -> MNIST
2. dataload
3. Model = [...]
4. Optimizer = SGD, ADAM
5. Criterium = ...

# inspiration:
for x,l from dataloader:

y = model(x)
l = criterium (y,t)
l.backward
optimizer.step()


torch.vtills.data.dataset -> California Dataset

_len_
_get_item(index): x,t



Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./data\MNIST\raw\train-images-idx3-ubyte.gz
100.0%
Extracting ./data\MNIST\raw\train-images-idx3-ubyte.gz to ./data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./data\MNIST\raw\train-labels-idx1-ubyte.gz
100.0%
Extracting ./data\MNIST\raw\train-labels-idx1-ubyte.gz to ./data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./data\MNIST\raw\t10k-images-idx3-ubyte.gz
100.0%
Extracting ./data\MNIST\raw\t10k-images-idx3-ubyte.gz to ./data\MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./data\MNIST\raw\t10k-labels-idx1-ubyte.gz
100.0%
Extracting ./data\MNIST\raw\t10k-labels-idx1-ubyte.gz to ./data\MNIST\raw

Training using SGD
Epoch 1/10, Loss: 0.7216104205483312, Accuracy: 0.8315666666666667
Test Accuracy: 0.9088
Epoch 2/10, Loss: 0.29511667981044826, Accuracy: 0.9154333333333333
Test Accuracy: 0.9305
Epoch 3/10, Loss: 0.2388503029386499, Accuracy: 0.9313
Test Accuracy: 0.9388
Epoch 4/10, Loss: 0.20196787727825932, Accuracy: 0.94115
Test Accuracy: 0.9459
Epoch 5/10, Loss: 0.17370034217326116, Accuracy: 0.9506166666666667
Test Accuracy: 0.951
Epoch 6/10, Loss: 0.15232944041331695, Accuracy: 0.9560833333333333
Test Accuracy: 0.9559
Epoch 7/10, Loss: 0.134697213919877, Accuracy: 0.9611666666666666
Test Accuracy: 0.961
Epoch 8/10, Loss: 0.12053467908989328, Accuracy: 0.9659833333333333
Test Accuracy: 0.9619
Epoch 9/10, Loss: 0.10855695756890976, Accuracy: 0.9693
Test Accuracy: 0.9659
Epoch 10/10, Loss: 0.09827216080665938, Accuracy: 0.9726333333333333
Test Accuracy: 0.9686
Training using SGD with momentum
Epoch 1/10, Loss: 0.2925365518417154, Accuracy: 0.9158166666666666
Test Accuracy: 0.96
Epoch 2/10, Loss: 0.09828376759395702, Accuracy: 0.9705333333333334
Test Accuracy: 0.9736
Epoch 3/10, Loss: 0.062141891855841826, Accuracy: 0.98065
Test Accuracy: 0.9779
Epoch 4/10, Loss: 0.04230962705581006, Accuracy: 0.9872666666666666
Test Accuracy: 0.9776
Epoch 5/10, Loss: 0.029715883202847345, Accuracy: 0.9911
Test Accuracy: 0.9807
Epoch 6/10, Loss: 0.02192876435577252, Accuracy: 0.9933333333333333
Test Accuracy: 0.9805
Epoch 7/10, Loss: 0.014453899819338853, Accuracy: 0.996
Test Accuracy: 0.9809
Epoch 8/10, Loss: 0.010020812032221785, Accuracy: 0.9974833333333334
Test Accuracy: 0.98
Epoch 9/10, Loss: 0.006316865862341973, Accuracy: 0.9988666666666667
Test Accuracy: 0.9816
Epoch 10/10, Loss: 0.003853901661567119, Accuracy: 0.9994166666666666
Test Accuracy: 0.9827
Training using Adam
Epoch 1/10, Loss: 0.20735757376875547, Accuracy: 0.9368666666666666
Test Accuracy: 0.9642
Epoch 2/10, Loss: 0.08806696924259827, Accuracy: 0.9728333333333333
Test Accuracy: 0.9681
Epoch 3/10, Loss: 0.06271882952679433, Accuracy: 0.9800666666666666
Test Accuracy: 0.976
Epoch 4/10, Loss: 0.047222758918977786, Accuracy: 0.9848666666666667
Test Accuracy: 0.9759
Epoch 5/10, Loss: 0.03867183843542432, Accuracy: 0.9872
Test Accuracy: 0.9707
Epoch 6/10, Loss: 0.032793914477922584, Accuracy: 0.9895333333333334
Test Accuracy: 0.9759
Epoch 7/10, Loss: 0.027376655828763932, Accuracy: 0.9911166666666666
Test Accuracy: 0.9747
Epoch 8/10, Loss: 0.027826782074637277, Accuracy: 0.9910833333333333
Test Accuracy: 0.9816
Epoch 9/10, Loss: 0.019381230448796093, Accuracy: 0.9936666666666667
Test Accuracy: 0.9782
Epoch 10/10, Loss: 0.01930406813199006, Accuracy: 0.9937
Test Accuracy: 0.9795

Process finished with exit code 0
