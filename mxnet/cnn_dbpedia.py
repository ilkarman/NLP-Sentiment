"""
Attempt to replicate torch model using MXNET:
https://github.com/zhangxiangxiao/Crepe

"To run this example succesfully you will also need a NVidia GPU with at
least 3GB of memory."

Log from nvidia-smi:
C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi
Sun Aug 21 09:57:48 2016
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 362.84                 Driver Version: 362.84                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           TCC  | 1B5E:00:00.0     Off |                    0 |
| N/A   50C    P0    74W / 149W |    119MiB / 11425MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla K80           TCC  | 4486:00:00.0     Off |                    0 |
| N/A   72C    P0    64W / 149W |    112MiB / 11425MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla K80           TCC  | 75FD:00:00.0     Off |                    0 |
| N/A   62C    P0   127W / 149W |   7356MiB / 11425MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla K80           TCC  | 8933:00:00.0     Off |                    0 |
| N/A   65C    P0    61W / 149W |    112MiB / 11425MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      6904    C   C:\Anaconda2\python.exe                        119MiB |
|    1      6904    C   C:\Anaconda2\python.exe                        112MiB |
|    2      4520    C   C:\Anaconda2\python.exe                       5268MiB |
|    2      6904    C   C:\Anaconda2\python.exe                        112MiB |
|    3      6904    C   C:\Anaconda2\python.exe                        112MiB |
+-----------------------------------------------------------------------------+
"""

import numpy as np
import pandas as pd
import mxnet as mx
import wget
import time
import os.path
from mxnet.io import DataBatch

ctx = mx.gpu(0)
AZ_ACC = "amazonsentimenik"
AZ_CONTAINER = "textclassificationdatasets"
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
FEATURE_LEN = 1014
BATCH_SIZE = 128
NUM_FILTERS = 256
EPOCHS = 1000
SD = 0.05  # std for gaussian distribution
NOUTPUT = 14  # good or bad
DATA_SHAPE = (BATCH_SIZE, 1, FEATURE_LEN, len(ALPHABET))


#classes
#Company
#EducationalInstitution
#Artist
#Athlete
#OfficeHolder
#MeanOfTransportation
#Building
#NaturalPlace
#Village
#Animal
#Plant
#Album
#Film
#WrittenWork




def download_file(url):
    # Create file-name
    local_filename = url.split('/')[-1]
    if os.path.isfile(local_filename):
        pass
        # print("The file %s already exist in the current directory\n" % local_filename)
    else:
        # Download
        print("downloading ...\n")
        wget.download(url)
        print('saved data\n')


def load_data_frame(infile, batch_size=128, shuffle=True):
    print("processing data frame: %s" % infile)
    # Get data from windows blob
    download_file('https://%s.blob.core.windows.net/%s/%s' % (AZ_ACC, AZ_CONTAINER, infile))
    # load data into dataframe
    df = pd.read_csv(infile,
                     header=None,
                     names=['sentiment', 'summary', 'text'])
    # concat summary, review; trim to 1014 char; reverse; lower
    df['rev'] = df.apply(lambda x: "%s %s" % (x['summary'], x['text']), axis=1)
    df.rev = df.rev.str[:FEATURE_LEN].str[::-1].str.lower()
    # store class as nparray
    df.sentiment -= 1
    y_split = np.asarray(df.sentiment, dtype='int')
    # drop columns
    df.drop(['text', 'summary', 'sentiment'], axis=1, inplace=True)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    # Dictionary to create character vectors
    character_hash = pd.DataFrame(np.identity(len(ALPHABET)), columns=ALPHABET)
    print("finished processing data frame: %s" % infile)
    print("data contains %d obs, each epoch will contain %d batches" % (df.shape[0], df.shape[0]//BATCH_SIZE))

    # Yield mini-batch amount of character vectors
    X_split = np.zeros([batch_size, 1, FEATURE_LEN, len(ALPHABET)], dtype='int')
    for ti, tx in enumerate(df.rev):

        chars = list(tx)
        for ci, ch in enumerate(chars):
            if ch in ALPHABET:
                X_split[ti % batch_size][0][ci] = np.array(character_hash[ch])

        if (ti + 1) % batch_size == 0:
            yield mx.nd.array(X_split), mx.nd.array(y_split[ti + 1 - batch_size:ti + 1])
            X_split = np.zeros([batch_size, 1, FEATURE_LEN, len(ALPHABET)], dtype='int')


def example(infile='amazon_review_polarity_test.csv'):
    mbatch = 3
    counter = 0
    for batchX, batchY in load_data_frame(infile, batch_size=mbatch, shuffle=False):
        print("batch: ", batchY.asnumpy().astype('int32'))
        counter += 1
        if counter == 4:
            break

    df = pd.read_csv(infile, header=None)
    train_y = df[[0]].values.ravel() - 1
    print("actual: ", train_y[:mbatch * 4])


def create_crepe():
    """
    Number of features = 70, input feature length = 1014
    2 Dropout modules inserted between 3 fully-connected layers (0.5)
    Number of output units for last layer = num_classes
    For polarity test = 2

    Replicating: https://github.com/zhangxiangxiao/Crepe/blob/master/train/config.lua
    """

    input_x = mx.sym.Variable('data')  # placeholder for input
    input_y = mx.sym.Variable('softmax_label')  # placeholder for output

    # 6 Convolutional layers

    # 1. alphabet x 1014
    conv1 = mx.symbol.Convolution(
        data=input_x, kernel=(7, 69), num_filter=NUM_FILTERS)
    relu1 = mx.symbol.Activation(
        data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 1), stride=(1, 1))

    # 2. 336 x 256
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(7, 1), num_filter=NUM_FILTERS)
    relu2 = mx.symbol.Activation(
        data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(
        data=relu2, pool_type="max", kernel=(3, 1), stride=(1, 1))

    # 3. 110 x 256
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 1), num_filter=NUM_FILTERS)
    relu3 = mx.symbol.Activation(
        data=conv3, act_type="relu")

    # 4. 108 x 256
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 1), num_filter=NUM_FILTERS)
    relu4 = mx.symbol.Activation(
        data=conv4, act_type="relu")

    # 5. 106 x 256
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 1), num_filter=NUM_FILTERS)
    relu5 = mx.symbol.Activation(
        data=conv5, act_type="relu")

    # 6. 104 x 256
    conv6 = mx.symbol.Convolution(
        data=relu5, kernel=(3, 1), num_filter=NUM_FILTERS)
    relu6 = mx.symbol.Activation(
        data=conv6, act_type="relu")
    pool6 = mx.symbol.Pooling(
        data=relu6, pool_type="max", kernel=(3, 1), stride=(1, 1))

    # 34 x 256
    flatten = mx.symbol.Flatten(data=pool6)

    # 3 Fully-connected layers

    # 7.  8704
    fc1 = mx.symbol.FullyConnected(
        data=flatten, num_hidden=1024)
    act_fc1 = mx.symbol.Activation(
        data=fc1, act_type="relu")
    drop1 = mx.sym.Dropout(act_fc1, p=0.5)

    # 8. 1024
    fc2 = mx.symbol.FullyConnected(
        data=drop1, num_hidden=1024)
    act_fc2 = mx.symbol.Activation(
        data=fc2, act_type="relu")
    drop2 = mx.sym.Dropout(act_fc2, p=0.5)

    # 9. 1024
    fc3 = mx.symbol.FullyConnected(
        data=drop2, num_hidden=NOUTPUT)
    crepe = mx.symbol.SoftmaxOutput(
        data=fc3, label=input_y, name="softmax")

    return crepe


# Create mx.mod.Module()
cnn = create_crepe()
mod = mx.mod.Module(cnn, context=ctx)

mod.bind(data_shapes=[('data', DATA_SHAPE)],
         label_shapes=[('softmax_label', (BATCH_SIZE,))])

mod.init_params(mx.init.Normal(sigma=SD))
mod.init_optimizer(optimizer='sgd',
                   optimizer_params={
                       "learning_rate": 0.01,
                       "momentum": 0.9,
                       "wd": 0.00001,
                       "rescale_grad": 1.0/BATCH_SIZE
                   })


def test_net(epoch):
    """" Assess perf on test every epoch ... should really have validatoin data """
    metric = mx.metric.Accuracy()
    print("started testing - epoch %d" % epoch)
    for batchX, batchY in load_data_frame('dbpedia_test.csv', batch_size=BATCH_SIZE):
        batch = DataBatch(data=[batchX], label=[batchY])
        mod.forward(batch)
        mod.update_metric(metric, batch.label)
    metric_m, metric_v = metric.get()
    print("TEST(%s): %.4f" % (metric_m, metric_v))

# Train
print("Alphabet %d characters: " % len(ALPHABET), ALPHABET)
print("started training")
tic = time.time()
# Evaluation metric:
metric = mx.metric.Accuracy()
# Train EPOCHS
for epoch in range(EPOCHS):
    t = 0
    metric.reset()
    for batchX, batchY in load_data_frame('dbpedia_train.csv',
                                          batch_size=BATCH_SIZE,
                                          shuffle=True):
        # Wrap X and y in DataBatch (need to be in a list)
        batch = DataBatch(data=[batchX], label=[batchY])
        # Push data forwards and update metric
        # For training + testing
        mod.forward(batch)
        mod.update_metric(metric, batch.label)
        # Get weights and update
        # For training only
        mod.backward()
        mod.update()
        # Log every 128,000 examples (1000 * BATCH)
        t += 1
        if t % 1000 == 0:
            toc = time.time()
            train_t = toc - tic
            metric_m, metric_v = metric.get()
            print("epoch: %d iter: %d metric(%s): %.4f dur: %.0f" % (epoch, t, metric_m, metric_v, train_t))

    print("Finished epoch, testing ...")
    test_net(epoch)

print("Finished in %.0f seconds" % (time.time() - tic))

# To get 4 epochs a day (20 in 5 days), need to do 112,500 batches
# in 1440 minutes -> 13 mins per 1000 batches (around 800 sec)
