"""
Attempt to replicate torch model using MXNET:
https://github.com/zhangxiangxiao/Crepe

"To run this example succesfully you will also need a NVidia GPU with at
least 3GB of memory.
Otherwise, you can configure the model in train/config.lua for less parameters."

ISSUE: mxnet requires way more than 3GB memory for the model ...

ERROR:
[01:38:23] D:\chhong\mxnet\dmlc-core\include\dmlc/logging.h:235: [01:38:23] d:\chhong\mxnet\src\storage\
./gpu_device_storage.h:39: Check failed: e == cudaSuccess || e == cudaErrorCudartUnloading CUDA: out of memory
Traceback (most recent call last):
  File "C:/Users/superuser/PycharmProjects/cnn_crepe/cnn.py", line 190, in <module>
    label_shapes=[('softmax_label', (BATCH_SIZE,))])
  File "C:\Anaconda2\lib\site-packages\mxnet-0.7.0-py2.7.egg\mxnet\module\module.py", line 256, in bind
    shared_group, logger=self.logger)
  File "C:\Anaconda2\lib\site-packages\mxnet-0.7.0-py2.7.egg\mxnet\module\executor_group.py", line 95, in __init__
    self.bind_exec(data_shapes, label_shapes, shared_group)
  File "C:\Anaconda2\lib\site-packages\mxnet-0.7.0-py2.7.egg\mxnet\module\executor_group.py", line 123, in bind_exec
    self.execs.append(self._bind_ith_exec(i, data_shapes, label_shapes, shared_group))
  File "C:\Anaconda2\lib\site-packages\mxnet-0.7.0-py2.7.egg\mxnet\module\executor_group.py", line 379, in _bind_ith_exec
    arg_arr = nd.zeros(arg_shapes[j], context, dtype=arg_types[j])
  File "C:\Anaconda2\lib\site-packages\mxnet-0.7.0-py2.7.egg\mxnet\ndarray.py", line 688, in zeros
    arr = empty(shape, ctx, dtype)
  File "C:\Anaconda2\lib\site-packages\mxnet-0.7.0-py2.7.egg\mxnet\ndarray.py", line 513, in empty
    return NDArray(handle=_new_alloc_handle(shape, ctx, False, dtype))
  File "C:\Anaconda2\lib\site-packages\mxnet-0.7.0-py2.7.egg\mxnet\ndarray.py", line 65, in _new_alloc_handle
    ctypes.byref(hdl)))
  File "C:\Anaconda2\lib\site-packages\mxnet-0.7.0-py2.7.egg\mxnet\base.py", line 77, in check_call
    raise MXNetError(py_str(_LIB.MXGetLastError()))
mxnet.base.MXNetError: [01:38:23] d:\chhong\mxnet\src\storage\./gpu_device_storage.h:39: Check failed:
e == cudaSuccess || e == cudaErrorCudartUnloading CUDA: out of memory

"""

import numpy as np
import pandas as pd
import mxnet as mx
import wget
import time
import os.path
from mxnet.io import DataBatch

ctx = mx.gpu(0)  # Run on one GPU

AZ_ACC = "amazonsentimenik"
AZ_CONTAINER = "textclassificationdatasets"
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")

FEATURE_LEN = 1014  # First 1014 characters
BATCH_SIZE = 8  # 128 is too big!!
NUM_FILTERS = 16  # 256 is too big!!!
EPOCHS = 1000  # config.main.epoches = 5000
SD = 0.05  # std for gaussian distribution
NOUTPUT = 2  # good or bad
DATA_SHAPE = (BATCH_SIZE, 1, FEATURE_LEN, len(ALPHABET))

print("Alphabet %d characters: " % len(ALPHABET), ALPHABET)


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
        data=input_x, kernel=(7, 7), num_filter=NUM_FILTERS)
    relu1 = mx.symbol.Activation(
        data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(1, 1))

    # 2. 336 x 256
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(7, 7), num_filter=NUM_FILTERS)
    relu2 = mx.symbol.Activation(
        data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(
        data=relu2, pool_type="max", kernel=(3, 3), stride=(1, 1))

    # 3. 110 x 256
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), num_filter=NUM_FILTERS)
    relu3 = mx.symbol.Activation(
        data=conv3, act_type="relu")

    # 4. 108 x 256
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), num_filter=NUM_FILTERS)
    relu4 = mx.symbol.Activation(
        data=conv4, act_type="relu")

    # 5. 106 x 256
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), num_filter=NUM_FILTERS)
    relu5 = mx.symbol.Activation(
        data=conv5, act_type="relu")

    # 6. 104 x 256
    conv6 = mx.symbol.Convolution(
        data=relu5, kernel=(3, 3), num_filter=NUM_FILTERS)
    relu6 = mx.symbol.Activation(
        data=conv6, act_type="relu")
    pool6 = mx.symbol.Pooling(
        data=relu6, pool_type="max", kernel=(3, 3), stride=(1, 1))

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
    for batchX, batchY in load_data_frame('amazon_review_polarity_test.csv', batch_size=BATCH_SIZE):
        batch = DataBatch(data=[batchX], label=[batchY])
        mod.forward(batch)
        mod.update_metric(metric, batch.label)
    metric_m, metric_v = metric.get()
    print("TEST(%s): %.4f" % (metric_m, metric_v))

# Train
print("started training")
tic = time.time()
# Evaluation metric:
metric = mx.metric.Accuracy()
# Train EPOCHS
for epoch in range(EPOCHS):
    t = 0
    metric.reset()
    for batchX, batchY in load_data_frame('amazon_review_polarity_train.csv',
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
        # Log every 10 batches
        t += 1
        if t % 1 == 0:
            toc = time.time()
            train_t = toc - tic
            metric_m, metric_v = metric.get()
            print("epoch: %d iter: %d metric(%s): %.4f dur: %.0f" % (epoch, t, metric_m, metric_v, train_t))

    print("Finished epoch, testing ...")
    test_net(epoch)

print("Finished in %.0f seconds" % (time.time() - tic))


