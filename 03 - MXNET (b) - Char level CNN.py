import mxnet as mx
import numpy as np
import pandas as pd
import wget
import os.path
import time
from collections import namedtuple

AZ_ACC = "amazonsentimenik"
AZ_CONTAINER = "textclassificationdatasets"
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/|_@#$%^&*~`+ =<>()[]{}")
print("Alphabet %d characters: " % len(ALPHABET), ALPHABET)
FEATURE_LEN = 1014
BATCH_SIZE = 50
EMBED_SIZE = 16
NUM_FILTERS = 256
NUM_EPOCHS = 10


def download_file(url):
    # Create file-name
    local_filename = url.split('/')[-1]

    if os.path.isfile(local_filename):
        print("The file %s already exist in the current directory\n" % local_filename)
    else:
        # Download
        print("downloading ...\n")
        wget.download(url)
        print('saved data\n')


def load_data_frame(infile, batch_size=128, shuffle=True):
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
    # print(Y_split[:30])
    # drop columns
    df.drop(['text', 'summary', 'sentiment'], axis=1, inplace=True)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Dictionary to create character vectors
    character_hash = pd.DataFrame(np.identity(len(ALPHABET)), columns=ALPHABET)
    # Yield mini-batch amount of character vectors
    for ti, tx in enumerate(df.rev):
        if ti % batch_size == 0:
            # output
            if ti > 0:
                yield X_split, y_split[ti - batch_size:ti]
            X_split = np.zeros([batch_size, 1, FEATURE_LEN, len(ALPHABET)], dtype='int')

        chars = list(tx)
        for ci, ch in enumerate(chars):
            if ch in ALPHABET:
                X_split[ti % batch_size][0][ci] = np.array(character_hash[ch])


def example():
    count = 0
    for minibatch in load_data_frame('amazon_review_polarity_test.csv', batch_size=5, shuffle=True):
        count += 1
        print(minibatch[-1])
        if count == 6:
            break


def create_crepe():
    """
    Number of features = 70, input feature length = 1014

    2 Dropout modules inserted between 3 fully-connected layers (0.5)

    Number of output units for last layer = num_classes
    For polarity test = 2
    """

    """
    input_x = mx.sym.Variable('data')# placeholder for input
    input_y = mx.sym.Variable('softmax_label') # placeholder for output

    # 1. alphabet x 1014
    conv1 = mx.symbol.Convolution(
        data=input_x, kernel=(7, 7), num_filter=256)
    relu1 = mx.symbol.Activation(
        data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(1, 1))

    # 2. 336 x 256
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(7, 7), num_filter=256)
    relu2 = mx.symbol.Activation(
        data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(
        data=relu2, pool_type="max", kernel=(3, 3), stride=(1, 1))

    # 3. 110 x 256
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), num_filter=256)
    relu3 = mx.symbol.Activation(
        data=conv3, act_type="relu")

    # 4. 108 x 256
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), num_filter=256)
    relu4 = mx.symbol.Activation(
        data=conv4, act_type="relu")

    # 5. 106 x 256
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), num_filter=256)
    relu5 = mx.symbol.Activation(
        data=conv5, act_type="relu")

    # 6. 104 x 256
    conv6 = mx.symbol.Convolution(
        data=relu5, kernel=(3, 3), num_filter=256)
    relu6 = mx.symbol.Activation(
        data=conv6, act_type="relu")
    pool6 = mx.symbol.Pooling(
        data=relu6, pool_type="max", kernel=(3, 3), stride=(1, 1))

    # 34 x 256
    flatten = mx.symbol.Flatten(data=pool6)

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
        data=drop2, num_hidden=2)

    crepe = mx.symbol.SoftmaxOutput(
        data=fc3, label=input_y, name="softmax")
    """

    input_x = mx.sym.Variable('data')  # placeholder for input
    input_y = mx.sym.Variable('softmax_label')  # placeholder for output

    num_label = 2
    filter_list = [3, 4, 5]
    num_filter = 100

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(
            data=input_x, kernel=(filter_size, len(ALPHABET)), num_filter=num_filter)
        relui = mx.sym.Activation(
            data=convi, act_type='relu')
        pooli = mx.sym.Pooling(
            data=relui, pool_type='max', kernel=(FEATURE_LEN - filter_size + 1, 1), stride=(1, 1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(BATCH_SIZE, total_filters))
    
    # dropout layer
    dropout = 0.5
    h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    
    # fully connected
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')
    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)
    
    # softmax output
    crepe = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')
    return crepe

"""
# Superseded by bit below ...
# create the NN
ctx = mx.cpu()
cnn = create_crepe()

m = mx.model.FeedForward(
    ctx=ctx,
    symbol=cnn,
    num_epoch=10,
    learning_rate=0.01,
    momentum=0.9,
    wd=0.00001
)

# train NN
for epoch in range(10):

    num_correct = 0
    num_total = 0

    for batchX, batchY in load_data_frame('amazon_review_polarity_test.csv', BATCH_SIZE * 10):
        train_iter = mx.io.NDArrayIter(batchX, batchY, batch_size=BATCH_SIZE, shuffle=True)
        m.fit(X=train_iter)

        # evaluate on training
        num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
        num_total += len(batchY)

    # end of training loop
    train_acc = num_correct * 100 / float(num_total)
    print("Iter [%d], Training Accuracy: %.3f" % (epoch, train_acc))
"""

cnn = create_crepe()
ctx = mx.cpu()
initializer = mx.initializer.Uniform(0.1)

arg_names = cnn.list_arguments()

input_shapes = {'data': (BATCH_SIZE, 1, FEATURE_LEN, len(ALPHABET))}

arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
args_grad = {}
for shape, name in zip(arg_shape, arg_names):
    if name in ['softmax_label', 'data']:  # input, output
        continue
    args_grad[name] = mx.nd.zeros(shape, ctx)

cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

param_blocks = []
arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
for i, name in enumerate(arg_names):
    if name in ['softmax_label', 'data']:  # input, output
        continue
    initializer(name, arg_dict[name])

    param_blocks.append((i, arg_dict[name], args_grad[name], name))

out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

data = cnn_exec.arg_dict['data']
label = cnn_exec.arg_dict['softmax_label']

CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])
m = CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)

# Train
learning_rate = 0.01
epochs = 20
opt = mx.optimizer.create('rmsprop')
opt.lr = learning_rate

updater = mx.optimizer.get_updater(opt)

for iteration in range(NUM_EPOCHS):
    tic = time.time()
    num_correct = 0
    num_total = 0
    for batchX, batchY in load_data_frame('amazon_review_polarity_test.csv', BATCH_SIZE):

        m.data[:] = batchX
        m.label[:] = batchY

        # forward
        m.cnn_exec.forward(is_train=True)

        # eval on training data
        num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
        num_total += len(batchY)

    # end of training loop
    toc = time.time()
    train_time = toc - tic
    train_acc = num_correct * 100 / float(num_total)
    print('Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc))