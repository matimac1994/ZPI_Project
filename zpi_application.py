import pandas as pd
import tensorflow as tf
from tensorflow import contrib
from numpy import genfromtxt


def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return tf.dtypes.cast(features, tf.float32), labels


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def readCSV(source, column_names):
    return pd.read_csv(source, sep=',', decimal='.', header=None, names=column_names)

tf.enable_eager_execution()

net_split_train_csv_name = "splitTrainNetLogs.csv"
net_split_test_csv_name = "splitTestNetLogs.csv"
net_train_data_source = "NETtrain.csv"
net_test_data_source = "NETtest.csv"


net_train_column_names = ['time', 'duration', 'source_computer', 'source_port', 'destination_computer', 'destination_port',
                      'protocol', 'packet_count', 'byte_count', 'class']
net_test_column_names = ['time', 'duration', 'source_computer', 'source_port', 'destination_computer', 'destination_port',
                      'protocol', 'packet_count', 'byte_count']

split_train_column_names = ['time', 'duration', 'packet_count', 'byte_count', 'class']
split_test_column_names = ['time', 'duration', 'packet_count', 'byte_count']

net_train_data_CSV = readCSV(net_train_data_source, net_train_column_names)
net_test_data_CSV = readCSV(net_test_data_source, net_test_column_names)

split_train_data = net_train_data_CSV[split_train_column_names]
split_test_data = net_test_data_CSV[split_test_column_names]

split_train_data.to_csv(net_split_train_csv_name, index=False, header=False)
split_test_data.to_csv(net_split_test_csv_name, index=False, header=False)

feature_names = split_train_column_names[:-1]
label_name = split_train_column_names[-1]

class_names = ['No attack', 'Attack']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    net_split_train_csv_name,
    batch_size,
    column_names=split_train_column_names,
    label_name=label_name,
    num_epochs=2
)

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(2)
])

predictions = model(features)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

global_step = tf.Variable(0)

loss_value, grads = grad(model, features, labels)

optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

tfe = contrib.eager

train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)

        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

test_dataset = genfromtxt(net_split_test_csv_name, delimiter=',')

predictions = model(test_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))

#changes......