import pandas as pd
import tensorflow as tf
from tensorflow import contrib
from numpy import genfromtxt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return tf.dtypes.cast(features, tf.float32), labels


def loss(new_model, x, y):
    y_ = new_model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(new_model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(new_model, inputs, targets)
    return loss_value, tape.gradient(loss_value, new_model.trainable_variables)


def read_csv(source, column_names):
    return pd.read_csv(source, sep=',', decimal='.', header=None, names=column_names)


tf.enable_eager_execution()

proc_train_data_source = "PROCtrain.csv"
proc_test_data_source = "PROCtest.csv"

proc_split_train_csv_name = "splitTrainProcLogs.csv"
proc_split_test_csv_name = "splitTestProcLogs.csv"

proc_train_column_names = ['time', 'user@domain', 'computer', 'process_name', 'start_end', 'class']
proc_test_column_names = ['time', 'user@domain', 'computer', 'process_name', 'start_end']

proc_train_data_CSV = readCSV(proc_train_data_source, proc_train_column_names)
proc_test_data_CSV = readCSV(proc_test_data_source, proc_test_column_names)

X = proc_train_data_CSV.iloc[:, 1:5].values
Y = proc_test_data_CSV.iloc[:, 1:5].values

# Replace string values into numeric

labelencoder1 = LabelEncoder()
X[:, 0] = labelencoder1.fit_transform(X[:, 0])
Y[:, 0] = labelencoder1.fit_transform(Y[:, 0])

labelencoder2 = LabelEncoder()
X[:, 1] = labelencoder2.fit_transform(X[:, 1])
Y[:, 1] = labelencoder2.fit_transform(Y[:, 1])

labelencoder3 = LabelEncoder()
X[:, 2] = labelencoder3.fit_transform(X[:, 2])
Y[:, 2] = labelencoder3.fit_transform(Y[:, 2])

labelencoder4 = LabelEncoder()
X[:, 3] = labelencoder4.fit_transform(X[:, 3])
Y[:, 3] = labelencoder4.fit_transform(Y[:, 3])

proc_train_data_CSV.iloc[:, 1:5] = X
proc_test_data_CSV.iloc[:, 1:5] = Y

proc_train_data_CSV.to_csv(proc_split_train_csv_name, header=False, index=False)
proc_test_data_CSV.to_csv(proc_split_test_csv_name, header=False, index=False)

feature_names = proc_train_column_names[:-1]
label_name = proc_train_column_names[-1]

class_names = ['No attack', 'Attack']

batch_size = 10

train_dataset = tf.data.experimental.make_csv_dataset(
    proc_split_train_csv_name,
    batch_size,
    column_names=proc_train_column_names,
    label_name=label_name,
    num_epochs=2
)

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(5,)),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
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

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)

        epoch_loss_avg(loss_value)
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

test_dataset = genfromtxt(proc_split_test_csv_name, delimiter=',')

predictions = model(test_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))
