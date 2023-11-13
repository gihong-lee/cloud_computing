import click
import os
import sys
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

width = 32
height = 32
channels = 3
num_classes = 10 

activation = 'relu'
optimizer = 'adam'

def set_config(task_index, port = 15000):
  node_list = os.environ['SLURM_NODELIST']
  base, nodes = node_list.split('-', 1)
  nodes = nodes[1:-1].split('-')

  node_idx_list = []
  for i in range(int(nodes[0]),int(nodes[1])+1):
    node_idx_list.append(f"0{i}")

  tf_config = {
        'cluster': {'worker': [f"{base}-{node_idx}:{port}" for node_idx in node_idx_list]},
        'task': {'type': 'worker', 'index': task_index}
        }
  
  os.environ['TF_CONFIG'] = json.dumps(tf_config)
  print(f"# of worker : {len(node_idx_list)}")

def dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  print(len(x_train))
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(width,height,channels)),
      tf.keras.layers.Conv2D(32, 3, activation=activation),
      tf.keras.layers.Conv2D(64, 3, activation=activation),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=activation),
      tf.keras.layers.Dense(64, activation=activation),
      tf.keras.layers.Dense(num_classes)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=optimizer,
      metrics=['accuracy'])
  return model

@click.command()
def train_dense_model_click():
    return train_dense_model()


def train_dense_model():
    sample_num = 50000
    task_index = int(os.environ['SLURM_PROCID'])
    num_workers = int(os.environ['SLURM_NPROCS'])
    set_config(task_index)

    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    per_worker_batch_size = 512
    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = dataset(global_batch_size)
    with mirrored_strategy.scope():
        model = build_and_compile_cnn_model()
    
    # training and inference
    model.fit(multi_worker_dataset, epochs=50, steps_per_epoch=sample_num//global_batch_size+1)
    return True

if __name__ == '__main__':
    train_dense_model_click()
