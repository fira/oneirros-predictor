#/usr/bin/env python

import keras
import numpy as np
import pandas as pd
import pprint
import csv
import math

# Config
convwindow = 4
timesteps = 32
predict_steps = 32
features = 17
labels_idx = [ 10, 11, 12, 13]
dataset = pd.read_csv("norm_data.csv")
#dataset = pd.read_csv("testbash2.csv")

# Code
pp = pprint.PrettyPrinter()
window_size = timesteps / convwindow
no_samples = len(dataset.index)-timesteps-window_size-predict_steps+1
most_recent_sample = predict_steps+window_size
oldest_sample = most_recent_sample+no_samples
batch_size = 1 # Don't change.

def prepare_samples(dataset): 
  features = prepare_feature_persample(dataset, most_recent_sample, oldest_sample)
  labels = prepare_label_persample(dataset, most_recent_sample, oldest_sample)
  predictset = prepare_feature_persample(dataset, 0, predict_steps)

  return features, labels, predictset

def prepare_label_persample(dataset, start_sample, end_sample): 
  label_array = []
  for sample in reversed(range(start_sample, end_sample)):
    label_array.append( prepare_label_perlabel ( dataset, sample ) )
  return label_array

def prepare_label_perlabel(dataset, sample):
  label_array = []
  for labeli in labels_idx: 
    label_set = prepare_label_perwindow(dataset, labeli, sample) 
    np_label_set = np.array(label_set).reshape(( 1, window_size, 1, 1, predict_steps ))
    label_array.append(np_label_set)
  return label_array

def prepare_label_perwindow(dataset, label, sample):
  window = []
  for wini in range(window_size): 
    window.append( prepare_label_perstep(dataset, label, sample, wini) )
  return window

# Labels have predict_steps with future values to compare with exit of RNN
def prepare_label_perstep(dataset, label, sample, window):
  label_set = []
  for predictstep in range(0, predict_steps): 
    label_set.append( dataset.iloc[sample-predictstep-window, label] )
  return label_set

  

def prepare_feature_persample(dataset, start_sample, end_sample):
  feature_array = []
  for samplei in reversed(range(start_sample, end_sample)): 
     feature_array.append( prepare_feature_perfeature( dataset, samplei ))
  return feature_array

def prepare_feature_perfeature(dataset, sample):
  feature_array = []
  for featurei in range(1, features+1):
     feature_set = prepare_feature_perwindow(dataset, featurei, sample) 
     np_feature_set = np.array(feature_set).reshape(( 1, timesteps/convwindow, 1, convwindow, 1 ))
     feature_array.append(np_feature_set)
  return feature_array

def prepare_feature_perwindow(dataset, feature, sample):
  window = []
  for wini in reversed(range(0, window_size)): 
     window.append( prepare_feature_perstep(dataset, feature, sample, wini) )
  return window

# Features have timesteps with past values for conv
def prepare_feature_perstep(dataset, feature, sample, window): 
  feature_set = []
  for timestepi in reversed(range(0, convwindow)): 
    feature_set.append( dataset.iloc[sample+timestepi+window, feature] )
  return feature_set

X, Y, Xp = prepare_samples(dataset)

# Generate model
# inputs[n] = n+1 th model input = n+2 th csv input (time discarded) 

inputs = []
for i in range(1, features+1): 
  inputs.append ( keras.layers.Input ( name = "input_var_" + str(i), batch_shape = (1, timesteps/convwindow, 1, convwindow, 1), shape = (timesteps/convwindow, 1, convwindow, 1), dtype = 'float32' ) )


# intput[0, 1, 2] = hospital, military_base, school
# intput[3, 4, 5] = missile, port, airport
# intput[6, 7]    = power_plant, spaceport
# intput[8]       = housing_fund
# intput[9, 10]   = oil, ore
# intput[11,12]   = uranium, diamond
# intput[13,14]   = tanks, bombers
# intput[15,16]   = battleships, lunar_tanks


# Oil depends on hospitals / bases / school / housing / tank / bomber / battleship and ofc oil
oil_input = keras.layers.Concatenate(axis=4, name = "oil_input")([inputs[0], inputs[1], inputs[2], inputs[8], inputs[9]])
oil_lstm = keras.layers.ConvLSTM2D(filters = 128, kernel_size = (1, convwindow), return_sequences = True, stateful = True)(oil_input)
oil_output = keras.layers.TimeDistributed( keras.layers.Dense( predict_steps ) )(oil_lstm)

# Ore depends on hospitals / bases / shcool / housing / tank / bomber / battleship and ore
ore_input = keras.layers.Concatenate(axis=4, name = "ore_input")([inputs[0], inputs[1], inputs[2], inputs[8], inputs[10]])
ore_lstm = keras.layers.ConvLSTM2D(filters = 128, kernel_size = (1, convwindow), return_sequences = True, stateful = True)(ore_input) 
ore_output = keras.layers.TimeDistributed( keras.layers.Dense( predict_steps ) )(ore_lstm)

# Uranium depends on power_plant, spaceport, uranium, lunar_tanks
ura_input = keras.layers.Concatenate(axis=4, name = "ura_input")([inputs[6], inputs[7], inputs[11], inputs[16]])
ura_lstm = keras.layers.ConvLSTM2D(filters = 128, kernel_size = (1, convwindow), return_sequences = True, stateful = True)(ura_input) 
ura_output = keras.layers.TimeDistributed( keras.layers.Dense( predict_steps ) )(ura_lstm)

# Dia depends on missile/port/airport, power/space, bombers
dia_input = keras.layers.Concatenate(axis=4, name = "dia_input")([inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[14]])
dia_lstm = keras.layers.ConvLSTM2D(filters = 128, kernel_size = (1, convwindow), return_sequences = True, stateful = True)(inputs[12]) 
dia_output = keras.layers.TimeDistributed( keras.layers.Dense( predict_steps ) )(dia_lstm)

model = keras.models.Model ( inputs = inputs, outputs = [ oil_output, ore_output, ura_output, dia_output ] )
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer = optimizer, loss='mean_squared_error', metrics=['accuracy'], sample_weight_mode='temporal')
model.summary()

# Make it so each further timestep is less relevant for prediction
distributed = []
for win in range(window_size):
  for step in range(predict_steps): 
#    distributed.append( (3.0) / ( math.log(2, step+10) ) )
     distributed.append(1.0)

distributed_np = np.array ( distributed ).reshape( ( window_size, predict_steps ) )
distributed_outputs = { 
  "oil_lstm": distributed_np,
  "ore_lstm": distributed_np,
  "ura_lstm": distributed_np,
  "dia_lstm": distributed_np
}

# Fit the model sample by sample, for 5 epochs
for epoch in range(5):
  model.reset_states()
  for sample in range(int(no_samples)):
    print("[Epoch " + str(epoch) + "] Training on batch/sample " + str(sample))
  #  pp.pprint(X[sample][0])
  #  pp.pprint(Y[sample][0])
  #  print("====================================================")
    model.train_on_batch( X[sample], Y[sample], sample_weight = distributed_outputs )
 
# Go back
model.reset_states()
# Reload older steps leading to this one, and predict with last one
for sample in reversed(range(0, predict_steps)):
  Yp = model.predict(Xp[sample]) 

np_Yp = np.array(Yp).reshape( (4, timesteps/convwindow, predict_steps) ) # Outputs * Windows * Steps

# Get time of predicted step
starttime = dataset.iloc[0, 0]

print("Outputting prediction starting at " + str(int(starttime)))

with open('results.csv', mode='w') as resfile:
  writer = csv.writer(resfile, delimiter=',')
  writer.writerow ([ 'time', 'oil', 'ore', 'uranium', 'diamonds' ])
  for i in range(0, predict_steps):
    starttime += 3600
    writer.writerow ( [ starttime, np_Yp[0, 0, i], np_Yp[1, 0, i], np_Yp[2, 0, i], np_Yp[3, 0, i] ] )
