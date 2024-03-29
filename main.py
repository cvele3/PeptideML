# tensorinjo = "tensorflow-2.11.0-cp38-cp38-win_amd64.whl"
# rdkitinjo = "rdkit_pypi-2022.9.5-cp38-cp38-win_amd64.whl"
# stelargrfinjo = "stellargraph-1.2.1-py3-none-any.whl"
#
# import pip
#
# def install_whl(path):
#    pip.main(['install', path])

#install_whl(tensorinjo)
#install_whl(rdkitinjo)
#install_whl(stelargrfinjo)



from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator


from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import pandas as pd
import rdkit.Chem
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import numpy as np
from stellargraph import StellarGraph

# Import required libraries
from rdkit import Chem
from rdkit.Chem import Draw
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow.keras import Model, optimizers, losses, metrics
import numpy as np
import pandas as pd


filepath_raw = 'avpdb_smiles.xlsx'
data_file = pd.read_excel(filepath_raw, header=0, usecols=["smiles", "label"])


listOfTuples = []

data_file.reset_index()
for index, row in data_file.iterrows():
    smiles = row['smiles']
    label = row["label"]
    molecule = (row["smiles"], row["label"])
    listOfTuples.append(molecule)

ZeroActivity = 0
OneActivity = 0
stellarGraphAllList = []


from collections import Counter

# Get a list of all unique elements in the molecules
all_elements = []
for molecule in listOfTuples:
    mol = Chem.MolFromSmiles(molecule[0])
    atoms = mol.GetAtoms()
    for atom in atoms:
        element = atom.GetSymbol()
        if element not in all_elements:
            all_elements.append(element)

# Create a dictionary that maps each element to a unique index
element_to_index = {element: index for index, element in enumerate(all_elements)}

for molecule in listOfTuples:

    smileString = molecule[0]
    smileLabel = molecule[1]

    # Convert the SMILES string into a molecular graph using RDKit
    mol = Chem.MolFromSmiles(smileString)
    atoms = mol.GetAtoms()
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

    # Convert the RDKit atom objects to a Numpy array of node features
    node_features = []
    for atom in atoms:
        element = atom.GetSymbol()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        num_radical_electrons = atom.GetNumRadicalElectrons()
        hybridization = atom.GetHybridization().real
        aromatic = atom.GetIsAromatic()
        element_onehot = [0] * len(all_elements)
        element_onehot[element_to_index[element]] = 1
        node_features.append(element_onehot)
        #node_features.append(element_onehot + [degree, formal_charge, num_radical_electrons, hybridization, aromatic])
    node_features = np.array(node_features)

    # Convert the edges to a pandas DataFrame
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    # Create a StellarGraph object from the molecular graph
    G = StellarGraph(nodes=node_features, edges=edges_df)


    if smileLabel == 1 and OneActivity < 1000:
      OneActivity += 1
      skup = (G, smileLabel)
      stellarGraphAllList.append(skup)

    if smileLabel == 0 and ZeroActivity < 1000:
        ZeroActivity += 1
        skup = (G, smileLabel)
        stellarGraphAllList.append(skup)


print(ZeroActivity)
print(OneActivity)

print(len(stellarGraphAllList))



import pandas as pd

# assume that the 'stellarGraphAllList' variable is defined somewhere in the code

graphs = []
labels = []

for triple in stellarGraphAllList:
  grafinjo = triple[0]
  active = triple[1]
  graphs.append(grafinjo)
  labels.append(active)


import pandas as pd

# assume that the 'stellarGraphAllList' variable is defined somewhere in the code

graph_labels = pd.Series(labels)

print(graph_labels.value_counts().to_frame())

graph_labels = pd.get_dummies(graph_labels, drop_first=True)

generator = PaddedGraphGenerator(graphs=graphs)

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import Sequence


epochs = 10000

# Define the number of rows for the output tensor and the layer sizes
k = 35
layer_sizes = [32, 32, 32, 1]

# Create the DeepGraphCNN model
dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

# Create the model and compile it
model = Model(inputs=x_inp, outputs=predictions)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
import tensorflow.keras.backend as K

gm_values = []
precision_values = []
recall_values = []
f1_values = []
roc_auc_values = []


# Define evaluation metrics
def metrics(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), K.floatx())
    y_true = K.cast(y_true, K.floatx())

    y_pred_np = K.eval(y_pred)  # Convert y_pred tensor to NumPy array

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_np).ravel()
    tpr = K.variable(tp / (tp + fn + K.epsilon()))
    tnr = K.variable(tn / (tn + fp + K.epsilon()))
    gm = K.sqrt(tpr * tnr)
    precision = precision_score(y_true, y_pred_np)
    recall = recall_score(y_true, y_pred_np)
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    roc_auc = roc_auc_score(y_true, y_pred_np)

    print("GM: ", gm)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("ROC AUC: ", roc_auc)

    gm_values.append(K.get_value(gm))
    precision_values.append(K.get_value(precision))
    recall_values.append(K.get_value(recall))
    f1_values.append(K.get_value(f1))
    roc_auc_values.append(K.get_value(roc_auc))


restOfMetrics_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: metrics(test_gen.targets, model.predict(test_gen)))



mcc_values = []

# Create a LambdaCallback to calculate MCC at the end of each epoch
def mcc_metric(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred = y_pred.round()
    mcc = matthews_corrcoef(y_true, y_pred)
    print("MCC: ", mcc)
    mcc_values.append(mcc)

mcc_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: mcc_metric(test_gen.targets, model.predict(test_gen)))
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12)

model.compile(
    optimizer=Adam(lr=0.0001),
    loss=binary_crossentropy,
    metrics=["acc"]
)

# Add evaluation metrics to the model
model.metrics_names.append('mcc')
model.metrics_names.append('tpr')
model.metrics_names.append('tnr')
model.metrics_names.append('gm')
model.metrics_names.append('precision')
model.metrics_names.append('recall')
model.metrics_names.append('f1')
model.metrics_names.append('roc_auc')

imageCounter = 0
metrics_dir = "plotHistory 444-598 split with k35 [32,32,32,1]_newMetrics/"



cv = StratifiedKFold(n_splits=5, shuffle=True)

import matplotlib.pyplot as plt
import numpy as np

histories = []

# Use the cross-validator to get the train and test indices
for train_index, test_index in cv.split(graphs, graph_labels):
    # Extract the train and test sets using the indices
    graphs = np.array(graphs)
    X_train, X_test = graphs[train_index.astype(int)], graphs[test_index.astype(int)]
    y_train, y_test = graph_labels.iloc[train_index.astype(int)], graph_labels.iloc[test_index.astype(int)]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    gen = PaddedGraphGenerator(graphs=graphs)

    # Create generators for the train and test sets
    train_gen = gen.flow(
        X_train,
        targets=y_train,
        batch_size=50,
        symmetric_normalization=False,
    )

    val_gen = gen.flow(
        X_val,
        targets=y_val,
        batch_size=50,
        symmetric_normalization=False,
    )

    test_gen = gen.flow(
        X_test,
        targets=y_test,
        batch_size=50,
        symmetric_normalization=False,
    )

    # Train the model and evaluate on the test set
    history = model.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=val_gen, shuffle=True, callbacks=[mcc_callback, callback, restOfMetrics_callback]
    )

    imageCounter += 1;

    # Create ROC AUC plot
    plt.figure()
    plt.plot(range(len(roc_auc_values)), roc_auc_values)
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC over Epochs')
    plt.savefig('graphs/' + metrics_dir + 'roc_auc_plot_'+ str(imageCounter) +'.png')

    # Create F1 plot
    plt.figure()
    plt.plot(range(len(f1_values)), f1_values)
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs')
    plt.savefig('graphs/' + metrics_dir + 'f1_plot_'+ str(imageCounter) +'.png')

    # Create GM plot
    plt.figure()
    plt.plot(range(len(gm_values)), gm_values)
    plt.xlabel('Epochs')
    plt.ylabel('G-Measure')
    plt.title('G-Measure over Epochs')
    plt.savefig('graphs/' + metrics_dir + 'gm_plot_'+ str(imageCounter) +'.png')

    # Create Precision plot
    plt.figure()
    plt.plot(range(len(precision_values)), precision_values)
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision over Epochs')
    plt.savefig('graphs/' + metrics_dir + 'precision_plot_'+ str(imageCounter) +'.png')

    # Create Recall plot
    plt.figure()
    plt.plot(range(len(recall_values)), recall_values)
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall over Epochs')
    plt.savefig('graphs/' + metrics_dir + 'recall_plot_'+ str(imageCounter) +'.png')

    gm_values.clear()
    precision_values.clear()
    recall_values.clear()
    f1_values.clear()
    roc_auc_values.clear()

    histories.append(history)

    y_pred = model.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    y_pred = [0 if prob < 0.5 else 1 for prob in y_pred]

    y_test = y_test.to_numpy()
    y_test = np.reshape(y_test, (-1,))
    mcc_metric(y_test, y_pred)
    # returnMCC = mcc_metric(y_test, y_pred)
    # mcc_values.append(returnMCC)


import os
import matplotlib.pyplot as plt
import stellargraph as sg

#save_dir = r"C:\Users\jcvetko\Desktop\stuff\school\6. semestar\Zavrsni rad\plotHistory 444-598 split with k30 [30,30,30,1]"
save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\plotHistory 444-598 split with k35 [32,32,32,1]_history"

for i, history in enumerate(histories):
    fig = sg.utils.plot_history(history, individual_figsize=(7, 4), return_figure=True)
    fig.savefig(os.path.join(save_dir, f"history_{i}.png"))
    plt.close(fig)





test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("AVG MCC: ")
print(np.mean(mcc_values))
print("MAX MCC: ")
print(np.max(mcc_values))
print("MIN MCC: ")
print(np.min(mcc_values))