
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
seed = 0
tf.random.set_seed(seed)
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test  = pd.read_csv('test.csv')

# EDA
## Data Discovery
train

## Split the data as features and target
X_train=train.drop('Label',axis=1)
y_train=train['Label']
X_test=test.drop('Label',axis=1)
y_test=test['Label']
X_val=val.drop('Label',axis=1)
y_val=val['Label']
X_train

### Convert categorical data into numerical by one hot encoding
X_train_numeric = pd.get_dummies(X_train, drop_first=True)
X_val_numeric = pd.get_dummies(X_val, drop_first=True)
X_test_numeric = pd.get_dummies(X_test, drop_first=True)
X_train_numeric

## Number of instances per class
y_train.value_counts()

### Number of null values per feature
X_train_numeric.info()

### There is no null values per features

# Checking the outliers via boxplot
for i in range(0,18,9) :
    fig,axis =plt.subplots(1,3,figsize=(30,8))
    sns.boxplot(data=X_train.iloc[:,i:i+3],ax=axis[0])
    sns.boxplot(data=X_train.iloc[:,i+3:i+6],ax=axis[1])
    sns.boxplot(data=X_train.iloc[:,i+6:i+9],ax=axis[2])
    plt.show()

## Count the outliers
for i,col in enumerate(X_train_numeric.columns):
    if i > 18 :
        break
    q_low = X_train[col].quantile(0.25)
    q_hi  = X_train[col].quantile(0.75)
    IQR = (q_hi-q_low)
    df_filtered = X_train[(X_train[col] > (q_hi + 1.5 * IQR)) | (X_train[col] < (q_low - 1.5 * IQR))]
    print(f'Feature name {col} --> number of outliers is {len(df_filtered)}')

### The statistical analysis for every feature (mean, std, min, max)
X_train_numeric.describe()

## Prepare the data for neural network model
### Data Scaling for making the all the features in the same range
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_numeric), columns=X_train_numeric.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_numeric), columns=X_val_numeric.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_numeric), columns=X_test_numeric.columns)
X_train_scaled
X_train_scaled.info()

### Convert to tensors
X_train_scaled_tensor = tf.convert_to_tensor(X_train_scaled, dtype=np.float64)
X_val_scaled_tensor = tf.convert_to_tensor(X_val_scaled, dtype=np.float64)
X_test_scaled_tensor = tf.convert_to_tensor(X_test_scaled, dtype=np.float64)
X_train_scaled_tensor

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)
y_train_encoded

# Build MLP General Model
## Helper Functions
!pip install tensorflow-addons
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense
from tensorflow_addons.optimizers import AdamW
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
%matplotlib inline

### Plot History
def plot_history(dict_of_lists, type='loss'):
    axis_idx = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for i in range(len(dict_of_lists)):
        ax = axes[axis_idx[i][0]][axis_idx[i][1]]
        # summarize history for accuracy
        ax.plot(dict_of_lists[i].history[type])
        ax.plot(dict_of_lists[i].history[f'val_{type}'])
        ax.set_title(f'Model {i+1} {type.title()}', size=15)
        ax.set_ylabel(f'{type.title()}', size=10)
        ax.set_xlabel('Epoch', size=10)
        ax.legend(['train', 'test'], loc='upper left')
    plt.suptitle(f'Models {type.title()} per Epoch', size=15, y=.93)
    plt.show()

### Build Model
def Build_model(X_train, y_train, X_val, y_val, X_test, y_test, optimizer, n_of_hidden_layers, n_neurons, activation='relu', epochs=500, batch_size=1024, early_stopping=True):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(X_train.shape[1],)))
    for _ in range(n_of_hidden_layers):
        model.add(keras.layers.Dense(n_neurons, activation=activation))
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    # Fit model
    if early_stopping:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    else:
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)
    # Evaluate the model
    train_evaluation = model.evaluate(X_train, y_train)
    test_evaluation = model.evaluate(X_test, y_test)
    validation_evaluation = model.evaluate(X_val, y_val)
    return model, history, train_evaluation, validation_evaluation, test_evaluation

### Build Experiment model and save the model
def Build_experiment(X_train, y_train, X_val, y_val, X_test, y_test, optimizer, n_of_hidden_layers, n_neurons, activation='relu', epochs=500, batch_size=1024, n_of_models=5, early_stopping=True):
    models_dict = {'models': [], 'history': []}

    models_train_acc = []
    models_test_acc = []
    models_valid_acc = []
    for j in range(n_of_models):
        # Build model
        model, history, train_evaluation, valid_evaluation, test_evaluation = Build_model(X_train, y_train, X_val, y_val, X_test, y_test, optimizer, n_of_hidden_layers, n_neurons, activation=activation, epochs=epochs, batch_size=batch_size, early_stopping=early_stopping)
        # Save data
        models_train_acc.append(train_evaluation[1])
        models_test_acc.append(test_evaluation[1])
        models_valid_acc.append(valid_evaluation[1])

        models_dict['models'].append(model)
        models_dict['history'].append(history)

    accuracies_dict = {}
    accuracies_dict["Min_train_acc"] = min(models_train_acc)
    accuracies_dict["Max_train_acc"] = max(models_train_acc)
    accuracies_dict["AVG_train_acc"] = mean(models_train_acc)

    accuracies_dict["Min_test_acc"] = min(models_test_acc)
    accuracies_dict["Max_test_acc"] = max(models_test_acc)
    accuracies_dict["AVG_test_acc"] = mean(models_test_acc)

    accuracies_dict["Min_valid_acc"] = min(models_valid_acc)
    accuracies_dict["Max_valid_acc"] = max(models_valid_acc)
    accuracies_dict["AVG_valid_acc"] = mean(models_valid_acc)

    return accuracies_dict, models_dict, models_train_acc, models_test_acc, models_valid_acc

### Finally, save the built model for the tool
last_model.save('intrusion_detection_model.h5')
