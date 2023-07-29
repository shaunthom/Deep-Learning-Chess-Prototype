import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
ag = pd.read_csv('')
for index, row in ag.iterrows():
    if row['Res'] == "Jan-00":
        ag.at[index, 'Res'] = "1-0"

def latest_form(dataset):
    
    form = []
    for index, row in dataset.iterrows():
        
        if row["White"] == 'Giri, Anish':
            if row["Res"] == '1-0':
                form.append("W")
            elif row["Res"] == '0-1':
                form.append("L")
            else:
                form.append("D")
        elif row["Black"] == 'Giri, Anish':
            if row["Res"] == '0-1':
                form.append("W")
            elif row["Res"] == '1-0':
                form.append("L")
            else:
                form.append("D")
    last_10_values = form[-1:-11:-1]
    return last_10_values
    
def present_form(dataset):
    
    form = []
    for index, row in dataset.iterrows():
        
        if row["White"] == 'Giri, Anish':
            if row["Res"] == '1-0':
                form.append("W")
            elif row["Res"] == '0-1':
                form.append("L")
            else:
                form.append("D")
        elif row["Black"] == 'Giri, Anish':
            if row["Res"] == '0-1':
                form.append("W")
            elif row["Res"] == '1-0':
                form.append("L")
            else:
                form.append("D")
    last_100_values = form[-1:-100:-1]
    return last_100_values

latest_form(ag)

def result_for_giri(row):
    if row['White'] == 'Giri, Anish':
        return 'W' if row['Res'] == '1-0' else 'L' if row['Res'] == '0-1' else 'D'
    elif row['Black'] == 'Giri, Anish':
        return 'W' if row['Res'] == '0-1' else 'L' if row['Res'] == '1-0' else 'D'
    return None

results_giri = []
last_10_results = []

for index, row in ag.iterrows():
    result = result_for_giri(row)
    if result is not None:
        results_giri.insert(0, result)
    last_10_results.append(list(results_giri[1:11]))  # copy the current state of last 10 results

ag['Last_10_Results'] = last_10_results

ag['Elo Difference w.r.t. Anish'] = ""
for index, row in ag.iterrows():
    if row["White"] == "Giri, Anish":
        x = float(row["W ELO"]) - float(row["B ELO"])
        ag.at[index, "Elo Difference w.r.t. Anish"] = x
    elif row["Black"] == "Giri, Anish":
        x = float(row["B ELO"]) - float(row["W ELO"])
        ag.at[index, "Elo Difference w.r.t. Anish"] = x

  ag_w['Num_Result'] = ""
for index, row in ag_w.iterrows():
    
    if row["White"] == "Giri, Anish":
        if row["Res"] == '1-0':
            ag_w.at[index, "Num_Result"] = 1
        elif row["Res"] == '0-1':
            ag_w.at[index, "Num_Result"] = 0
        else:
            ag_w.at[index, "Num_Result"] = 0.5
    
    if row["Black"] == "Giri, Anish":
        if row["Res"] == '0-1':
            ag_w.at[index, "Num_Result"] = 1
        elif row["Res"] == '1-0':
            ag_w.at[index, "Num_Result"] = 0
        else:
            ag_w.at[index, "Num_Result"] = 0.5

  AG["Binarized_Form"] = None
for index, row in AG.iterrows():
    k = [1 if x == "W" else 0 if x == "L" else 0.5 for x in row["Last_10_Results"]]
    AG.at[index, "Binarized_Form"] = k

AG['Binarized_Form'] = AG['Binarized_Form'].apply(lambda x: [float(val) for val in x])
cols = AG.drop(["Num_Result","White","Black","Last_10_Results"], axis=1)
output_col = AG["Num_Result"].drop(AG["Num_Result"].index[:10])
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(input_col['Elo Difference w.r.t. Anish'].values.reshape(-1, 1))

input_col['Normalized Difference'] = normalized_values
input_col = input_col.drop(["Elo Difference w.r.t. Anish"], axis=1)
X = input_col
Y = output_col
train_size = int(250 * 0.7)
X_train = X[:train_size]
Y_train = Y[:train_size]

# Remaining 20% goes into the test set
X_test = X[train_size:]
Y_test = Y[train_size:]
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Convert lists in 'Binarized_Form' into numpy array of type float32
X_train['Binarized_Form'] = X_train['Binarized_Form'].apply(lambda x: np.array([float(i) for i in x], dtype=np.float32))

# Convert lists into a multi-dimensional numpy array and pad sequences to the same length
X_train_padded = pad_sequences(X_train['Binarized_Form'].tolist(), padding='post', dtype='float32')  # use dtype='float32'

# # Add 'Elo Difference w.r.t. Fabi' as additional features
X_train_final = np.hstack((X_train['Normalized Difference'].values.reshape(-1, 1), X_train_padded))

# Convert Y_train to float32
Y_train = Y_train.astype(np.float32)
np.set_printoptions(threshold=np.inf)
print(X_train_final)
def create_model():
    model = Sequential()
    model.add(Dense(units=98, activation='relu', input_dim = 11))
    model.add(Dense(units=32))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
    return model
  model = create_model()
Y_train_encoded = pd.get_dummies(Y_train).values
model.fit(X_train_final, Y_train_encoded, epochs=200, batch_size=16)

# Convert 'Elo Difference w.r.t. Fabi' to float32
X_test['Normalized Difference'] = X_test['Normalized Difference'].astype(np.float32)

# Convert lists in 'Binarized_Form' into numpy array of type float32
X_test['Binarized_Form'] = X_test['Binarized_Form'].apply(lambda x: np.array([float(i) for i in x], dtype=np.float32))

# Convert lists into a multi-dimensional numpy array and pad sequences to the same length
X_test_padded = pad_sequences(X_test['Binarized_Form'].tolist(), padding='post', dtype='float32')  # use dtype='float32'

# Add 'Elo Difference w.r.t. Fabi' as additional features
X_test_final = np.hstack((X_test['Normalized Difference'].values.reshape(-1, 1), X_test_padded))

# Convert Y_train to float32
Y_test = Y_test.astype(np.float32)

result = model.predict(X_test_final)
np.set_printoptions(threshold=np.inf)
print(result)
# Get the indices of the maximum probabilities
predicted_classes = np.argmax(result, axis=1)

# Create a mapping of class indices to labels
class_index_to_label_mapping = {0: 0, 1: 0.5, 2: 1}

# Map the predicted class indices to labels
predicted_labels = [class_index_to_label_mapping[i] for i in predicted_classes]

print(predicted_labels)

Y_test = (Y_test).astype(int) 
predicted_labels = [int(label) for label in predicted_labels]
print('Accuracy: ', accuracy_score(Y_test, predicted_labels))

      
