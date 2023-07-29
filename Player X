import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

fg = pd.read_csv('')
for index, row in fg.iterrows():
    if row['Res'] == "Jan-00":
        fg.at[index, 'Res'] = "1-0"
win_streak = 0
lose_streak = 0
draw_count = 0
for index, row in ten_games.iterrows():
    if row["White"] == 'Caruana, Fabiano':
        if row["Res"] == '1-0':
            win_streak += 1
            draw_count = 0
        elif row["Res"] == '0-1':
            lose_streak += 1
            draw_count = 0
        else:
            draw_count += 1
            if draw_count >= 10:
                win_streak = 0
                lose_streak = 0
    elif row["Black"] == 'Caruana, Fabiano':
        if row["Res"] == '0-1':
            win_streak += 1
            draw_count = 0
        elif row["Res"] == '1-0':
            lose_streak += 1
            draw_count = 0
        else:
            draw_count += 1
            if draw_count >= 10:
                win_streak = 0
                lose_streak = 0

print(win_streak, lose_streak, sep="\n")

def latest_form(dataset):
    
    form = []
    for index, row in dataset.iterrows():
        
        if row["White"] == 'Caruana, Fabiano':
            if row["Res"] == '1-0':
                form.append("W")
            elif row["Res"] == '0-1':
                form.append("L")
            else:
                form.append("D")
        elif row["Black"] == 'Caruana, Fabiano':
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
        
        if row["White"] == 'Caruana, Fabiano':
            if row["Res"] == '1-0':
                form.append("W")
            elif row["Res"] == '0-1':
                form.append("L")
            else:
                form.append("D")
        elif row["Black"] == 'Caruana, Fabiano':
            if row["Res"] == '0-1':
                form.append("W")
            elif row["Res"] == '1-0':
                form.append("L")
            else:
                form.append("D")
    last_100_values = form[-1:-100:-1]
    return last_100_values

latest_form(fg)

def result_for_caruana(row):
    if row['White'] == 'Caruana, Fabiano':
        return 'W' if row['Res'] == '1-0' else 'L' if row['Res'] == '0-1' else 'D'
    elif row['Black'] == 'Caruana, Fabiano':
        return 'W' if row['Res'] == '0-1' else 'L' if row['Res'] == '1-0' else 'D'
    return None

results_caruana = []
last_10_results = []

for index, row in fg.iterrows():
    result = result_for_caruana(row)
    if result is not None:
        results_caruana.insert(0, result)
    last_10_results.append(list(results_caruana[1:11]))  # copy the current state of last 10 results

fg['Last_10_Results'] = last_10_results

fg['Elo Difference w.r.t. Fabi'] = ""
for index, row in fg.iterrows():
    if row["White"] == "Caruana, Fabiano":
        x = float(row["W ELO"]) - float(row["B ELO"])
        fg.at[index, "Elo Difference w.r.t. Fabi"] = x
    elif row["Black"] == "Caruana, Fabiano":
        x = float(row["B ELO"]) - float(row["W ELO"])
        fg.at[index, "Elo Difference w.r.t. Fabi"] = x

fg_w['Num_Result'] = ""
for index, row in fg_w.iterrows():
    
    if row["White"] == "Caruana, Fabiano":
        if row["Res"] == '1-0':
            fg_w.at[index, "Num_Result"] = 1
        elif row["Res"] == '0-1':
            fg_w.at[index, "Num_Result"] = 0
        else:
            fg_w.at[index, "Num_Result"] = 0.5
    
    if row["Black"] == "Caruana, Fabiano":
        if row["Res"] == '0-1':
            fg_w.at[index, "Num_Result"] = 1
        elif row["Res"] == '1-0':
            fg_w.at[index, "Num_Result"] = 0
        else:
            fg_w.at[index, "Num_Result"] = 0.5

FG["Binarized_Form"] = None
for index, row in FG.iterrows():
    k = [1 if x == "W" else 0 if x == "L" else 0.5 for x in row["Last_10_Results"]]
    FG.at[index, "Binarized_Form"] = k

FG['Binarized_Form'] = FG['Binarized_Form'].apply(lambda x: [float(val) for val in x])

input_col = cols.drop(cols.index[:10])
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(input_col['Elo Difference w.r.t. Fabi'].values.reshape(-1, 1))

input_col['Normalized Difference'] = normalized_values
output_col = FG["Num_Result"].drop(FG["Num_Result"].index[:10])
X = input_col
Y = output_col
train_size = int(990 * 0.8)
X_train = X[:train_size]
Y_train = Y[:train_size]

# Remaining goes into the test set
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
def model():
    model = Sequential()
    model.add(Dense(units=98, activation='relu', input_dim = 11))
    model.add(Dense(units=32))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
    return model
model = model()
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
