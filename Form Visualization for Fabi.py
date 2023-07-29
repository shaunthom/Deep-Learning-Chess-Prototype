fg = pd.read_csv("~/Desktop/Datasets/Chess_Datasets/Fabi_Games.csv", encoding='cp1252')
for index, row in fg.iterrows():
    if row['Res'] == "Jan-00":
        fg.at[index, 'Res'] = "1-0"

fg.head(50)
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
x_values_latest = range(len(latest_form(fg)))

plt.plot(x_values_latest, latest_form(fg))

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Last 10 Values')
plt.grid(True)

plt.show()
import matplotlib.pyplot as plt
import numpy as np

form = present_form(fg)

category_order = ['L', 'D', 'W']

numerical_values = np.array([category_order.index(category) for category in form])

x_values_latest = range(len(form))

plt.plot(x_values_latest, numerical_values)

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Form of the past 100 games')
plt.grid(True)
plt.yticks(range(len(category_order)), category_order)

plt.show()
