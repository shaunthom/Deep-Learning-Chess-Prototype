# Deep-Learning-Chess-Prototype
This is a novel approach in accurately predicting winner of chess games. Instead of spawning an entire generation of players, we focus on a single player. We factor in his ELO rating, his opposition as well as the color of the pieces. But most importantly, the form of the player is heavily emphasized as must be in a real-world scenario. This form is known as momentum.

###Objective:

To accurately predict outcome of a chess game based on the player's strength, the quality of his opposition, the color of the pieces as well as his current form.

###Basic Principle:

Momentum can also refer to a team or player's perceived psychological edge in a competition. It's often used to describe a situation where one athlete begins to outperform the other, usually following a particularly successful run of events. There are numerous real-world proofs to corroborate this statement. One of the most famous examples of momentum shift in chess history is the 1985 World Chess Championship between Garry Kasparov and Anatoly Karpov. Karpov initially led the series 5-0 and everyone was predicting a whitewash. But Kasparov managed to stall further victories from Karpov and started winning games, eventually the series was at 5-3. Many analysts attribute Kasparov's comeback to the momentum he built up through his series of draws and victories. 

###About the Dataset:

The dataset was scraped from the website old.chesstempo.com . I utilized BeautifulSoup for this. 

###Working:

The model inputs a specific player's games and uses neural network to learn his form or momentum. It also analyses the quality of the opponent (ELO Rating). The player with white pieces is given a minor advantage by using a bias. It then predicts the winner of the game. 

For the first file named "Fabi.py", we used an American chess player named "Fabiano Caruana". The model studied his games using his ELO as well as his opponent's rating at the time of the game. It also gave a slight edge to whoever was having the white pieces. Lastly, it used the last 10 of his results to quantify his form. The model had 3 layers of architecture and used sigmoid activation function. It attained an accuracy of 72%.

For the second instance, a Dutch super grandmaster "Anish Giri" was the experiment. The model used his strength, the stength of his opposing player as well as Anish's winning or losing form. It got an accuracy of 76%.
