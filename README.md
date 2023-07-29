# Deep-Learning-Chess-Prototype
This is a novel approach in accurately predicting winner of chess games. Instead of spawning an entire generation of players, we focus on a single player. We factor in his ELO rating, his opposition as well as the color of the pieces. But most importantly, the form of the player is heavily emphasized as must be in a real-world scenario.

Working:

The model inputs a specific player's games and uses neural network to learn his form or momentum. It also analyses the quality of the opponent (ELO Rating). The player with white pieces is given a minor advantage by using a bias. It then predicts the winner of the game. An accuracy of 72% was obtained. 

