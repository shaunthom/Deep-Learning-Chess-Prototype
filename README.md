# Deep-Learning-Chess-Prototype
This is a novel approach in accurately predicting winner of chess games. Instead of spawning an entire generation of players, we focus on a single player. We factor in his ELO rating, his opposition as well as the color of the pieces. But most importantly, the form of the player is heavily emphasized as must be in a real-world scenario. This form is known as momentum.

Basic Principle:

Momentum can also refer to a team or player's perceived psychological edge in a competition. It's often used to describe a situation where one athlete begins to outperform the other, usually following a particularly successful run of events. There are numerous real-world proofs to corroborate this statement. One of the most famous examples of momentum shift in chess history is the 1985 World Chess Championship between Garry Kasparov and Anatoly Karpov. Karpov initially led the series 5-0 and everyone was predicting a whitewash. But Kasparov managed to stall further victories from Karpov and started winning games, eventually the series was at 5-3. Many analysts attribute Kasparov's comeback to the momentum he built up through his series of draws and victories. 

Working:

The model inputs a specific player's games and uses neural network to learn his form or momentum. It also analyses the quality of the opponent (ELO Rating). The player with white pieces is given a minor advantage by using a bias. It then predicts the winner of the game. For the first file, 
