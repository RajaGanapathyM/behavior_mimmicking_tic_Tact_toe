# behavior_mimmicking_tic_Tact_toe
A simple tic tact toe game backed by neural netwrok which mich mimmicks user style of playing


Dependencies:
Python-3
tensorflow
Tkinter library

Edit file path for model files before running

Neural network became a key part in the advancement of artificial intelligence towards perfection. With good architecture and have large dataset, resent discoveries challenged humans in normal day to days over perfection. By going through the human defined dataset,rules and objective, these computational network are able to understand the behavioural action needed to be taken over the given situation.

But can we make this network learn on their own just as the way humans do by observing the other human/subject with help of some guidelines and mimicking the activites so that given the same situtation,the network will react in the same way as the subject being observed and desired results achieved.

A simple experiment is done with the above objective. A tic tac toe game environment is created using python with a small simple convolution neural network as opponent to learn the game. The neural network structure is created using tensorflow as below,

--------------------------------------------
t_1= tf.placeholder(tf.float32, (3,3))
t= tf.placeholder(tf.float32, (3,3))
bias_1=tf.Variable(tf.truncated_normal([20], dtype=tf.float32))
bias_2=tf.Variable(tf.truncated_normal([1], dtype=tf.float32))
t_playfield = tf.convert_to_tensor(t, np.float32)
t_playfield_previous = tf.convert_to_tensor(t_1, np.float32)

t_playfield=tf.reshape(t_playfield,[1,3,3,1])
t_playfield_previous=tf.reshape(t_playfield_previous,[1,3,3,1])

hidden_layer_1=tf.nn.bias_add(tf.layers.conv2d(t_playfield_previous,20,2,strides=(1,1),padding="SAME"),bias_1)
hidden_layer=tf.tanh(tf.nn.bias_add(tf.layers.conv2d(hidden_layer_1,1,2,strides=(1,1)),bias_2))
predicted=tf.layers.conv2d_transpose(hidden_layer,1,2,strides=(1,1))

loss=tf.losses.mean_squared_error(t_playfield,predicted)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

--------------------------------------------------------------------------------

The above netwrok is trained in real time while playing with the following guidelines,

1.Watch the human player input and train to replicate the same move as neural network's move
2.If human player won a match, train by taking human player's last move as your next move at t-1 which will block the human player winning move .
3.If the network make a move and wins, retrain on the same move again, so that it can learn the move better.
So with the above settings, I started playing the game with the AI, And after few matches It performed well in reacting to some move in the same way I did. Though it's not learning to win, But just reacting in the same way for given situtation which is not exact as the old one but similar will evenutally gives desired result. Some of the moves is recorded  which is attached as video below,

https://youtu.be/2wnVLrjBDh8
