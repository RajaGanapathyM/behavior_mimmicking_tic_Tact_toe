
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf


# In[2]:

from IPython.display import clear_output


# In[3]:

file_name="ses.ckpt"
file_path="./model_files/"
model_file_path=file_path+file_name


# In[4]:

combinations=[
    [(0,0),(0,1),(0,2)],[(1,0),(1,1),(1,2)],[(2,0),(2,1),(2,2)],
    [(0,0),(1,0),(2,0)],[(0,1),(1,1),(2,1)],[(0,2),(1,2),(2,2)],
    [(0,0),(1,1),(2,2)],[(0,2),(1,1),(2,0)]
]


# In[5]:

def get_state(playfield):
    dominance=0
    for each_comb in combinations:
        advantage=0
        for each in each_comb:
            advantage+=playfield[each[0],each[1]]
            if abs(advantage)==3:
                return advantage
    return 0


# In[6]:

def get_dominance(playfield):
    dominance=0
    for each_comb in combinations:
        advantage=0
        for each in each_comb:
            advantage+=playfield[each[0],each[1]]
        if advantage==2:
            dominance+=1
    return dominance


# In[7]:

t_1= tf.placeholder(tf.float32, (3,3))
t= tf.placeholder(tf.float32, (3,3))
bias_1=tf.Variable(tf.truncated_normal([20], dtype=tf.float32))
bias_2=tf.Variable(tf.truncated_normal([1], dtype=tf.float32))
t_playfield = tf.convert_to_tensor(t, np.float32)
t_playfield_previous = tf.convert_to_tensor(t_1, np.float32)

t_playfield=tf.reshape(t_playfield,[1,3,3,1])
t_playfield_previous=tf.reshape(t_playfield_previous,[1,3,3,1])


# In[8]:

hidden_layer_1=tf.nn.bias_add(tf.layers.conv2d(t_playfield_previous,20,2,strides=(1,1),padding="SAME"),bias_1)
hidden_layer=tf.tanh(tf.nn.bias_add(tf.layers.conv2d(hidden_layer_1,1,2,strides=(1,1)),bias_2))
predicted=tf.layers.conv2d_transpose(hidden_layer,1,2,strides=(1,1))


# In[9]:

loss=tf.losses.mean_squared_error(t_playfield,predicted)
#train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)


# In[ ]:

init = tf.global_variables_initializer()
playfield=np.zeros((3,3))
playfield_previous=np.zeros((3,3))
saver = tf.train.Saver()
flak=0
sess=tf.Session()
sess.run(init)
saver.restore(sess, model_file_path)
def make_train(train_mode,playfield_previous,playfield,turn):
    global sess,saver
    pred=""
    if train_mode==1:
        _, ls,pred = sess.run([train_op, loss,predicted], feed_dict={t_1: -1*turn*playfield_previous,t:-1*turn*playfield})
    elif train_mode==0:
        pred = sess.run([predicted], feed_dict={t_1: -1*turn*playfield_previous,t:-1*turn*playfield})
    save_path = saver.save(sess, model_file_path)
    return pred

turn=1
iter=0

buttons=[]
count=0
def update_btn(ind):
    global playfield_old,playfield_previous,playfield,iter,buttons,flak
    turn=1
    for k in range(0,3):
        try:
            x=0
            y=0
            iter+=1
            playfield_old=playfield_previous.copy()
            playfield_previous=playfield.copy()
            if turn==1:
                #print("Raja TUrn")
                turn=-1
            else:
                if turn==-2:
                    turn=-1
                    ctypes.windll.user32.MessageBoxW(0, "I WON last game.So I play First move...", "Ai.apply/TIC TAC TOE", 1)
                    print("AI Initiated new Game...Your Turn")
                    flak=1
                #print("Opp Turn")
                turn=1
            if turn==-1:
                x=int(float(ind)/3.0)
                y=int(ind)%3
                nzx,nzy=(playfield==0).nonzero()
                nxy=zip(nzx,nzy)
                if (x,y) not in nxy:
                    print("Position already Marked.Enter Correct Position")
                    iter-=1
                    turn=1
                    playfield_previous=playfield_old.copy()
                    break
                playfield[x,y]=1
                
                buttons[ind].config(text="X")
                pred = make_train(1,playfield_previous,playfield,turn)
                pred=np.reshape(pred,[3,3])
                #print("LOSS",ls)
            else:
                pred = make_train(0,playfield_previous,playfield,turn)
                pred=np.reshape(pred,[3,3])
                nzx,nzy=(playfield==0).nonzero()
                best=-1000
                g2=0
                si=-5
                sj=-5
                for i,j in zip(nzx,nzy):
                    if pred[i,j]>best:
                        si=i
                        sj=j
                        g2=1
                        best=pred[i,j]
                if g2==0:
                    playfield[nzx[0],nzy[0]]=-1
                    buttons[(3*nzx[0])+nzy[0]].config(text="O")
                else:
                    playfield[si,sj]=-1                
                    buttons[(3*si)+sj].config(text="O")
                    
            if flak==0:
                clear_output()
            else:
                flak=0
            #print("Previou State")
            #print(playfield_previous)
            print("Current State")
            print(playfield)
            print("Predicted")
            print(pred)
            #print("Predicted_current_State")
            #print(-1*turn*np.reshape(pred,[3,3]))
            #print("Loss",ls)        
            f=get_state(playfield)
            if iter==9 or len(np.nonzero(playfield))==0 or f==-3 or f==3:  
                turn=1
                if f==3:
                    playfield=playfield_old.copy()
                    playfield_previous=playfield_old.copy()
                    playfield[x,y]=-1                
                    buttons[ind].config(text="X")
                    #print("Retro training...")
                    #print(-1*turn*playfield_previous)
                    #print(-1*turn*playfield)
                    pred = make_train(1,playfield_previous,playfield,turn)  
                    pred=np.reshape(pred,[3,3])
                    ctypes.windll.user32.MessageBoxW(0, "You WON", "Ai.apply/TIC TAC TOE", 1)
                    #print("LOSS",ls)
                    turn=1
                elif f==-3:
                    #buttons[ind].config(text="O")                    
                    pred = make_train(1,playfield_previous,playfield,turn)
                    ctypes.windll.user32.MessageBoxW(0, "I WON...", "Ai.apply/TIC TAC TOE", 1)
                    turn=-2                

                ctypes.windll.user32.MessageBoxW(0, "New Game Started...", "Ai.apply/TIC TAC TOE", 1)
                playfield=np.zeros((3,3))
                playfield_previous=np.zeros((3,3))
                for each_btn in buttons:
                    each_btn.config(text="-")
                iter=0
                if f!=-3:
                    break
        except Exception as e:
            print(str(e))
            ctypes.windll.user32.MessageBoxW(0, "Some error happened. Restarting Game..".format(str(e)), "Ai.apply/TIC TAC TOE", 1)
            playfield=np.zeros((3,3))
            playfield_previous=np.zeros((3,3))
            for each_btn in buttons:
                each_btn.config(text="-")
            turn=1
            iter=0


# In[ ]:

def start_game():
    helv36 =  font.Font(family='Helvetica', size=20, weight='bold')
    buttons.append(Button(text="-",fg="red",font=helv36, height = 5, width = 10,command=lambda: update_btn(0)))
    buttons[0].grid(row=0,column=0)
    buttons[0].config()
    buttons.append(Button(text="-",fg="red",font=helv36, height = 5, width = 10,command=lambda: update_btn(1)))
    buttons[1].grid(row=0,column=1)
    buttons[1].config()
    buttons.append(Button(text="-",fg="red",font=helv36, height = 5, width = 10,command=lambda: update_btn(2)))
    buttons[2].grid(row=0,column=2)
    buttons[2].config()



    buttons.append(Button(text="-",fg="red", font=helv36,height = 5, width = 10,command=lambda: update_btn(3)))
    buttons[3].grid(row=1,column=0)
    buttons[3].config()
    buttons.append(Button(text="-",fg="red", font=helv36,height = 5, width = 10,command=lambda: update_btn(4)))
    buttons[4].grid(row=1,column=1)
    buttons[4].config()
    buttons.append(Button(text="-",fg="red", font=helv36,height = 5, width = 10,command=lambda: update_btn(5)))
    buttons[5].grid(row=1,column=2)
    buttons[5].config()

    buttons.append(Button(text="-",fg="red", font=helv36,height = 5, width = 10,command=lambda: update_btn(6)))
    buttons[6].grid(row=2,column=0)
    buttons[6].config()
    buttons.append(Button(text="-",fg="red", font=helv36,height = 5, width = 10,command=lambda: update_btn(7)))
    buttons[7].grid(row=2,column=1)
    buttons[7].config()
    buttons.append(Button(text="-",fg="red", font=helv36,height = 5, width = 10,command=lambda: update_btn(8)))
    buttons[8].grid(row=2,column=2)
    buttons[8].config()

    mainloop()


# In[ ]:

from tkinter import *
from tkinter import font
import ctypes
root = Tk()
root.title("aI.apply/Tic Tac Toe")
start_game()


# In[ ]:

save_path = saver.save(sess, model_file_path)

