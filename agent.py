import torch 
import random 
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer, Conv_QNet
from Helper import plot
MAX_MEMORY = 100_000
#BATCH_SIZE = 1000
BATCH_SIZE = 100
LR = 0.001
MAX_CHANNEL = 5

do_conv = True
do_train_only_success = False
class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.channel = deque(maxlen=MAX_CHANNEL)
        self.is_init_channel = False
        if do_conv:
            self.model = Conv_QNet(5*5*12, 24, 4)
        else:
            self.model = Linear_QNet(7, 256, 3)
        # self.model = Linear_QNet(11,256,3)

        # self.model = Linear_QNet(12, 256, 3)
        # self.model = Linear_QNet(12, 200, 200, 3)
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)         
        # TODO: model,trainer

    def get_state(self,game):
        # return self.get_state_conv(game)
        if do_conv:
            return self.get_state_conv(game)
        else:
            return self.get_state_7_vector(game)

    def get_state_conv(self,game):
        def pos_to_int(pos):
            return int(pos.x / BLOCK_SIZE) + 1, int(pos.y / BLOCK_SIZE) + 1

        # return 3x22x22 tensor
        # state = torch.zeros(3, 22, 22)
        state = np.zeros((22, 22), dtype=float)
        state_wall = 1.0
        state_body = 0.75
        state_head = 0.5
        state_food = 0.25

        # wall
        for i in range(22):
            state[0][i] = state_wall
            state[21][i] = state_wall
            state[i][0] = state_wall
            state[i][21] = state_wall

        # snake body
        for i, body in enumerate(game.snake):
            x, y = pos_to_int(body)
            if i == 0:
                state[y][x] = state_head
            else:
                state[y][x] = state_body

        # food
        x, y = pos_to_int(game.food)
        state[y][x] = state_food

        if not self.is_init_channel:
            for i in range(MAX_CHANNEL-1):
                self.channel.append(state)
            self.is_init_channel = True

        self.channel.append(state)
        states = [self.channel.popleft()]
        for state in self.channel:
            states = np.vstack([states, [state]])
        return states

    def get_state_7_vector(self,game):
        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #Danger Left
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),

            #Food Straight
            (dir_u and game.food.y < game.head.y) or
            (dir_d and game.food.y > game.head.y) or
            (dir_l and game.food.x < game.head.x) or
            (dir_r and game.food.x > game.head.x),

            # Food Straight even
            (dir_u and game.food.y == game.head.y) or
            (dir_d and game.food.y == game.head.y) or
            (dir_l and game.food.x == game.head.x) or
            (dir_r and game.food.x == game.head.x),

            #Food Right
            (dir_u and game.food.x > game.head.x) or
            (dir_d and game.food.x < game.head.x) or
            (dir_l and game.food.y > game.head.y) or
            (dir_r and game.food.y < game.head.y),

            # Food Right even
            (dir_u and game.food.x == game.head.x) or
            (dir_d and game.food.x == game.head.x) or
            (dir_l and game.food.y == game.head.y) or
            (dir_r and game.food.y == game.head.y)

        ]
        return np.array(state,dtype=int)

    def get_state_original(self,game):
        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #Danger Left
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),

            # dummy
            # 0,

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x == game.head.x,
            game.food.y == game.head.y,
            game.food.x < game.head.x, # food is in left
            game.food.x > game.head.x, # food is in right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y  # food is down
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action_orig(self,state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0,0,0]
        if(random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            # state0 = torch.tensor(state,dtype=torch.float).cuda()
            # prediction = self.model(state0).cuda() # prediction by model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # prediction by model

            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

    def get_legal_move(self, game):
        x = game.snake[0].x
        y = game.snake[0].y
        legal_move = []
        i=0
        x1 = x+BLOCK_SIZE
        y1 = y
        if not game.is_collision(Point(x1,y1)):
            legal_move.append(i)

        i+=1
        x1 = x
        y1 = y+BLOCK_SIZE
        if not game.is_collision(Point(x1,y1)):
            legal_move.append(i)

        i+=1
        x1 = x-BLOCK_SIZE
        y1 = y
        if not game.is_collision(Point(x1,y1)):
            legal_move.append(i)

        i+=1
        x1 = x
        y1 = y-BLOCK_SIZE
        if not game.is_collision(Point(x1,y1)):
            legal_move.append(i)

        return legal_move

    def get_action(self,state, game):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 8000 - self.n_game
        # self.epsilon = 100
        final_move = [0,0,0,0]
        if self.n_game<1000 or random.randint(0,20000)<self.epsilon:
        # if total_score < 10:
            legal_move = self.get_legal_move(game)
            if len(legal_move)<1:
                final_move[0] = 1
                # print("move blocked ************")
            else:
                move = random.randint(0,len(legal_move)-1)
                final_move[legal_move[move]]=1
                # print("move "+str(legal_move[move]))
        else:
            # state0 = torch.tensor(state,dtype=torch.float).cuda()
            # prediction = self.model(state0).cuda() # prediction by model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # prediction by model

            move = torch.argmax(prediction).item()
            final_move[move]=1
        return final_move

    def get_action_orig(self, state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if (random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # state0 = torch.tensor(state,dtype=torch.float).cuda()
            # prediction = self.model(state0).cuda() # prediction by model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # prediction by model

            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


total_score = 0
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    agent.model.load()
    game = SnakeGameAI()
    while True:
        # Get Old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, game)

        # perform move and get new state
        reward, done, score, loop = game.play_step(final_move)
        state_new = agent.get_state(game)

        if not do_train_only_success:
            # train short memory
            agent.train_short_memory(state_old,final_move,reward,state_new,done)

            #remember
            agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            if do_train_only_success:
                agent.remember(state_old, final_move, reward, state_new, done)
            # Train long memory,plot result
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if agent.n_game%100 == 0:
                agent.model.save()
            if(score > record): # new High score
                record = score
                game.high_score = score
                agent.model.save()
            if loop:
                print('No:', agent.n_game, 'Score:', score, 'Record:', record, ' loop')
            else:
                print('No:',agent.n_game,'Score:',score,'Record:',record)
            
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            # plot(plot_scores,plot_mean_scores)


if(__name__=="__main__"):
    train()


'''
neural network ??? 11*256*3 ??????. 200??? ????????? high score 40, 300??? ????????? 50~60
12*256*3 ?????? ????????????.  input ????????? ??????????????? dummy 0 ??? ?????????. ????????? ????????? ?????????.
12*24*24*3 ?????? ????????????. 
network ??? ???????????????? 200??? ????????? score 1 ??????. ???????????? ??????.
hidden ??? 1??? ?????? ?????? 2????????? ??? ?????????. ????????? ?????????????
12*120*120*3 ?????? 200??? ?????? ??? 40 ?????????. network ??? ?????? ?????? ??????.
12*200*200*3 ??????
12*2560*3 ?????? 200??? ?????? ??? 50 ?????????. 300??? ????????? 54. 12*256*3 ??? ????????????. ?????? ????????? ??????.

food ????????? state ??? ?????? ??? == ??? ?????? ?????????. ????????? == ????????? ?????????.
11*256*3 ?????? 13*256*3 ?????? ?????? ????????? ???. 330??? 48

state??? relative ??? ?????????
network ??? 7*256*3.
straight wall, right wall, left wall, straight food, straight even food, right food, right even food
200??? 50, 

??? ?????? ????????? ?????????.. back??? ????????? ????????????? ????????? ???????????? ????????? ?????? ??????????
collision -10, food 10
???????????? 
collision -100, food 10
??? ?????????
200??? 62, ?????? ?????? ??? ????????? ??? ?????? ?????? ??????. ????????? ?????? ?????? ?????? ????????? ?????? ?????????.
????????? ??? ???????????? ?????? ????????? ???????????? ????????? ???????????? ????????? ????????? ????????? ?????????.
??????????????? ??????????????? ??????????????? ????????? ??????????????? ????????? ???????????? ??? ????????? ?????? ?????? ?????????.
????????? ?????? ??? ?????? ?????? ??????. 
300??? 71 ??? ?????? ??????????????????.



neural network input ??? 11???, 7??? ???????????? binary ?????? neural net ??? ????????????.
?????? Table??? ???????????? ??????.



state??? absolute ??? ?????????

convolution ??? ????????????

'''