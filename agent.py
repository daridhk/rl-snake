import torch 
import random 
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
from Helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # self.model = Linear_QNet(11,256,3)
        self.model = Linear_QNet(7, 256, 3)
        # self.model = Linear_QNet(12, 256, 3)
        # self.model = Linear_QNet(12, 200, 200, 3)
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)         
        # TODO: model,trainer

    # state (11 Values)
    #[ danger straight, danger right, danger left,
    #   
    # direction left, direction right,
    # direction up, direction down
    # 
    # food left,food right,
    # food up, food down]
    def get_state(self,game):
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

    def get_action(self,state):
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
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            # Train long memory,plot result
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if(score > record): # new High score
                record = score
                game.high_score = score
                agent.model.save()
            print('No:',agent.n_game,'Score:',score,'Record:',record)
            
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            # plot(plot_scores,plot_mean_scores)


if(__name__=="__main__"):
    train()


'''
neural network 가 11*256*3 이다. 200번 돌리면 high score 40, 300번 돌리면 50~60
12*256*3 으로 바꿔본다.  input 하나가 늘어났는데 dummy 0 을 넣었다. 동일한 성능이 나온다.
12*24*24*3 으로 바꿔본다. 
network 이 잘못되었나? 200번 돌려도 score 1 이다. 뺑뺑이만 돈다.
hidden 이 1개 일때 보다 2개일때 더 못한다. 이유가 무엇일까?
12*120*120*3 이면 200번 돌릴 때 40 나온다. network 이 커야 되나 보다.
12*200*200*3 이면
12*2560*3 이면 200번 돌릴 때 50 나온다. 300번 돌리면 54. 12*256*3 과 비슷하다. 성능 개선이 없다.

food 위치를 state 로 넣을 때 == 이 빠져 있었음. 그래서 == 두개를 추가함.
11*256*3 인데 13*256*3 으로 바꿔 테스트 함. 330번 48

state를 relative 로 바꾸면
network 는 7*256*3.
straight wall, right wall, left wall, straight food, straight even food, right food, right even food
200에 50, 

왜 빠져 나오지 못하지.. back을 못하기 때문일까? 생존을 우선으로 바꿔야 하지 않을까?
collision -10, food 10
이었는데 
collision -100, food 10
로 바꾸면
200번 62, 동일 한데 단 갖혔을 때 좀더 오래 산다. 갖히는 것을 막을 수는 없는데 자주 갖힌다.
갖혔을 때 시계방향 또는 반시계 방향으로 돌면서 좁아지기 때문에 무조건 갖히면 죽는데.
갖히더라도 지그재그로 채워나가면 꼬리가 딸려나오기 때문에 빠져나올 수 있는데 그런 것을 못한다.
학습을 하면 할 수록 크게 돈다. 
300번 71 로 최고 점수이긴하다.



neural network input 이 11개, 7개 정도이고 binary 이면 neural net 이 필요없다.
그냥 Table을 학습해도 된다.



state를 absolute 로 바꾸면

convolution 을 사용하면

'''