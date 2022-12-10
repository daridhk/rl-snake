import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


BOARD_SIZE = 22
CHANNEL1 = 5
KERNEL1 = 3
CHANNEL2 = 16
KERNEL2 = 5
CHANNEL3 = 32

class Conv_QNet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNEL1, CHANNEL2, KERNEL1)
        self.conv2 = nn.Conv2d(CHANNEL2, CHANNEL3, KERNEL2)

        calc_input_size = int(((BOARD_SIZE-(KERNEL1-1))/2 - (KERNEL2-1))/2)
        calc_input_size = calc_input_size*calc_input_size*CHANNEL3
        # calc_input_size =
        self.linear1 = nn.Linear(calc_input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        if len(x.shape)>3:
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x, 0)

        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = '.'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        # todo
        # return
        model_folder_path = '.'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.linear1 = nn.Linear(input_size, hidden_size).cuda()
        # self.linear2 = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    def save(self, file_name='model.pth'):
        model_folder_path = '.'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        # todo
        return
        model_folder_path = '.'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))

class Linear2_QNet(nn.Module):
    def __init__(self,input_size,hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        # self.linear1 = nn.Linear(input_size, hidden_size).cuda()
        # self.linear2 = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    def save(self, file_name='model.pth'):
        model_folder_path = '.'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        # return
        model_folder_path = '.'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimer = optim.Adam(model.parameters(),lr = self.lr)    
        self.criterion = nn.MSELoss()
        for i in self.model.parameters():
            print(i.is_cuda)

    
    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)

        if(len(action.shape) == 1): # only one parameter to train , Hence convert to tuple of shape (1, x)
            #(1 , x)
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )

        # 1. Predicted Q value with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            i = torch.argmax(action).item()
            target[idx][torch.argmax(action).item()] = Q_new
        # 2. Q_new = reward + gamma * max(next_predicted Qvalue) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimer.step()

    def train_step_cuda(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.long).cuda()
        reward = torch.tensor(reward, dtype=torch.float).cuda()

        if (len(state.shape) == 1):  # only one parameter to train , Hence convert to tuple of shape (1, x)
            # (1 , x)
            state = torch.unsqueeze(state, 0).cuda()
            next_state = torch.unsqueeze(next_state, 0).cuda()
            action = torch.unsqueeze(action, 0).cuda()
            reward = torch.unsqueeze(reward, 0).cuda()
            done = (done,)

        # 1. Predicted Q value with current state
        pred = self.model(state).cuda()
        target = pred.clone().cuda()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).cuda()
            target[idx][torch.argmax(action).item()] = Q_new
            # 2. Q_new = reward + gamma * max(next_predicted Qvalue) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimer.step()



