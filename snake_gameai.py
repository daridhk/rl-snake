import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
pygame.init()
#font = pygame.font.Font('freesansbold.ttf',18)
font = pygame.font.SysFont('arial', 18)

# Reset 
# Reward
# Play(action) -> Direction
# Game_Iteration
# is_collision
REWARD_COLLISION = -11
REWARD_FOOD = 10

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Point = namedtuple('Point','x , y')

BLOCK_SIZE=20
SPEED = 40
# SPEED = 0
WHITE = (255,255,255)
GRAY = (127, 127, 127)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)
YELLOW1 = (255,255,0)
YELLOW2 = (200,200,0)
GREEN1 = (0x3c, 0xb0, 0x43)
GREEN2 = (0x23, 0x4f, 0x1e)

class SnakeGameAI:
    def __init__(self,w=400,h=400):
    #    def __init__(self, w=640, h=480):

        self.w=w
        self.h=h
        #init display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('DQN Snake Game')
        icon_image = pygame.image.load('snake.png')
        pygame.display.set_icon(icon_image)

        self.food_image = pygame.image.load('food.jpg')
        self.food_image = pygame.transform.scale(self.food_image, [BLOCK_SIZE, BLOCK_SIZE])
        self.food_rect = self.food_image.get_rect()

        self.clock = pygame.time.Clock()
        
        self.high_score = 0
        #init game state
        self.reset()
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE,self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]
        self.score = 0
        self.food = None
        self._place__food()
        self.frame_iteration = 0
      

    def _place__food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if(self.food in self.snake):
            self._place__food()


    def play_step(self,action):
        self.frame_iteration+=1
        # 1. Collect the user input
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
            
        # 2. Move
        self._move(action)
        self.snake.insert(0,self.head)

        # 3. Check if game Over
        reward = 0  # eat food: +10 , game over: -10 , else: 0
        game_over = False
        loop = False
        if(self.is_collision() or self.frame_iteration > 1000*len(self.snake) ):
            if self.frame_iteration > 1000 * len(self.snake):
                loop = True
            game_over = True
            reward = REWARD_COLLISION
            return reward,game_over,self.score, loop
        # 4. Place new Food or just move
        if(self.head == self.food):
            self.score+=1
            reward= REWARD_FOOD
            self._place__food()
            
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        # todo
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game Over and Display Score
        
        return reward,game_over,self.score, loop

    def _update_ui(self):
        self.display.fill(WHITE)

        text = font.render("Score: " + str(self.score), True, GRAY)
        self.display.blit(text,[0,19*BLOCK_SIZE])
        text = font.render("High Score: " + str(self.high_score), True, GRAY)
        self.display.blit(text,[100,19*BLOCK_SIZE])

        radius = BLOCK_SIZE/2
        for idx, pt in enumerate(self.snake):
            if idx == 0:
                # pygame.draw.rect(self.display, YELLOW1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                # pygame.draw.rect(self.display, YELLOW2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
                pygame.draw.circle(self.display, GREEN2, [pt.x + radius, pt.y + radius], radius)
            else:
                # pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                # pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
                pygame.draw.circle(self.display, GREEN1, [pt.x+radius, pt.y+radius], radius)

        # pygame.draw.rect(self.display,RED,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        self.food_rect.left = self.food.x
        self.food_rect.top = self.food.y
        self.display.blit(self.food_image, self.food_rect)

        pygame.display.flip()

    def _move(self,action):
        # Action
        # [1,0,0 0] -> right
        # [0,1,0 0] -> down
        # [0,0,1 0] -> left
        # [0 0 0 1] -> up

        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clock_wise.index(self.direction)
        next_idx = np.argmax(action)
        new_dir = clock_wise[next_idx]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if(self.direction == Direction.RIGHT):
            x+=BLOCK_SIZE
        elif(self.direction == Direction.LEFT):
            x-=BLOCK_SIZE
        elif(self.direction == Direction.DOWN):
            y+=BLOCK_SIZE
        elif(self.direction == Direction.UP):
            y-=BLOCK_SIZE
        self.head = Point(x,y)
    def _move_orig(self,action):
        # Action
        # [1,0,0] -> Straight
        # [0,1,0] -> Right Turn
        # [0,0,1] -> Left Turn

        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if(self.direction == Direction.RIGHT):
            x+=BLOCK_SIZE
        elif(self.direction == Direction.LEFT):
            x-=BLOCK_SIZE
        elif(self.direction == Direction.DOWN):
            y+=BLOCK_SIZE
        elif(self.direction == Direction.UP):
            y-=BLOCK_SIZE
        self.head = Point(x,y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hit boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
