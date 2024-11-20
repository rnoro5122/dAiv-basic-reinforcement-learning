import pygame
import random
import numpy as np
import config
import time

# 방향 정의
DIRECTIONS = np.array([
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (0, 1),   # DOWN
    (-1, 0)   # LEFT
])

class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        center_x, center_y = config.SCREEN_SIZE // 2, config.SCREEN_SIZE // 2
        self.snake = np.array([
            [center_x, center_y],
            [center_x, center_y + 1],
            [center_x, center_y + 2],
            [center_x, center_y + 3]
        ])
        self.direction = 0
        self.score = 0
        self.done = False
        self.last_move_time = time.time()
        self.place_fruit()
        return self.get_observation()

    def place_fruit(self):
        while True:
            x = random.randint(0, config.SCREEN_SIZE - 1)
            y = random.randint(0, config.SCREEN_SIZE - 1)
            if list([x, y]) not in self.snake.tolist():
                break
        self.fruit = np.array([x, y])

    def get_observation(self):
        obs = np.zeros((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), dtype=np.uint8)
        for bit in self.snake:
            obs[bit[1], bit[0], 0] = 1  # 뱀
        obs[self.fruit[1], self.fruit[0], 1] = 1  # 과일
        return obs

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, self.done

        self.direction = action
        old_head = self.snake[0]
        movement = DIRECTIONS[self.direction]
        new_head = old_head + movement

        # 충돌 검사
        if (new_head[0] < 0 or new_head[0] >= config.SCREEN_SIZE or
                new_head[1] < 0 or new_head[1] >= config.SCREEN_SIZE or
                new_head.tolist() in self.snake.tolist()):
            self.done = True
            return self.get_observation(), -10, self.done

        # 과일을 먹으면 점수 증가
        reward = 1
        if all(new_head == self.fruit):
            self.score += 1
            self.place_fruit()
            reward = 10
        else:
            self.snake = self.snake[:-1, :]  # 꼬리 삭제

        # 새로운 머리 추가
        self.snake = np.concatenate([[new_head], self.snake], axis=0)
        return self.get_observation(), reward, self.done

    def render(self, screen):
        screen.fill((0, 0, 0))

        # 테두리 그리기
        pygame.draw.rect(screen, (255, 255, 255), [0, 0, config.SCREEN_SIZE * config.PIXEL_SIZE, config.LINE_WIDTH])
        pygame.draw.rect(screen, (255, 255, 255), [0, config.SCREEN_SIZE * config.PIXEL_SIZE - config.LINE_WIDTH, config.SCREEN_SIZE * config.PIXEL_SIZE, config.LINE_WIDTH])
        pygame.draw.rect(screen, (255, 255, 255), [0, 0, config.LINE_WIDTH, config.SCREEN_SIZE * config.PIXEL_SIZE])
        pygame.draw.rect(screen, (255, 255, 255), [config.SCREEN_SIZE * config.PIXEL_SIZE - config.LINE_WIDTH, 0, config.LINE_WIDTH, config.SCREEN_SIZE * config.PIXEL_SIZE])

        # 뱀과 과일 그리기
        for bit in self.snake:
            pygame.draw.rect(screen, (0, 255, 0),
                             (bit[0] * config.PIXEL_SIZE, bit[1] * config.PIXEL_SIZE, config.PIXEL_SIZE, config.PIXEL_SIZE))
        pygame.draw.rect(screen, (255, 0, 0),
                         (self.fruit[0] * config.PIXEL_SIZE, self.fruit[1] * config.PIXEL_SIZE, config.PIXEL_SIZE, config.PIXEL_SIZE))


    # 점수 표시
        font = pygame.font.SysFont(None, 20)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (5, 5))

        pygame.display.flip()