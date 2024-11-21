from itertools import count

import pygame
import torch
from torch import optim

import config
import base
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_dqn():
    # Pygame 초기화
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (config.SCREEN_SIZE * config.PIXEL_SIZE, config.SCREEN_SIZE * config.PIXEL_SIZE)
    )
    pygame.display.set_caption('Snake Game with DQN')

    # 환경 및 네트워크 초기화
    env = base.SnakeEnv()
    n_actions = env.action_space.n
    policy_net = model.DQN((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), n_actions).to(device)
    target_net = model.DQN((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=model.LR)
    memory = model.ReplayMemory(10000)

    steps_done = 0

    # 에피소드 반복
    for i_episode in range(model.NUM_EPISODES):
        # 환경 초기화
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0  # 총 보상 초기화

        for t in count():
            # Pygame 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # 행동 선택
            action = model.DQN.select_action(state, steps_done, policy_net, n_actions)
            steps_done += 1

            # 환경 단계 진행
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            next_state_tensor = None if done else torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # 리플레이 메모리에 저장
            memory.push(state, action, next_state_tensor, reward)

            # 상태 업데이트
            state = next_state_tensor

            # 모델 최적화
            model.DQN.optimize_model(memory, policy_net, target_net, optimizer)

            # 게임 렌더링
            env.render(screen)

            if done:
                break

        # 타겟 네트워크 소프트 업데이트
        with torch.no_grad():
            for key in policy_net.state_dict():
                target_net.state_dict()[key].data.copy_(
                    model.TAU * policy_net.state_dict()[key].data + (1.0 - model.TAU) * target_net.state_dict()[key].data
                )

        # 에피소드 결과 출력
        print(f"Episode {i_episode + 1}: Total Reward = {total_reward}, Score = {env.score}")

    # Pygame 종료
    pygame.quit()



if __name__ == "__main__":
    main_dqn()
