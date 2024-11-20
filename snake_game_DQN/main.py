import pygame

import config
import base
import DQN

def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((config.SCREEN_SIZE * config.PIXEL_SIZE, config.SCREEN_SIZE * config.PIXEL_SIZE))
    pygame.display.set_caption('Snake Game with DQN')

    env = base.SnakeEnv()
    state_shape = (config.SCREEN_SIZE, config.SCREEN_SIZE, 3)
    num_actions = 4
    agent = DQN.DQNAgent(state_shape, num_actions)

    num_episodes = 500
    target_update_freq = 10
    clock = pygame.time.Clock()  # 게임 속도 제어를 위한 Clock 객체

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 이벤트 처리 (게임 종료를 위한 기본 처리)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # DQN 행동 선택 및 환경 업데이트
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # DQN 학습
            agent.replay()

            # 화면 렌더링
            env.render(screen)

            # 화면 업데이트 및 게임 속도 제어
            pygame.display.update()
            clock.tick(config.FPS)

        # 타겟 네트워크 업데이트
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # 탐험률 감소
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    pygame.quit()


if __name__ == "__main__":
    main()
