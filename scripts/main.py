import game


def main():
    env = game.Game(["../images/team1.png", "../images/team2.png"], ["../images/flag1.png", "../images/flag2.png"])

    for _ in range(3): # Num episodes
        t1_pos, t2_pos = env.reset()
        done = False

        t1_score, t2_score = 0, 0

        while not done:
            done, t1_pos, t2_pos, t1_act, t2_act, t1_r, t2_r = env.step(env.team1.sample_action(), env.team2.sample_action())
            print(t1_pos, t2_pos, t1_r, t2_r, t1_act, t2_act)
            t1_score += t1_r
            t2_score += t2_r
            env.render()

        # Given rollout of states, actions, value estimates, rewards

if __name__ == "__main__":
    main()