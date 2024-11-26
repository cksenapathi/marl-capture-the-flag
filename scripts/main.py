import game


def main():
    ctf = game.Game(["../images/team1.png", "../images/team2.png"], ["../images/flag1.png", "../images/flag2.png"])
    state = ctf.reset()

    while True:
        ctf.render()

if __name__ == "__main__":
    main()