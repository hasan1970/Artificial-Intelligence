from blackjack import train, play

print("Training")
ai=train(10000)

while True:
    play(ai)

    print()
    exit=input("Do you want to play again? (y/n) ")

    if exit=='n':
        break
