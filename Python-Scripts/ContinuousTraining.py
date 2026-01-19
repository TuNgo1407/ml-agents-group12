import StartTraining

def main():
    q1 = input("Do you want a set number of runs or to go on indefinitely?\nSet Number: N \nIndefinite: I\n")

    if q1 == "N":
        q2 = input("How many runs?\n")
        n = 1
        while n <= int(q2):
            StartTraining.main()
            n+=1
    elif q1 == "I":
        while True:
            StartTraining.main()

if __name__ == "__main__":
    main()