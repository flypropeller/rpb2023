# team 9
# add

def main():
    print("Let's implement addition. Type two numbers for x and y.")

    x = int(input("x > "))
    y = int(input("y > "))

    print("%d + %d = %d" % (x, y, add(x, y)))


def add(x,y):
    return x+y

if __name__ == "__main__":
    main()


# divide

def main():
    print("Let's implement division. Type two numbers for x and y")

    x = int(input("x > "))
    y = int(input("y > "))

    print("%d / %d = %0.3f" % (x, y, divide(x, y)))


def add(a, b):
    return a + b


def divide(x,y):
    if y == 0:
        print("Error: cannot divide by zero.")
    else:
        return x/y

if __name__ == "__main__":
    main()
