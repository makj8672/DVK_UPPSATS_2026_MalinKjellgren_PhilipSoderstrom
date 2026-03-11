print("Hello, World!")

print("Hej Malin!")

# En enkel kalkylator som adderar två tal med input från användaren och hanterar ogiltiga inmatningar.
def get_numbers():
    while True:
        try:
            a = int(input("First number: "))
            b = int(input("Second number: "))
            return a, b
        except ValueError:
            print("Invalid input, try again!")

def add_numbers(a, b):
    return a + b

def print_result(a, b, result):
    print(f"{a} + {b} = {result}")

def main():
    a, b = get_numbers()
    result = add_numbers(a, b)
    print_result(a, b, result)

if __name__ == "__main__":
    main()