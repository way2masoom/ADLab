import pickle

def add_numbers(num1, num2):
    result = num1 + num2
    with open("add.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"Result {num1} + {num2} = {result} saved to add.pkl")

if __name__ == "__main__":
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    add_numbers(num1, num2)
