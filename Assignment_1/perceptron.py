class Perceptron:
    def __init__(self, lr= 0.1, epochs=100):
        self.lr= lr
        self.epochs= epochs
        self.w1=0
        self.w2=0
        self.b=0

    def predict(self, x1, x2):
        z= self.w1*x1 + self.w2*x2 + self.b
        return 1 if z>= 0 else 0

    def train(self, dataset):
        for _ in range(self.epochs):
            for x1, x2, y in dataset:
                y_hat= self.predict(x1, x2)
                error= y- y_hat

                self.w1+= self.lr* error* x1
                self.w2+= self.lr* error* x2
                self.b+= self.lr*error


if __name__ == "__main__":
    gates = {
        "AND":  [(0,0,0),(0,1,0),(1,0,0),(1,1,1)],
        "OR":   [(0,0,0),(0,1,1),(1,0,1),(1,1,1)],
        "NAND": [(0,0,1),(0,1,1),(1,0,1),(1,1,0)],
        "NOR":  [(0,0,1),(0,1,0),(1,0,0),(1,1,0)],
        "XOR":  [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
    }

    for name, dataset in gates.items():
        print("\nTraining:", name)

        p = Perceptron(lr=0.1, epochs=50)
        p.train(dataset)

        print("Final Weights:", p.w1, p.w2)
        print("Final Bias:", p.b)

        for x1, x2, y in dataset:
            print(x1, x2, "->", p.predict(x1, x2))