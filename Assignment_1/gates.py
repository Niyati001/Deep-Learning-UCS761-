AND= [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]
OR= [(0,0,0), (0,1,0), (1,0,1), (1,1,1)]
NAND= [(0,0,1), (0,1,1), (1,0,1), (1,1,0)]
NOR= [(0,0,1), (0,1,0), (1,0,0), (1,1,0)]
XOR=[(0,0,0), (0,1,1), (1,0,1),(1,1,0)]

def run_gate(name, dataset):
    p= Perceptron(lr=0.1, epochs=50)
    p.train(dataset)

    print(f"\n{name}")
    print("Weights:", p.w1, p.w2)
    print("Bias:", p.b)

    for x1,x2,y in dataset:
        print(x1,x2, "->", p.predict(x1,x2))

run_gate("AND", AND)
run_gate("OR", OR)
run_gate("NAND", NAND)
run_gate("NOR", NOR)
run_gate("XOR", XOR)