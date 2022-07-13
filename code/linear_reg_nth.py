import numpy as np
import matplotlib.pyplot as plt

class linear_reg_nth():
    def __init__(self, degree):
        self.degree = degree
        self.ws = [np.random.random() for x in range(degree)] # w1, w2, w3
        self.b1 = 0.5
        self.lr = 0.1

    def predict(self, X, Y):
        preds = [sum([x**(i+1) * self.ws[i] for i in range(self.degree)]) + self.b1 for x in X]
        error = 1/2 * sum([(y-yhat)**2 for y, yhat in zip(Y, preds)])
        return error, preds

    def backpropagate(self, X, Y, preds):
        w_changes = [sum([(yhat - y) * x**(i+1) for x, y, yhat in zip(X, Y, preds)])/len(preds) for i in range(self.degree)]
        b_change = sum([(yhat - y) for y, yhat in zip(Y, preds)])/len(preds)

        self.ws = [w - self.lr * (c) for w, c in zip(self.ws, w_changes)]
        self.b1 = self.b1 - self.lr * b_change

def main():

    xs = [0.09,-0.44,-0.15,0.69,-0.99,-0.76,0.34,0.65,-0.73,0.15,0.78,-0.58,-0.63,-0.78,-0.56,0.96,0.62,-0.66,0.63,-0.45,-0.14,0.88,0.64,-0.33,-0.65]
    ys = [1.12,0.65,0.92,0.93,0.22,0.5,1.05,0.97,0.66,1.11,0.97,0.79,0.72,0.48,0.54,0.49,0.97,0.63,0.83,0.82,0.82,0.67,0.76,0.81,0.73]
    epochs = 1000
    deg = 3

    model = linear_reg_nth(degree=deg)
    error, preds = model.predict(xs, ys)

    for i in range(epochs+1):
        model.backpropagate(xs, ys, preds)
        error, preds = model.predict(xs, ys)

        line_x = np.linspace(min(xs),max(xs))
        _, line_y = model.predict(line_x, line_x)

        if i%50 == 0:
            plt.scatter(line_x, line_y, linestyle='solid',linewidth=1, c='grey', alpha=0.1)
            plt.scatter(xs, ys, c='blue')
            plt.scatter(xs, preds, c='red')
            plt.xlim([-1.2, 1.2])
            plt.ylim([0, 1.2])
            plt.title("Epoch {}/{}, Error: {}".format(i, epochs, error))
            if i == 1000:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(0.2)
                plt.close()

    print(model.ws, model.b1)
    
if __name__ == '__main__':
    main()