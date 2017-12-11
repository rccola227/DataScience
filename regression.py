import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

for zzz in range(1000):
    plt.ion()



    data = pd.read_csv('./ex1data1.txt', names=['x', 'y'])
    X_df = pd.DataFrame(data.x)
    Y_df = pd.DataFrame(data.y)


    X = np.array(X_df)
    Y = np.array(Y_df)
    Y = Y.flatten()
    X_df ['intercept'] = 1
    samples = np.array(X_df)

    printIterations = 5

    def gradientDescent(theta, alpha, threshold, maxIterations):
        i = 0
        while True:
        
     
        
            hypothesis = samples.dot(theta)
            
            if i % 100 == 0:
                line1.set_ydata(hypothesis)
                ax = plt.gca()
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
            
            loss = hypothesis - Y
            lossRespectTheta0 = np.dot(loss, X)
            lossGradientTheta0 = np.sum(lossRespectTheta0) / len(X)
            
            lossRespectTheta1 = loss;
            lossGradientTheta1 = np.sum(lossRespectTheta1) / len(X)

            theta[0] -= lossGradientTheta0 * alpha
            theta[1] -= lossGradientTheta1 * alpha

            if (i > maxIterations):
                return theta[0]
            i = i+1
            time.sleep(0.0001)
      
    fig = plt.figure()
    graph = fig.add_subplot(111)
    line2, = graph.plot(X, Y, 'kx')
    line1, = graph.plot(X, Y, 'k.')
    fig.canvas.draw()
    plt.show()
    print(gradientDescent([1, 1], 0.01, 0.1, 1500))
    plt.close()


