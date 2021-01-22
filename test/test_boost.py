import eat

#Generate simulated data (seed, N)
dataset = eat.Data(1, 10).data

x = ["x1", "x2"]
y = ["y1", "y2"]

numStop = 5
J = 6
v = 1
M = 2

#Create model
model = eat.EATBoost(dataset, x, y, numStop, J, M, v)
#Fit model
model.fit_eat_boost()
print(model.predict())