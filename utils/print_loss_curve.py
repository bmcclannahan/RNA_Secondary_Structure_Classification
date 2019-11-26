import matplotlib.pyplot as plt

loss_file_location = "/scratch/b523m844/RNA_Secondary_Structure_Classification/resnet/loss.txt"

loss_file = open(loss_file_location,'r')

lines = loss_file.readlines()
y_values = list(map(lambda x: float(x.rstrip()),lines))
x_values = list(range(len(y_values)))


fig = plt.figure()
ax = plt.axes()

ax.plot(x_values, y_values)
plt.show()