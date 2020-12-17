import matplotlib.pyplot as plt

train_loss_file_location = "utils/train_loss.txt"
val_loss_file_location = "utils/val_loss.txt"

train_loss_file = open(train_loss_file_location,'r')

lines = train_loss_file.readlines()
train_y_values = list(map(lambda x: float(x.rstrip()),lines))
train_x_values = list(range(len(train_y_values)))

val_loss_file = open(val_loss_file_location,'r')

lines = val_loss_file.readlines()
val_y_values = list(map(lambda x: float(x.rstrip()),lines))
val_x_values = list(range(len(val_y_values)))


fig = plt.figure()
ax = plt.axes()

ax.plot(train_x_values,train_y_values,color='blue')
ax.plot(val_x_values, val_y_values,color='orange')
plt.show()