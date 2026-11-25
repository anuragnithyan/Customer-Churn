import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
tenure = np.random.randint(1, 60, 1000)
monthly_charges = np.random.uniform(20, 120, 1000)
data_usage = np.random.randint(1, 60, 1000)
support_tickets = np.random.randint(0, 10, 1000)
np.random.seed(42)
churn=((tenure<12)& (support_tickets >3)) |(monthly_charges>100)
Churn=churn.astype(int)
df=pd.DataFrame({'Tenure':monthly_charges,'Data usage':data_usage,'Support tickets':support_tickets,'Churn':churn})
X = df[['Tenure', 'Data usage', 'Support tickets']]
Y = df['Churn']
print(df['Churn'].value_counts())
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
loss, acc = model.evaluate(X_test, Y_test)
print('Loss:', loss)
print('Accuracy:', acc)
customer = pd.DataFrame([[200.0, 110.0, 5.0]], columns=['Tenure', 'Data usage', 'Support tickets'])
customer_scaled = scaler.transform(customer)
prediction = model.predict(customer_scaled)
print('\nChurn Probability:', prediction[0][0])
print('Predicted Churn:', 'Yes' if prediction[0][0] > 0.5 else 'No')
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()
