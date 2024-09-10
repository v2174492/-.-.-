import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def house_model(y_new):
    # Входные данные (количество комнат)
    xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    
    # Целевые значения (цены в миллионах)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
    
    # Создаем простую модель с 1 плотным слоем и 1 нейроном
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    
    # Компиляция модели
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    # Обучение модели
    model.fit(xs, ys, epochs=500)
    
    # Возвращаем предсказание
    return model.predict(y_new)[0]

# Предсказание для дома с 7 комнатами
prediction = house_model([7])
print(f"Цена дома с 7 спальнями: {prediction[0]} миллионов")
