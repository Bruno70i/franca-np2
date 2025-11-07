import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Definição de Parâmetros ---
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10 # Comece com 10, aumente se não houver overfitting

# Caminhos para os dados (ajuste conforme sua estrutura)
# (Assumindo que você já separou em 'train', 'validation' e 'test'
# e já agrupou as sub-classes em 'urbana' e 'natural')
train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

# --- 2. Pré-processamento e Augmentation ---
#
# Data Augmentation para o conjunto de treino
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apenas normalização para validação e teste
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary' # Classificação binária
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Importante para a matriz de confusão
)

# --- 3. Modelagem (Transfer Learning) ---
#
# Carregar o modelo base (MobileNetV2) pré-treinado no ImageNet
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False, # Não incluir a camada de classificação
    weights='imagenet'
)

# Congelar as camadas do modelo base
base_model.trainable = False

# - Justificativa da escolha:
# A MobileNetV2 é uma arquitetura leve e eficiente, ideal para
# aprendizado rápido e com bom desempenho em tarefas de classificação,
# tornando o Transfer Learning eficaz.

# Adicionar nossas próprias camadas no topo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # Regularização para evitar overfitting
predictions = Dense(1, activation='sigmoid')(x) # 1 neurônio para saída binária

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Treinamento ---
#
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

print("\n--- Treinamento Concluído ---")






# ----------------------- parte dois do codigo ------------------------





from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

print("\n--- 5. Avaliação do Modelo ---")

# --- 5.1 Gráficos de Treinamento (Acurácia e Perda) ---
#
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia (Treino)')
plt.plot(epochs_range, val_acc, label='Acurácia (Validação)')
plt.legend(loc='lower right')
plt.title('Acurácia de Treinamento e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda (Treino)')
plt.plot(epochs_range, val_loss, label='Perda (Validação)')
plt.legend(loc='upper right')
plt.title('Perda de Treinamento e Validação')
plt.show()

# --- 5.2 Previsões e Métricas no Conjunto de Teste ---
#
Y_pred_probs = model.predict(test_generator)
# Converter probabilidades (sigmoid) para classes (0 ou 1)
Y_pred = (Y_pred_probs > 0.5).astype(int)

# Obter as classes verdadeiras
Y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys()) # ['natural', 'urbana']

# --- 5.3 Matriz de Confusão ---
#
cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_labels, 
    yticklabels=class_labels
)
plt.title('Matriz de Confusão')
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Prevista')
plt.show()

# --- 5.4 Precisão, Recall e F1-Score ---
#
print("\nRelatório de Classificação (Métricas):")
# target_names: 0='natural', 1='urbana' (Keras ordena alfabeticamente)
report = classification_report(Y_true, Y_pred, target_names=class_labels)
print(report)

# --- 5.5 Curva ROC e AUC ---
#
fpr, tpr, thresholds = roc_curve(Y_true, Y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.show()