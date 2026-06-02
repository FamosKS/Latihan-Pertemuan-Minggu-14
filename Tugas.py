import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ==========================================
# 1. PERSIAPAN DATASET (CIFAR-10)
# ==========================================
def prepare_cifar10_dataset():
    print("1. Memuat Dataset CIFAR-10...")
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Sub-sampling untuk mempercepat eksperimen (opsional, ganti jika perlu full data)
    subset_size = 20000 
    X_train_full, y_train_full = X_train_full[:subset_size], y_train_full[:subset_size]
    X_test, y_test = X_test[:4000], y_test[:4000]

    # Normalisasi piksel
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # One-hot encoding
    y_train_full = keras.utils.to_categorical(y_train_full, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    # Split Validation (80:20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"Bentuk Data Latih: {X_train.shape}")
    print(f"Bentuk Data Validasi: {X_val.shape}")
    print(f"Bentuk Data Uji: {X_test.shape}\n")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test_cat, y_test)

# ==========================================
# 2. DATA AUGMENTATION PIPELINE
# ==========================================
def create_augmenter():
    print("2. Menyiapkan ImageDataGenerator...")
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )
    return datagen

def visualize_augmentation(X_train, y_train, datagen, class_names):
    # Ambil satu gambar sampel
    img = X_train[0]
    label = np.argmax(y_train[0])
    img_tensor = np.expand_dims(img, axis=0)
    
    # Generate augmented images
    aug_iter = datagen.flow(img_tensor, batch_size=1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(img)
    plt.title(f"Asli ({class_names[label]})", fontsize=10)
    plt.axis('off')
    
    for i in range(5):
        plt.subplot(1, 6, i + 2)
        batch = next(aug_iter)
        plt.imshow(batch[0])
        plt.title(f"Aug {i+1}", fontsize=10)
        plt.axis('off')
        
    plt.suptitle("Visualisasi Teknik Data Augmentation")
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. MEMBANGUN CNN FROM SCRATCH
# ==========================================
def build_cnn_scratch(input_shape=(32, 32, 3), num_classes=10):
    print("3. Membangun CNN dari Awal (From Scratch)...")
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.35),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 4. TRANSFER LEARNING (MobileNetV2)
# ==========================================
def build_transfer_learning(input_shape=(32, 32, 3), num_classes=10):
    print("4. Membangun Transfer Learning Model (MobileNetV2)...")
    
    # Input tensor
    inputs = keras.Input(shape=input_shape)
    
    # Resize layer (MobileNetV2 butuh min 32x32, tapi bekerja lebih baik di ukuran lebih besar. Kita ubah ke 96x96)
    x = layers.Resizing(96, 96)(inputs)
    
    # Preprocess
    x = keras.applications.mobilenet_v2.preprocess_input(x * 255.0) 
    
    # Base Model
    base_model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 4.a Feature Extraction (Freeze base)
    base_model.trainable = False
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    return model, base_model

# ==========================================
# 5. GRAD-CAM (Visualisasi Keputusan)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Buat sub-model untuk mendapatkan output layer conv terakhir dan output prediksi
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Gradient Tape
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Hitung gradien
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Timbang channel dengan gradien
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def display_gradcam(img, heatmap, label, class_names):
    # Resize heatmap agar sama dengan gambar
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Gabungkan gambar asli dengan heatmap
    img_bgr = np.uint8(255 * img)
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Asli: {class_names[label]}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. EVALUASI & PLOTTING
# ==========================================
def plot_training_curves(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'Akurasi: {title}')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'Loss: {title}')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.show()

def evaluate_model(model, X_test, y_test_cat, y_test_true, class_names, model_name):
    print(f"\n--- EVALUASI: {model_name} ---")
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    inf_time = time.time() - start_time
    
    print(f"Waktu Inference: {inf_time:.2f} detik untuk {len(X_test)} sampel.")
    print("\nClassification Report:")
    print(classification_report(y_test_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.show()
    
    return y_pred_probs

# ==========================================
# PROGRAM UTAMA
# ==========================================
def main():
    print("=" * 60)
    print("KLASIFIKASI CITRA: CNN FROM SCRATCH vs TRANSFER LEARNING")
    print("=" * 60)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
                   
    # 1. Load Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test_cat, y_test_true) = prepare_cifar10_dataset()
    
    # 2. Augmentasi
    datagen = create_augmenter()
    visualize_augmentation(X_train, y_train, datagen, class_names)
    
    # Define Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # 3. Latih CNN Scratch (dengan augmentasi)
    cnn_model = build_cnn_scratch()
    print("Melatih CNN from Scratch (10 Epochs)...")
    start_time = time.time()
    history_cnn = cnn_model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=10, 
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    cnn_time = time.time() - start_time
    print(f"Waktu Pelatihan CNN Scratch: {cnn_time:.2f} detik")
    plot_training_curves(history_cnn, "CNN From Scratch (Augmented)")
    
    # 4. Latih Transfer Learning
    tl_model, tl_base = build_transfer_learning()
    print("\nMelatih Transfer Learning - Feature Extraction (5 Epochs)...")
    start_time = time.time()
    history_tl = tl_model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=5,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\nMelatih Transfer Learning - Fine Tuning...")
    # 4.b Fine Tuning: Unfreeze layer terakhir
    tl_base.trainable = True
    for layer in tl_base.layers[:-20]: # Unfreeze 20 layer teratas
        layer.trainable = False
        
    tl_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), # LR sangat kecil
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history_tl_fine = tl_model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=5,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    tl_time = time.time() - start_time
    print(f"Waktu Pelatihan Keseluruhan Transfer Learning: {tl_time:.2f} detik")
    plot_training_curves(history_tl_fine, "MobileNetV2 (Fine-Tuning)")
    
    # 5. Evaluasi Kinerja
    probs_cnn = evaluate_model(cnn_model, X_test, y_test_cat, y_test_true, class_names, "CNN From Scratch")
    probs_tl = evaluate_model(tl_model, X_test, y_test_cat, y_test_true, class_names, "MobileNetV2 Transfer Learning")
    
    # 6. Grad-CAM Visualization pada model CNN Scratch
    print("\nVisualisasi Keputusan Model (Grad-CAM)...")
    # Cari nama layer konvolusi terakhir di cnn_model (biasanya berakhiran conv2d_n)
    last_conv_name = [layer.name for layer in cnn_model.layers if isinstance(layer, layers.Conv2D)][-1]
    
    sample_idx = 10 # Ambil satu sampel gambar Kuda/Pesawat/dll dari X_test
    img_sample = X_test[sample_idx]
    label_true = y_test_true[sample_idx]
    img_tensor = np.expand_dims(img_sample, axis=0)
    
    heatmap = make_gradcam_heatmap(img_tensor, cnn_model, last_conv_name)
    display_gradcam(img_sample, heatmap, label_true, class_names)
    
    # 7. Rangkuman Trade-off
    print("\n" + "=" * 50)
    print("KESIMPULAN & TRADE-OFF")
    print("=" * 50)
    print(f"{'Metrik':<20} | {'CNN Scratch':<15} | {'MobileNetV2':<15}")
    print("-" * 55)
    print(f"{'Total Parameter':<20} | {cnn_model.count_params():<15,} | {tl_model.count_params():<15,}")
    print(f"{'Waktu Training':<20} | {cnn_time:.1f} detik{'':<4} | {tl_time:.1f} detik")

if __name__ == "__main__":
    main()