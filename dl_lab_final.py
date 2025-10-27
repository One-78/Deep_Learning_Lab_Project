

import pandas as pd
import json
import os
from tqdm.notebook import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

class CFG:
    seeds = [42, 119, 2020]
    vocab_size = 20000
    max_length = 256
    batch_size = 128
    fine_tune_epochs = 40
    learning_rate = 1e-3  # High LR
    warmup_epochs = 2  # Add warmup
    weight_decay = 0.01  # Add regularization

train_df = pd.read_csv(os.path.join(llm_classification_finetuning_path, 'train.csv'))
train_df.head()

prompt_list = []
targets = []
for i in tqdm(range(len(train_df))):
    prompts = json.loads(train_df.iloc[i]["prompt"])
    response_a = json.loads(train_df.iloc[i]["response_a"])
    response_b = json.loads(train_df.iloc[i]["response_b"])
    conversation_a = ""
    conversation_b = ""
    for j in range(len(prompts)):
        if response_a[j] is None:
            response_a[j] = "None"
        if response_b[j] is None:
            response_b[j] = "None"
        conversation_a += prompts[j] + "\n"
        conversation_a += response_a[j] + "\n"
        conversation_b += prompts[j] + "\n"
        conversation_b += response_b[j] + "\n"
    prompt_list.append((conversation_a, conversation_b))
    if train_df.iloc[i]["winner_tie"] == 1:
        targets.append(0)
    if train_df.iloc[i]["winner_model_a"] == 1:
        targets.append(1)
    if train_df.iloc[i]["winner_model_b"] == 1:
        targets.append(2)
len(prompt_list)

# Step 2: Define TextVectorization layer
text_vectorizer = TextVectorization(max_tokens=CFG.vocab_size, output_mode='int', output_sequence_length=CFG.max_length)
text_vectorizer.adapt([item[0] for item in prompt_list] + [item[1] for item in prompt_list])

def get_dataset(prompt_list, targets, shuffle=True, batch_size=128):
    part1 = [item[0] for item in prompt_list]
    part2 = [item[1] for item in prompt_list]
    dataset = tf.data.Dataset.from_tensor_slices(((part1, part2), targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.batch(CFG.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def get_base_model(inputs, embedding):
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    return x
def get_model():
    inputs1 = tf.keras.Input(shape=(1,), dtype=tf.string)
    inputs2 = tf.keras.Input(shape=(1,), dtype=tf.string)
    embedding = tf.keras.layers.Embedding(input_dim=CFG.vocab_size, output_dim=64, mask_zero=True)
    x1 = get_base_model(inputs1, embedding)
    x2 = get_base_model(inputs2, embedding)
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Conv1D(32, 3, activation="relu")(x)
    x = tf.keras.layers.Conv1D(32, 3, activation="relu")(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Conv1D(64, 3, activation="relu")(x)
    x = tf.keras.layers.Conv1D(64, 3, activation="relu")(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="swish")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)

    ## model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

"""# Train the model with
    vocab_size = 20000
    max_length = 1024
    batch_size = 64  
    fine_tune_epochs = 15  
    learning_rate = 2e-5  
    warmup_epochs = 2
    weight_decay = 0.01  
"""

# Fine tuning and training the model
models = []
for seed in CFG.seeds:
    model_name = f"model_{seed}.keras"
    model_name_path = os.path.join(llm_classification_finetuning_path, model_name)

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        prompt_list, targets, test_size=0.2, random_state=seed, stratify=targets  # Add stratify
    )
    valid_ds = get_dataset(valid_texts, valid_labels, shuffle=False)

    if not os.path.exists(model_name_path):
        train_ds = get_dataset(train_texts, train_labels)
        model = get_model()

        # Use AdamW with weight decay
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',  # or your loss
            metrics=['accuracy']
        )

        # Callbacks
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_name_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Increased patience
            verbose=1,
            restore_best_weights=True,
            min_delta=1e-4  # Add minimum improvement threshold
        )

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive reduction
            patience=3,  # Increased patience
            min_lr=1e-7,
            verbose=1,
            min_delta=1e-4
        )

        # Optional: Add learning rate warmup
        def lr_schedule(epoch):
            if epoch < CFG.warmup_epochs:
                return CFG.learning_rate * (epoch + 1) / CFG.warmup_epochs
            return CFG.learning_rate

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

        # Fine-tune
        history = model.fit(
            train_ds,
            epochs=CFG.fine_tune_epochs,
            validation_data=valid_ds,
            callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback, lr_scheduler],
            verbose=1
        )

        # Load best weights
        model = tf.keras.models.load_model(model_name_path)
    else:
        model = tf.keras.models.load_model(model_name_path)

    loss, acc = model.evaluate(valid_ds, verbose=0)
    print(f"Seed {seed} - Validation Loss: {loss:.4f} | Validation Accuracy: {acc * 100:.2f}%")
    if 'history' in locals():
        print(f"Best epoch: {np.argmin(history.history['val_loss']) + 1}")
    models.append(model)

"""Fine Tuning the model with  
vocab_size = 20000
max_length = 1024
batch_size = 32
fine_tune_epochs = 20
learning_rate = 5e-6  
warmup_epochs = 2  
weight_decay = 0.01  
"""

# Continue training from best saved model
for seed in CFG.seeds:
    model_name = f"model_{seed}.keras"
    model_name_path = os.path.join(llm_classification_finetuning_path, model_name)

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        prompt_list, targets, test_size=0.2, random_state=seed, stratify=targets
    )
    train_ds = get_dataset(train_texts, train_labels)
    valid_ds = get_dataset(valid_texts, valid_labels, shuffle=False)

    # Load existing model
    if os.path.exists(model_name_path):
        print(f"Loading existing model for seed {seed}...")
        model = tf.keras.models.load_model(model_name_path)
        initial_loss, initial_acc = model.evaluate(valid_ds, verbose=0)
        print(f"Initial - Loss: {initial_loss:.4f} | Accuracy: {initial_acc*100:.2f}%")
    else:
        print(f"No existing model found for seed {seed}, creating new one...")
        model = get_model()
        initial_loss = float('inf')

    # Reduce learning rate for continued training
    current_lr = CFG.learning_rate * 0.1  # Use 10x smaller LR
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=current_lr,
        weight_decay=CFG.weight_decay
    )
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_name_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True,
        min_delta=1e-5
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )

    # Continue training
    history = model.fit(
        train_ds,
        epochs=CFG.fine_tune_epochs,
        validation_data=valid_ds,
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback],
        verbose=1
    )

    # Load best weights and evaluate
    model = tf.keras.models.load_model(model_name_path)
    final_loss, final_acc = model.evaluate(valid_ds, verbose=0)
    print(f"Final - Loss: {final_loss:.4f} | Accuracy: {final_acc*100:.2f}%")
    print(f"Improvement: {(final_loss - initial_loss):.4f}")

