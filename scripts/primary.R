library(tidyverse)
library(keras)
library(tensorflow)
library(dplyr)
library(tidymodels)
library(yardstick)

# Set module-2-group6 as your working dir
load("./data/claims-clean-w-headers.RData")
load("./data/claims-clean-example.RData")
load("./data/claims-test.RData")
source("./scripts/preprocessing.R")

set.seed(123)

tokenizer <- text_tokenizer(num_words = 1000)
tokenizer %>% fit_text_tokenizer(claims_clean_w_h$text_clean)

# Convert text to sequences
sequences <- texts_to_sequences(tokenizer, claims_clean_w_h$text_clean)
# Set sequence length
maxlen <- 1000
# Set equal length
same_len_data <- pad_sequences(sequences, maxlen = maxlen)
claims_clean_w_h <- claims_clean_w_h %>% 
  mutate(sequences = same_len_data)

claims_clean_w_h <- claims_clean_w_h %>%
  mutate(
    # Encode binary labels
    bclass_numeric = as.numeric(as.factor(bclass)) - 1,  # Convert to numeric starting at 0
    y_binary = to_categorical(bclass_numeric),          # Convert to one-hot encoded vectors
    
    # Encode multiclass labels
    mclass_numeric = as.numeric(as.factor(mclass)) - 1,  # Convert to numeric starting at 0
    y_multiclass = to_categorical(mclass_numeric)        # Convert to one-hot encoded vectors
  )

# Train-test split
set.seed(1)
partition <- initial_split(claims_clean_w_h, prop = 0.8)  # 80% training, 20% testing
train_data <- training(partition)
test_data <- testing(partition)

X_train <- train_data$sequences
X_test <- test_data$sequences

#Binary label
y_train_binary <- train_data$y_binary
y_test_binary <- test_data$y_binary

#Multiclass label
y_train_multiclass <- train_data$y_multiclass
y_test_multiclass <- test_data$y_multiclass

# Build RNN model for binary classification
binary_model <- keras_model_sequential()
binary_model %>%
  layer_embedding(input_dim = 5000, output_dim = 128, input_length = maxlen) %>%  # Embedding layer
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 2) %>%  # Dense layer
  layer_dropout(rate = 0.3) %>% 
  layer_activation(activation = 'sigmoid')  # Activation layer

summary(binary_model)

binary_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c('binary_accuracy')
)

# Train the binary classification model -> 76% val acc
history_binary <- binary_model %>% fit(
  X_train, y_train_binary,
  validation_split = 0.3,
  batch_size = 32,
  epochs = 10,
)

multiclass_model <- keras_model_sequential()
multiclass_model %>% 
  layer_embedding(input_dim = 5000, output_dim = 128, input_length = maxlen) %>%  # Embedding layer
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 5) %>%  # Dense layer
  layer_dropout(rate = 0.3) %>% 
  layer_activation(activation = 'softmax')  # Activation layer

multiclass_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c('binary_accuracy')
)

# Train the multiclass classification model -> 90% val acc
history_multiclass <- multiclass_model %>% fit(
  X_train, y_train_multiclass,
  validation_split = 0.3,
  epochs = 10,
  batch_size = 32
)

# Extract class labels for binary classification
bclass_labels <- claims_raw %>% pull(bclass) %>% levels()

# Predict on test data for binary classification
binary_predictions <- binary_model %>% predict(X_test)

# Convert predictions to class labels
bclass_pred <- apply(binary_predictions, 1, which.max) - 1
bclass_pred_labels <- factor(bclass_pred, labels = bclass_labels)

# Predict on test data for multiclass classification
multiclass_predictions <- multiclass_model %>% predict(X_test)

# Convert multiclass probabilities to predicted classes
mclass_pred <- apply(multiclass_predictions, 1, which.max) - 1  # Get zero-indexed class predictions

# Add class labels for multiclass predictions if needed
mclass_labels <- claims_raw %>% pull(mclass) %>% levels()
mclass_pred_labels <- factor(mclass_pred, labels = mclass_labels)

# Combine predictions into a dataframe
pred_df <- test_data %>%
  dplyr::select(.id) %>%  # Assuming .id is in test_data
  mutate(
    bclass.pred = bclass_pred_labels,
    mclass.pred = mclass_pred_labels  # Use labeled predictions
  )

save_model_tf(binary_model, "results/binary-RNN-model")
save_model_tf(multiclass_model, "results/multiclass-RNN-model")
save(pred_df, file = 'results/preds-group6.RData')

# Binary Classification Metrics
# Set up validation indices
val_start <- floor(0.7 * nrow(X_train)) + 1
X_val_binary <- X_train[val_start:nrow(X_train), ]
y_val_bclass <- y_train_binary[val_start:nrow(y_train_binary), ]

# Make predictions on validation set
bclass_val_predictions <- binary_model %>% predict(X_val)

# Convert true labels (one-hot) to numeric class labels
true_val_labels_bclass <- apply(y_val_bclass, 1, which.max) - 1  # Convert one-hot to 0 or 1
true_val_labels_bclass <- factor(true_val_labels_bclass, levels = c(0, 1))  # Convert to factor

# Create prediction dataframe
bclass_val_pred_df <- tibble(
  true_labels = true_val_labels_bclass,
  pred_prob = bclass_val_predictions[, 2],  # Probability for class 1
  pred_class = factor(bclass_val_predictions[, 2] > 0.5, labels = c("0", "1"))  # Thresholded predictions
)

# Define metrics
bclass_metrics_panel <- metric_set(sensitivity, specificity)

# Compute binary classification metrics
binary_class_metrics <- bclass_val_pred_df %>%
  bclass_metrics_panel(truth = true_labels, estimate = pred_class, event_level = "second")

# Compute binary accuracy
binary_accuracy <- yardstick::accuracy(bclass_val_pred_df, truth = true_labels, estimate = pred_class)

# Combine binary metrics
binary_combined_metrics <- bind_rows(binary_class_metrics, binary_accuracy)

# Multiclass Validation Data
X_val_mclass <- X_train[val_start:nrow(X_train), ]
y_val_mclass <- y_train_multiclass[val_start:nrow(y_train_multiclass), ]

# Predict probabilities for multiclass validation set
mclass_val_predictions <- multiclass_model %>% predict(X_val_mclass)

# Convert true labels (one-hot) to numeric class labels
true_val_labels_mclass <- apply(y_val_mclass, 1, which.max) - 1  # Convert one-hot to 0:N-1
true_val_labels_mclass <- factor(true_val_labels_mclass, levels = 0:(ncol(y_val_mclass) - 1))  # Convert to factor

# Get predicted class labels from probabilities
pred_class_mclass <- apply(mclass_val_predictions, 1, which.max) - 1  # Predicted classes
pred_class_mclass <- factor(pred_class_mclass, levels = 0:(ncol(y_val_mclass) - 1))  # Convert to factor

# Create multiclass prediction dataframe
mclass_val_pred_df <- tibble(
  mclass_true_labels = true_val_labels_mclass,
  mclass_pred_class = pred_class_mclass
)

# Compute sensitivity and specificity (class metrics)
mclass_metrics_panel <- metric_set(sensitivity, specificity)

mclass_metrics <- mclass_val_pred_df %>%
  mclass_metrics_panel(truth = mclass_true_labels, estimate = mclass_pred_class)

# Compute accuracy separately
m_class_accuracy_metric <- mclass_val_pred_df %>%
  yardstick::accuracy(truth = mclass_true_labels, estimate = mclass_pred_class)

# Combine results for reporting
mclass_combined_metrics <- bind_rows(mclass_metrics, m_class_accuracy_metric)

