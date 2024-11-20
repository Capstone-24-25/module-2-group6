library(tidyverse)
library(keras)
library(tensorflow)
library(dplyr)
library(tidymodels)

# Set module-2-group6 as your working dir
load("./data/claims-clean-w-headers.RData")
load("./data/claims-clean-example.RData")
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

# Encode binary label
claims_clean_w_h <- claims_clean_w_h %>% 
  mutate(y_binary = to_categorical(as.numeric(as.factor(claims_clean_w_h$bclass))-1))
# Encode multiclass label
claims_clean_w_h <- claims_clean_w_h %>% 
  mutate(y_multiclass = to_categorical(as.numeric(as.factor(claims_clean_w_h$mclass))-1))

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

# Train the binary classification model
history_binary <- binary_model %>% fit(
  X_train, y_train_binary,
  validation_split = 0.3,
  epochs = 10,
  batch_size = 32
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
history_binary <- multiclass_model %>% fit(
  X_train, y_train_multiclass,
  validation_split = 0.3,
  epochs = 10,
  batch_size = 32
)
