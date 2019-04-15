library(keras)

# Data Preparation ---------------------------------------------------
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()

x_train <- array_reshape(x_train, c(nrow(x_train), 28*28))
x_test <- array_reshape(x_test, c(nrow(x_test), 28*28))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train)
y_test <- to_categorical(y_test)

# visualize the digits
mnist <- dataset_mnist()
index = sample(1:60000,10000)

train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

par(mfcol=c(3,6))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for (idx in index[1:18]) { 
  im <- train_images[idx, ,]
  im <- t(apply(im, 2, rev)) 
  image(1:28, 1:28, im, col=gray((0:255)/255), 
        xaxt='n', main=paste(train_labels[idx]))
}

par(mfrow=c(1,1))
digit <- x_train[16,]
plot(as.raster(digit, max = 255))

# Mix Train/Test---------------------------------------------------------------

x_test = x_train[index,]
x_train = rbind(x_train[-index,],x_test)

y_test = y_train[index,]
y_train = rbind(y_train[-index,],y_test)

str(x_train)
str(y_train)
str(x_test)
str(y_test)

# Define Model --------------------------------------------------------------
network <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", 
              input_shape = c(28*28)) %>%
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")

network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


# Training & Evaluation ----------------------------------------------------
# Fit model to data

network %>% fit(x_train, y_train, epochs = 20, validation_split = 0.20,batch_size = 128)


score <- network %>% evaluate(
  x_test, y_test,
  verbose = 0
)

score


# Output metrics
plot(history)
network %>% predict_classes(x_test[1:18,])
