import math

#fungsi aktivasi sigmoid
def activation(x):
    return 1 / (1 + math.exp(-x))

#fungsi prediksi, jika >=0.5 dibulatkan ke atas
def pred(activation):
    if activation >= 0.5:
        prediction = 1.0
    else:
        prediction = 0.0
    return prediction

def forward(weights, x):
    z = sum(weights[i + 1] * x[i] for i in range(4)) + weights[0]
    output = activation(z)
    predicted = 1 if output > 0.5 else 0
    return z, output, predicted

def evaluate(dataset, weights):
    correct = 0
    sum_squared_error = 0.0

    for item in dataset:
        x = item["x"]
        target = item["y"]
        z, output, predicted = forward(weights, x)
        error = output - target
        squared_error = error ** 2

        correct += int(predicted == target)
        sum_squared_error += squared_error

    accuracy = correct / len(dataset)
    mse = sum_squared_error / len(dataset)
    return accuracy, mse

def training_validation (training_data, validation_data, epochs, weights, learning_rate):
    summary = []
    for epoch in range(1, epochs + 1):
            # TRAINING: hitung output menggunakan bobot saat ini,
            # kemudian update bobot segera setelah satu baris selesai.
            train_correct = 0
            train_squared_error_sum = 0.0
    
            for item in training_data:
                x = item["x"]
                target = item["y"]
                weights_before = weights[:]
    
                z, output, predicted = forward(weights_before, x)
                error = output - target
                squared_error = error ** 2
    
                # Turunan Squared Error terhadap net input.
                delta = 2.0 * (output - target) * output * (1.0 - output)
    
                # x0=1 untuk bias, dilanjutkan X1-X4.
                input_with_bias = [1.0] + x
                weights = [
                    weights_before[i]
                    - learning_rate * delta * input_with_bias[i]
                    for i in range(5)
                ]
    
                train_correct += int(predicted == target)
                train_squared_error_sum += squared_error
    
            train_accuracy = train_correct / len(training_data)
            train_mse = train_squared_error_sum / len(training_data)
    
            # VALIDATION: gunakan bobot akhir epoch; tidak ada update bobot.
            val_accuracy, val_mse = evaluate( validation_data, weights[:])
    
            summary.append({
                "epoch": epoch,
                "training_accuracy": train_accuracy,
                "training_mse": train_mse,
                "validation_accuracy": val_accuracy,
                "validation_mse": val_mse,
                "bias_final": weights[0],
                "theta1_final": weights[1],
                "theta2_final": weights[2],
                "theta3_final": weights[3],
                "theta4_final": weights[4],
            })

    return summary