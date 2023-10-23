from dataset import LargeMovieDataset
import os
import cnn
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    MODELS_DIR = 'models'
    check_dir(MODELS_DIR)
    # Create dataset
    training_set = LargeMovieDataset(path='../aclImdb_v1/', set='train')
    test_set = LargeMovieDataset(path='../aclImdb_v1/', set='test')

    print("======================================")
    print("Data loaded")
    print('Training set size:', len(training_set))
    print('Test set size:', len(test_set))

    #make the tfidf for the training set
    tfidf = TfidfVectorizer(max_features=10000)
    X_train = []
    y_train = []
    for i in range(len(training_set)):
        X_train.append(training_set[i][0])
        y_train.append(training_set[i][1])
    X_train = tfidf.fit_transform(X_train)

    print("======================================")
    print("Tfidf done")
    print("Number of features:", tfidf.get_feature_names_out())

    # Create model
    cnnModel = cnn.SentimentCNN(vocab_size=10000, embedding_dim=100, hidden_dim=100)
    cnn.train(cnnModel, training_set, epochs=10)

    print("======================================")
    print("Training done")

    # Test model
    X_test = []
    y_test = []
    for i in range(len(test_set)):
        X_test.append(test_set[i][0])
        y_test.append(test_set[i][1])
    X_test = tfidf.transform(X_test)

    print("======================================")
    print("Testing done")

    # Evaluate model
    cnnModel.evaluate(X_test, y_test)

    print("======================================")
    print("Evaluation done")
    print("Accuracy:", cnnModel.accuracy)

    # Save model
    cnnModel.save('model.pt')

    print("======================================")
    print("Model saved")