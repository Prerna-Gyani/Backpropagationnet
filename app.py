import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
            
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100, st_placeholder=None, progress_bar=None):
        X = np.c_[X, np.ones((X.shape[0]))]
        
        losses = []

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
                
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                losses.append(loss)
                
                log_msg = "[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss)
                
                if st_placeholder:
                    st_placeholder.text(log_msg)
                else:
                    print(log_msg) 
            
            if progress_bar:
                progress_bar.progress((epoch + 1) / epochs)

        return losses

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)
            
        error = A[-1] - y
        
        D = [error * self.sigmoid_deriv(A[-1])]
        
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
            
        D = D[::-1]
        
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
            
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
            
        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss

@st.cache_data
def load_xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return X, y

@st.cache_data
def load_mnist_data():
    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min()) / (data.max() - data.min())
    
    (trainX, testX, trainY, testY) = train_test_split(data,
        digits.target, test_size=0.25)
    
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)
    
    return trainX, testX, trainY, testY

st.title("🧠 Neural Network Dashboard")
st.write("A Streamlit front-end for the custom `NeuralNetwork` class.")

st.sidebar.header("Configuration")
dataset_name = st.sidebar.selectbox("Choose Dataset", ("XOR", "MNIST Digits"))

if dataset_name == "XOR":
    X_train, y_train = load_xor_data()
    X_test, y_test = X_train, y_train 
    input_dim = 2
    output_dim = 1
    default_layers = "2"
    default_epochs = 20000
    default_alpha = 0.5
    st.sidebar.info("Input: 2, Hidden: ?, Output: 1")
    
else: 
    X_train, X_test, y_train, y_test = load_mnist_data()
    input_dim = X_train.shape[1] 
    output_dim = y_train.shape[1] 
    default_layers = "32, 16"
    default_epochs = 1000
    default_alpha = 0.1
    st.sidebar.info(f"Input: {input_dim}, Hidden: ?, Output: {output_dim}")

st.sidebar.subheader("Hyperparameters")
hidden_layers_str = st.sidebar.text_input("Hidden Layers (comma-separated)", default_layers)
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, default_alpha, 0.01)
epochs = st.sidebar.number_input("Epochs", min_value=100, max_value=100000, value=default_epochs, step=100)
display_update = 100 

try:
    hidden_layers = [int(l.strip()) for l in hidden_layers_str.split(",") if l.strip()]
    layers = [input_dim] + hidden_layers + [output_dim]
    st.sidebar.write(f"**Final Architecture:** `{' -> '.join(map(str, layers))}`")
except Exception as e:
    st.sidebar.error("Invalid hidden layer format. Use comma-separated numbers.")
    st.stop()


if st.sidebar.button("Start Training"):
    if 'results' in st.session_state:
        del st.session_state['results']

    st.header("Training Progress")
    
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    with st.spinner(f"Training on {dataset_name} for {epochs} epochs..."):
        nn = NeuralNetwork(layers, alpha=alpha)
        losses = nn.fit(X_train, y_train, epochs=epochs, 
                        displayUpdate=display_update,
                        st_placeholder=status_placeholder,
                        progress_bar=progress_bar)
        
        status_placeholder.text("Training complete!")
        progress_bar.progress(1.0)
        
        st.session_state.results = {
            "nn": nn,
            "losses": losses,
            "dataset": dataset_name
        }

if 'results' in st.session_state:
    results = st.session_state.results
    nn = results["nn"]
    
    st.header(f"Results for {results['dataset']} Dataset")
    
    st.subheader("Loss Curve")
    loss_df = pd.DataFrame({
        "Epoch": range(1, len(results["losses"]) + 1),
        "Loss": results["losses"]
    })
    st.line_chart(loss_df.set_index("Epoch"))

    st.subheader("Evaluation")
    
    if results["dataset"] == "XOR":
        st.write("Predictions on XOR data:")
        preds_output = []
        for (x, target) in zip(X_test, y_test):
            pred = nn.predict(x)[0][0]
            step = 1 if pred > 0.5 else 0
            preds_output.append(
                f"data={x}, ground-truth={target[0]}, pred={pred:.4f}, step={step}"
            )
        st.code("\n".join(preds_output))

    else: 
        st.write("Classification Report on Test Set:")
        predictions = nn.predict(X_test)
        predictions = predictions.argmax(axis=1)
        
        report = classification_report(y_test.argmax(axis=1), predictions, 
                                       target_names=[f"Digit {i}" for i in range(10)],
                                       output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)