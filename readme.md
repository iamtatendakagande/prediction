# 🚀 Prediction: A Shell for Model Deployment

**Prediction** is a base shell created for quickly deploying and serving pre-trained machine learning models (Keras/Scikit-Learn) with a modern web interface.

* The core models are saved in the `./machine/model` directory.
* Made by Tatenda Edmore Kagande.
* For more information, connect with me on [LinkedIn](https://www.linkedin.com/in/iamtatendakagande/).

---

## 1. Prerequisites

To run this application, you will need the following installed:

* **Python:** Version 3.8 or higher.
* **Tailwind CSS:** Used for styling the User Interface (UI).

---

## 2. Setup and Installation

Follow these steps to get your development environment set up.

### **Step 1: **Clone the Repository:**

```bash
git clone [https://github.com/iamtatendakagande/prediction.git](https://github.com/iamtatendakagande/prediction.git)
cd prediction
 ```

### **Step 2: **Create and activate a virtual environment:**
    (On Mac/Linux)
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    (On Windows)
    ```bash
    py -3 -m venv .venv
    .venv\Scripts\activate
    ```

### **Step 3: **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### **Step 4: **Configure Environment Variables:**
    Copy the example `.env.example` file to a new `.env` file.
    ```bash
    cp .env.example .env
    ```
    Now, open the `.env` file and add your database URL.

    **Example `.env` file:**
    ```
    FLASK_APP=prediction.py
    ```
---

### **Step 5: Running the Application:**
    # With your environment variables set, run the app:
    # With the dependencies and environment variables configured, run the following commands in two separate terminal windows to start the Flask server and the Tailwind CSS watcher.

    ```bash
    flask run
    ```

    ```bash
    npx tailwindcss -i ./static/src/input.css -o ./static/src/output.css --watch
    ```

### **Step 4: How to Use the App:**
    Option A The model can be re-trained [Keras/Scikit-Learn] model. The file is found is ./machine/model/NeuralNetworkModel.py and will saved as ./machine/pickled/NetworkModel.keras.

    Option B: This project is already trained and the file is ./machine/pickled/NeuralNetworkModel.keras.