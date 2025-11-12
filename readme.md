# Predication is a shell created as a base for deploying models
* Other model are also saved in ./machine/model
* Made by Tatenda Edmore Kagande.
* For more information contact me on LinkedIn.

---

## 1. Prerequisites

* Python 3.8+

---

## 2. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/iamtatendakagande/prediction.git](https://github.com/iamtatendakagande/prediction.git)
    cd prediction
    ```

2.  **Create and activate a virtual environment:**
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

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
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

## 3. Running the Application
    # With your environment variables set, run the app:

    ```bash
    flask run
    ```

## 4. How to Use the App
    Option A The model can be re-trained [Keras/Scikit-Learn] model. The file is found is ./machine/model/NeuralNetworkModel.py and will saved as ./machine/pickled/NetworkModel.keras.

    Option B: This project is already trained and the file is ./machine/pickled/NeuralNetworkModel.keras.