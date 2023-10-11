# cs451-101Z-HW3
Repository for cs451 HW3 - Advanced Topics

## Windows 10/11 Set Up Steps
```powershell
# run the following in a powershell window

# clone the repo
git clone https://github.com/matt-berseth/cs451-101Z-HW3

# path into the repo
cd cs451-101Z-HW3

# create the virtual env
python3.10 -m venv .venv

# activate
.\.venv\Scripts\activate.ps1

# install the deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# launch vscode
code .
```

## Ubuntu 22.04 Set Up Steps
```bash
# run the following in a ubuntu window

# clone the repo
git clone https://github.com/matt-berseth/cs451-101Z-HW3

# path into the repo
cd cs451-101Z-HW3

# create the virtual env
python3 -m venv .venv

# activate
source ./.venv/bin/activate

# install the deps
pip install --upgrade pip
pip install -r requirements.txt

# launch vscode
code .
```

## Instructions
**Machine Learning Homework Assignment - Deep Learning**

**Objective:**  
Evaluate the performance of different hyper parameters for a simple deep neural network.

---

**Tasks:**

1. Complete the TODOs in `main.py`
2. Write a 1-2 page report summarizing the findings of your optimization tasks.
   - Include the output of the confusion matrixes of your best performing model parameters.
   - Include a table that includes the hyperparameter values and corresponding accuracy
   - A minimum of 15 different combinations of hyper parameters need to be evaluated. Try to select the set of hyper-parameters that provide the highest val_accuracy value.

|activation|n_filters|n_neurons_in_dense_layer|optimzer|val_accuracy|
|---	|---	|---	|---	|---	|
|tanh|8|16|sgd|.9324|
|tanh|32|16|sgd|.9564|

---

**Deliverables**:

1. Python code (`main.py`
2. Report summarizing findings and insights (`.pdf`).

---

**Notes**:
- Plagiarism will result in a zero for the assignment.
