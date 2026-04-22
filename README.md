# 🎓 Student Pass/Fail Predictor

A simple ML web app built with **Streamlit** and **Logistic Regression** to predict whether a student will pass or fail based on their study habits and academic history.

---

## 📁 Project Structure

```
passfail-app/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── pass_fail_dataset_extended.csv  # Dataset (place here)
└── README.md
```

---

## 🚀 How to Run

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd passfail-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset
Place `pass_fail_dataset_extended.csv` in the project root.

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🧠 Model Details

| Item | Detail |
|------|--------|
| Algorithm | Logistic Regression |
| Features | Study Hours, Attendance, Previous Score |
| Target | Pass (1) / Fail (0) |
| Train/Test Split | 80% / 20% |
| Preprocessing | StandardScaler |

---

## 📊 App Pages

| Page | Description |
|------|-------------|
| 🔮 Predict | Enter student data and get a Pass/Fail prediction with confidence |
| 📊 Model Info | Accuracy, confusion matrix, classification report, feature coefficients |
| 📁 Dataset | Overview, sample rows, and feature distributions |

---

## 📦 Dataset Columns

| Column | Description |
|--------|-------------|
| `study_hours` | Average study hours per day |
| `attendance` | Attendance percentage (50–99%) |
| `previous_score` | Score in previous exam (40–99) |
| `pass` | 1 = Pass, 0 = Fail |
