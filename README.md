# SMS Spam Classifier

Project nay phan loai tin nhan thanh `spam` hoac `ham` bang cac mo hinh machine learning cho text classification.

Phan code goc van duoc giu lai de doi chieu. Phan refactor moi nam trong thu muc `spam_classifier/` va cac file chay chinh:

- `train.py`: train model bang scikit-learn Pipeline.
- `evaluate.py`: danh gia model da luu.
- `predict.py`: du doan mot tin nhan tu command line.
- `compare_models.py`: so sanh nhieu model tren cung mot cach chia du lieu.
- `web_app.py`: giao dien web nhe de test tin nhan.

## Cai Dat

```powershell
cd "D:\tài liệu\machine learning\SMS_Spam_Classifier"
& "D:\Python310\python.exe" -m pip install -r requirements.txt
```

## Train Model

Train Logistic Regression:

```powershell
& "D:\Python310\python.exe" train.py --model lr --max-features 3000
```

Train Naive Bayes:

```powershell
& "D:\Python310\python.exe" train.py --model nb --max-features 3000
```

Train Linear SVM:

```powershell
& "D:\Python310\python.exe" train.py --model svm --max-features 3000
```

Train k-NN:

```powershell
& "D:\Python310\python.exe" train.py --model knn --max-features 1000
```

Train Random Forest:

```powershell
& "D:\Python310\python.exe" train.py --model rf --max-features 3000
```

Model moi se duoc luu vao `Model/Pipeline/`. Bao cao metric se duoc luu vao `reports/`.

## Predict Tu Command Line

```powershell
& "D:\Python310\python.exe" predict.py "Win a free iPhone now" --model-path "Model/Pipeline/lr_tfidf_3000_pipeline.joblib"
```

## Chay Frontend

Sau khi train it nhat mot model:

```powershell
& "D:\Python310\python.exe" web_app.py --model-path "Model/Pipeline/lr_tfidf_3000_pipeline.joblib"
```

Mo trinh duyet tai:

```text
http://127.0.0.1:8000
```

## Chay Frontend Streamlit

Frontend Streamlit cho phep chon nhieu thuat toan da train va co che do so sanh tat ca model hien co:

```powershell
& "D:\Python310\python.exe" -m streamlit run apps/streamlit_app.py --server.port 8502
```

Mo trinh duyet tai:

```text
http://localhost:8502
```

Neu cong `8502` bi trung, doi sang cong khac:

```powershell
& "D:\Python310\python.exe" -m streamlit run apps/streamlit_app.py --server.port 8600
```

## So Sanh Model

```powershell
& "D:\Python310\python.exe" compare_models.py --models lr nb svm --max-features 3000
```

Ket qua se duoc luu tai `reports/model_comparison.csv`.

## Diem Cai Thien Chinh

- Split raw text truoc, fit TF-IDF chi tren train set de tranh data leakage.
- Dung `Pipeline` de tranh loi model va vectorizer khong khop so features.
- Dung chung preprocessing cho train va predict.
- Them confusion matrix, precision, recall, F1 cho spam.
- Them frontend web de test de hon.
- Giu lai code cu, khong thay doi muc dich chinh cua project.
