import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


data = {
    'Demam_Tinggi': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    'Sakit_Kepala': [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    'Mual_Muntah':  [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'Menggigil':    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    'Nyeri_Otot':   [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    'Hasil_Malaria':[1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0] 
}

df = pd.DataFrame(data)

X = df.drop('Hasil_Malaria', axis=1) 
y = df['Hasil_Malaria']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Sedang melatih model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"Model selesai dilatih dengan akurasi: {akurasi * 100}%")

joblib.dump(model, 'model_malaria.pkl')
print("Model berhasil disimpan sebagai 'model_malaria.pkl'")