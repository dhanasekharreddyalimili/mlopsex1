from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

df = pd.DataFrame({
    'bedrooms':[2,3,4],
    'bathrooms':[1,2,3],
    'area':[900,1200,2000],
    'age':[10,15,5],
    'price':[120000,150000,300000]
})

X = df[['bedrooms','bathrooms','area','age']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
