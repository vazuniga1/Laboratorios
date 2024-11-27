from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from utils import WaterPotability
import pickle
import os


# inicializamos API
app = FastAPI()

# Se define ruta del mejor modelo y se carga
PATH_BEST_MODEL = "models/best_model.pkl"
with open(PATH_BEST_MODEL, 'rb') as f:
    best_model = pickle.load(f)

# Defina GET con ruta tipo home que describa brevemente su modelo, \
# el problema que intenta resolver, su entrada y salida.
@app.get('/', response_class=HTMLResponse) # ruta
async def home(): 
    html_response = """ 
   <html>
    <h1>Modelo de Machine Learning para predecir la calidad del agua</h1>
    <p>Este modelo de Machine Learning se encarga de predecir la calidad del agua en base a diferentes features.</p>

    <body>
        <p> Ejemplo de entrada del modelo: </p>
    
    <p style="text-align: center;">     
            "ph":10.316400384553162, <br>
            "Hardness":217.2668424334475, <br>
            "Solids":10676.508475429378, <br>
            "Chloramines":3.445514571005745, <br>
            "Sulfate":397.7549459751925, <br>
            "Conductivity":492.20647361771086, <br>
            "Organic_carbon":12.812732207582542, <br>
            "Trihalomethanes":72.28192021570328, <br>
            "Turbidity":3.4073494284238364 <br>
    </p>

    <p> Ejemplo de salida del modelo: </p>
    <p style="text-align: center;"> "potabilidad": 0 </p>
     </body>    
    </html>
""" 
    return html_response


@app.post('/predict') # ruta
async def predict(data: WaterPotability):
    features = [data.ph, data.Hardness, data.Solids, data.Chloramines, data.Sulfate, \
        data.Conductivity, data.Organic_carbon, data.Trihalomethanes, data.Turbidity]

    # se realiza la predicción
    prediction = best_model.predict([features])
    
    html_content = f"""
    <html>
    <h1>Resultado de la predicción</h1>
    <p>El modelo predice que la calidad del agua es: {prediction[0]}</p>
    </html>
    """

    return HTMLResponse(content=html_content, status_code=200)
