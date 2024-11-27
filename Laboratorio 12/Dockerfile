#  Imagen base de Python
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto de la API
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
