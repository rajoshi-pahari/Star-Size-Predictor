from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

W = 1.982015  # Weight (slope)
b = 9.500380  # Bias (intercept)

@app.get('/')
def default():
    return {'App': 'Running'}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df.columns = ['Brightness', 'True Size']
    df['predictions'] = W * df['Brightness'] + b
    output = df.to_csv(index=False).encode('utf-8')
    return StreamingResponse(io.BytesIO(output),
                             media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.post("/plot/")
async def plot(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Ensure required columns exist
    if "Brightness" not in df.columns or "True Size" not in df.columns:
        return {"error": "Columns 'Brightness' and 'True Size' are required in the dataset"}
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['brightness'], df['size'], color='royalblue', label='Actual Targets', marker='x')
    df['predictions'] = W * df['brightness'] + b
    rmse_score = np.mean(np.square(df['predictions'].values - df['size'].values))
    plt.plot(df['brightness'], df['predictions'], color='k', label='Predictions', linewidth=2)
    plt.title(f'Linear Regression for Stars Data (RMSE: {round(rmse_score, 1)})', color='maroon', fontsize=15)
    plt.xlabel('Brightness', color='m', fontsize=13)
    plt.ylabel('Size', color='m', fontsize=13)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf,
                             media_type="image/png",
                             headers={"Content-Disposition": "attachment; filename=plot.png"})
    
@app.post("/create_dataset/")
def create_dataset(rows: int = 100):
    """
    Create a synthetic dataset with the specified number of rows.
    """
    import numpy as np

    # Generate synthetic data
    brightness = np.random.uniform(0, 100, rows)
    size = 1.982015 * brightness + 9.500380 + np.random.normal(0, 10, rows)  # Add some noise

    # Create DataFrame
    df = pd.DataFrame({"Brightness": brightness, "True Size": size})

    # Return CSV as a response
    output = df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(output),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=synthetic_data.csv"},
    )