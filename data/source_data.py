# purpose of this file is to download the raw data need for modelling
import requests
import pandas as pd
import io

def download_a01_dataset():
    # URL for the A01 dataset
    url = "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/summaryoflabourmarketstatistics/current"

    url = "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/summaryoflabourmarketstatistics/current/a01oct2024.xls"
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Read the Excel file from the response content
        df = pd.read_excel(io.BytesIO(response.content), sheet_name="1", engine="xlrd")
        
        # Save the DataFrame to a CSV file
        df.to_csv("a01_dataset.csv", index=False)
        print("A01 dataset downloaded and saved as 'a01_dataset.csv'")
    else:
        print(f"Failed to download the dataset. Status code: {response.status_code}")

# Call the function to download the dataset
download_a01_dataset()
