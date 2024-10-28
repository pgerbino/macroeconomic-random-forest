# Purpose: This script downloads raw datasets needed for modeling
import requests
import pandas as pd
import io

def download_dataset(url, file_name, file_type='csv', sheet_name=None):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        if file_type == 'excel':
            # Read the Excel file from the response content
            df = pd.read_excel(io.BytesIO(response.content), sheet_name=sheet_name, engine='xlrd')
        elif file_type == 'csv':
            # Read the CSV file from the response content
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        else:
            print(f"Unsupported file type: {file_type}")
            return
        
        # Save the DataFrame to a CSV file
        df.to_csv(file_name, index=False)
        print(f"Dataset downloaded and saved as '{file_name}'")
    else:
        print(f"Failed to download the dataset. Status code: {response.status_code}")

if __name__ == '__main__':
    # Define dataset URLs and details
    datasets = [
        {
            'url': "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/summaryoflabourmarketstatistics/current/a01oct2024.xls",
            'file_name': "a01_dataset.csv",
            'file_type': 'excel',
            'sheet_name': '1'
        },
        {
            'url': "https://www.ons.gov.uk/file?uri=/economy/grossdomesticproductgdp/datasets/gdpmonthlyestimateuktimeseriesdataset/current/mgdp.csv",
            'file_name': "monthly_gdp_dataset.csv",
            'file_type': 'csv'
        },
        {
            'url': "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/l55o/mm23",
            'file_name': "monthly_inflation_dataset.csv",
            'file_type': 'csv'
        },
        {
            'url': "https://www.ons.gov.uk/file?uri=/economy/governmentpublicsectorandtaxes/publicsectorfinance/datasets/publicsectorfinances/current/pusf.csv",
            'file_name' : "government_spending.csv",
            'file_type' : 'csv'
        }
    ]

    # Download each dataset
    for dataset in datasets:
        download_dataset(
            url=dataset['url'],
            file_name=dataset['file_name'],
            file_type=dataset['file_type'],
            sheet_name=dataset.get('sheet_name')
        )
