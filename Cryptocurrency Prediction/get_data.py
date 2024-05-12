import argparse
from datetime import datetime
import yfinance as yf

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--crypto-name", help="Crypto name tag (example: BTC, ETH)", type=str)
parser.add_argument("-s", "--start-date", help="Start date in format year-month-day", type=str)
parser.add_argument("-p", "--save-path", help="Path to save the data parquet", type=str)
args = parser.parse_args()

# Download data
crypto_currency = args.crypto_name
print(crypto_currency)
against_currency = "USD"

start = datetime(*[int(e) for e in args.start_date.split("-")])
end = datetime.now()

data = yf.download(f"{crypto_currency}-{against_currency}", start, end)
print(type(data))
parquet_path = f"data/{args.save_path}"
data.to_parquet(parquet_path)

# Prepare data
print(data.head())