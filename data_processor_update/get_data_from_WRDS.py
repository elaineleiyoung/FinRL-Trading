import wrds
import pandas as pd

db = wrds.Connection()

ticker_file = "sp500_tickers.txt"
with open(ticker_file, "r") as f:
    tickers = [line.strip() for line in f.readlines() if line.strip()]

ticker_str = "'" + "','".join(tickers) + "'"

permno_query = f"""
    SELECT DISTINCT permno, ticker 
    FROM crsp.msenames 
    WHERE ticker IN ({ticker_str});
"""
permno_df = db.raw_sql(permno_query)

permnos = permno_df['permno'].tolist()
permno_str = ",".join(map(str, permnos))

# if permnos:
#     price_query = f"""
#         SELECT d.*, m.ticker AS tic
#         FROM crsp.dsf d
#         LEFT JOIN crsp.msenames m ON d.permno = m.permno
#         WHERE d.date >= '1996-01-01' 
#         AND d.permno IN ({permno_str});
#     """
#     price_df = db.raw_sql(price_query)
#     price_df.to_csv("crsp_price_filtered.csv", index=False)
# else:
#     price_df = pd.DataFrame()
#     print("No matching PERMNO, S&P 500 tic might have error")

if permnos:
    fundamental_query = f"""
        SELECT f.*, s.tic
        FROM comp.fundq f
        LEFT JOIN comp.security s ON f.gvkey = s.gvkey
        WHERE f.datadate >= '1996-01-01' 
        AND s.iid = '01' 
        AND s.tic IN ({ticker_str});
    """
    fundamental_df = db.raw_sql(fundamental_query)
    fundamental_df.to_csv("sp500_fundamental_199601_202502.csv", index=False)
else:
    fundamental_df = pd.DataFrame()
    print("No matching GVKEY, S&P 500 tic might have error")

db.close()
# print(price_df.head())
print(fundamental_df.head())