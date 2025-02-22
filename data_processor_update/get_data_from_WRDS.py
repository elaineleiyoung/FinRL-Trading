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

# Get fundamental data first
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
    
    # Get unique gvkeys from the first query results
    if not fundamental_df.empty:
        gvkeys = fundamental_df['gvkey'].unique().tolist()
        if gvkeys:
            gvkey_str = "'" + "','".join(gvkeys) + "'"
            
            # Get sector data in a separate query
            sector_query = f"""
                SELECT gvkey, gsector
                FROM comp.company
                WHERE gvkey IN ({gvkey_str});
            """
            sector_df = db.raw_sql(sector_query)
            
            # Merge the dataframes
            fundamental_df = pd.merge(fundamental_df, sector_df, on='gvkey', how='left')
    
    fundamental_df.to_csv("sp500_fundamental_199601_202502_new.csv", index=False)
else:
    fundamental_df = pd.DataFrame()
    print("No matching GVKEY, S&P 500 tic might have error")

db.close()
print(fundamental_df.head())