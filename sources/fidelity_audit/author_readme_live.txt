## Data

Daily return matrices are generated from Yahoo Finance / FinanceDataReader and stored as
data/daily/{market}_{YYMMDD}_{YYMMDD}.csv (rows = dates, columns = assets, values =
simple returns). Data files are not distributed with the code.

```
python src/data_generator.py --data euro50    --start 2016-01-01 --end 2025-12-31
python src/data_generator.py --data ftse100   --start 2016-01-01 --end 2025-12-31
python src/data_generator.py --data nikkei225 --start 2016-01-01 --end 2025-12-31
python src/data_generator.py --data kospi200  --start 2016-01-01 --end 2025-12-31
```

kospi200 needs the KRX constituent list saved as data/raw/kospi200.csv (CP949 encoded,
with a 6-digit code column named 종목코드, as exported from the KRX data portal).
Constituents are taken as of the download date; the paper restricts attention to assets
with complete price histories over the sample period, which is what the generator's
dropna(axis="columns") does.

The default file suffix used by every script is 160101_251231; override with
--data_suffix if you generate a different period.

Note that index constituents and Yahoo's adjusted prices change over time, so data
downloaded today differs from the snapshot used in the paper and the reported numbers
will not be reproduced exactly.
