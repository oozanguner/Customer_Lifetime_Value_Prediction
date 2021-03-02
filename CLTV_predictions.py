
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option ('display.max_columns', 15)
pd.set_option ('display.max_rows', 10)
pd.set_option ('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel ("/Users/ozanguner/PycharmProjects/DSMLBC/Ders_Notlari/3.hafta/egzersizlerim/online_retail.xlsx",
                     sheet_name="Year 2010-2011")

df = df_.copy ()

df.head ()

df = df[df["Country"] == "United Kingdom"]

df.info ()

df.describe ([0.01, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T


def replace_with_thresholds(dataframe, col):
    quantile_range = dataframe[col].quantile (0.99) - dataframe[col].quantile (0.01)
    up_limit = dataframe[col].quantile (0.99) + 1.5 * quantile_range
    # low_limit = dataframe[col].quantile(0.01) - 1.5 * quantile_range
    dataframe.loc[dataframe[col] > up_limit, col] = up_limit


replace_with_thresholds (df, "Quantity")
replace_with_thresholds (df, "Price")

df = df[(df["Quantity"] > 0)]

df[df["Invoice"].str.startswith ("C", na=False)]

df[df["Price"] < 0]

df = df[df["Price"] > 0]

df.describe ([0.01, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

df.isnull ().sum ()

df = df.dropna ()

df.shape

df.columns

df.head ()

df["TotalPrice"] = df["Quantity"] * df["Price"]

df["InvoiceDate"].max ()
df["InvoiceDate"].min ()

today_date = dt.datetime (2011, 12, 11)

############
# RFM
############
rfm = df.groupby ("Customer ID").agg ({"InvoiceDate": [lambda date: (date.max () - date.min ()).days,
                                                       lambda date: (today_date - date.min ()).days],
                                       "Invoice": "nunique",
                                       "TotalPrice": "sum"})

rfm.columns = rfm.columns.droplevel (0)

rfm.columns = ["recency_cltv_p", "T", "frequency", "monetary"]

rfm.head ()

cltv = rfm.copy ()

# Selecting observations that have more than one frequency
cltv = cltv[cltv["frequency"] > 1]

cltv["monetary_avg"] = cltv["monetary"] / cltv["frequency"]

cltv["recency_weekly"] = cltv["recency_cltv_p"] / 7
cltv["T_weekly"] = cltv["T"] / 7

cltv["monetary_avg"][cltv["monetary_avg"] < 0].any ()
cltv[cltv["monetary_avg"]<0]

#######
# BG-NBD
######
bgf = BetaGeoFitter (penalizer_coef=0.001)
bgf.fit (cltv["frequency"], cltv["recency_weekly"], cltv["T_weekly"])

# Expected sales for 1 week
cltv["expected_number_of_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time (1,
                                                                                                cltv["frequency"],
                                                                                                cltv["recency_weekly"],
                                                                                                cltv["T_weekly"])

cltv.sort_values (by="expected_number_of_purchases", ascending=False).head ()


# Expected sales for whole company for 1 week
bgf.conditional_expected_number_of_purchases_up_to_time (4,
                                                         cltv["frequency"],
                                                         cltv["recency_weekly"],
                                                         cltv["T_weekly"]).sort_values (ascending=False).sum ()

plot_period_transactions (bgf)
plt.show ()

######
# GAMMA-GAMMA
#####
ggf = GammaGammaFitter (penalizer_coef=0.01)
ggf.fit (cltv["frequency"], cltv["monetary_avg"])

cltv["expected_average_profit"] = ggf.conditional_expected_average_profit (cltv["frequency"], cltv["monetary_avg"])

cltv.sort_values (by="expected_average_profit", ascending=False).head ()

###########
# 4. CLTV calculation with BG-NBD and GG models
###########
cltv["cltv_six_months"] = ggf.customer_lifetime_value (bgf,
                                                       cltv["frequency"],
                                                       cltv["recency_weekly"],
                                                       cltv["T_weekly"],
                                                       cltv["monetary_avg"],
                                                       time=6,
                                                       discount_rate=0.01,
                                                       freq="W")

### Best 5 customers for expected CLTV for 6 months
cltv.sort_values (by="cltv_six_months", ascending=False).head ()

cltv.describe ()

plot_cltv = cltv.quantile ([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot (x=plot_cltv.index, y=plot_cltv["cltv_six_months"], data=plot_cltv)
plt.show ()



cltv["cltv_one_month"] = ggf.customer_lifetime_value (bgf,
                                                      cltv["frequency"],
                                                      cltv["recency_weekly"],
                                                      cltv["T_weekly"],
                                                      cltv["monetary_avg"],
                                                      time=1,
                                                      discount_rate=0.01,
                                                      freq="W")

cltv["cltv_twelve_months"] = ggf.customer_lifetime_value (bgf,
                                                          cltv["frequency"],
                                                          cltv["recency_weekly"],
                                                          cltv["T_weekly"],
                                                          cltv["monetary_avg"],
                                                          time=12,
                                                          discount_rate=0.01,
                                                          freq="W")

cltv.sort_values (by="cltv_one_month", ascending=False).head (10)


cltv[["cltv_one_month", "cltv_twelve_months"]].sort_values (by="cltv_one_month", ascending=False).head (10)

cltv[["cltv_one_month", "cltv_twelve_months"]].sort_values (by="cltv_twelve_months", ascending=False).head (10)

cltv[(cltv.index == 13694) | (cltv.index == 14088)][
    ["cltv_twelve_months", "cltv_one_month", "frequency", "recency_weekly", "T_weekly", "expected_number_of_purchases",
     "expected_average_profit"]]

cltv.sort_values (by="cltv_one_month", ascending=True).head ()




cltv["segment_six_months"] = pd.qcut (cltv["cltv_six_months"], 3, labels=["C", "B", "A"])

num_cols = [col for col in cltv.columns if cltv[col].dtype != "O"]

pd.set_option ('display.max_columns', None)
pd.set_option ('display.max_rows', None)
cltv.groupby ("segment_six_months")["cltv_six_months"].describe ()
cltv.groupby ("segment_six_months")[num_cols].describe ()



cltv["top_flag"] = ""

cltv.head ()

pd.set_option ('display.max_rows', 15)
cltv_six_m = cltv.sort_values (by="cltv_six_months", ascending=False).reset_index ()

cltv_six_m["cltv_six_m_cum_total"] = cltv_six_m["cltv_six_months"].cumsum()
cltv_six_m.head()

for row, id in enumerate (cltv_six_m["Customer ID"], 1):
    customer_ratio = row / cltv_six_m.shape[0]
    if customer_ratio >= 0.2:
        break
print ("{} nolu index, {} no lu id. Customer Ratio : {}".format (row - 1, id, customer_ratio))

cltv_six_m.loc[cltv_six_m.index == 513, "Customer ID"]

cltv_six_m.loc[:513, "top_flag"] = 1
cltv_six_m.loc[514:, "top_flag"] = 0

cltv_six_m.head()


