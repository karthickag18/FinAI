
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(
    "data/data.xls",
    header=1
)

# Default customer count
sns.countplot(
    data=df,
    x="default payment next month"
)

plt.title("Customer Default Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.show()