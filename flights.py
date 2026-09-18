import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset("flights")
flights_year = flights.groupby("year", as_index=False)["passengers"].sum()
print(flights_year.head())
plt.plot(flights_year["year"], flights_year["passengers"])
plt.savefig('flights_year.png') #plt.show()

flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
print(flights_pivot.head())

sns.heatmap(flights_pivot, annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
plt.savefig('flights.png') #plt.show()