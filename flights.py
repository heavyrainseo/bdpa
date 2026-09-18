import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 추가
import koreanize_matplotlib # 추가

flights = sns.load_dataset("flights")
flights_year = flights.groupby("year", as_index=False)["passengers"].sum()
print(flights_year.head())

plt.figure(figsize=(10, 6))
plt.plot(flights_year["year"], flights_year["passengers"], marker='o')
plt.title("연도별 항공 승객 수 추이")
plt.xlabel("연도")
plt.ylabel("승객 수")
plt.grid(True)
plt.savefig('flights_year.png')
plt.close()

flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
print(flights_pivot.head())

plt.figure(figsize=(10, 6)) # 히트맵을 위한 새로운 figure 추가
sns.heatmap(flights_pivot, annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
plt.title("월별 및 연도별 항공 승객 수 히트맵")
plt.xlabel("연도")
plt.ylabel("월")
plt.savefig('flights.png')
plt.close()

print("flights_year.png와 flights.png 파일이 생성되었습니다.") # 추가