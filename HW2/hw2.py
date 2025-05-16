
import model_interface_hw2 as ml
import matplotlib.pyplot as plt
import visualization_interface_hw2 as vs
import add_none_values_interface as non
import fill_gaps_interface as fg


file_path = r'C:\Users\Кристина\Desktop\coffee_shop_revenue.csv'
df = ml.load_data(file_path)

if df.empty:
    print('DataFrame is empty!')

print(df.head())
print(df.info())


a = df['Number_of_Customers_Per_Day']
b = df['Average_Order_Value']
c = df['Operating_Hours_Per_Day']
d = df['Number_of_Employees']
e = df['Marketing_Spend_Per_Day']
f = df['Location_Foot_Traffic']
target_column = 'Daily_Revenue'

vs.visual_histogram(a,25,"green")
vs.visual_histogram(b,20,"blue")
vs.visual_histogram(c,10,"orange")
vs.visual_histogram(d,10,"magenta")
vs.visual_histogram(e,25,"pink")
vs.visual_histogram(f,20,"cyan")

vs.visual_kde(df,target_column)

vs.visual_distribution(df,a.name,target_column)
vs.visual_distribution(df,b.name,target_column)
vs.visual_distribution(df,c.name,target_column)
vs.visual_distribution(df,f.name,target_column)
vs.visual_distribution(df,f.name,a.name)

vs.visual_fancy_scatterplot(df,b.name,f.name,a.name,target_column)

vs.visual_boxplot(df,target_column,"grey")
vs.visual_boxplot(df,a.name,"green")
vs.visual_boxplot(df,b.name,"blue")
vs.visual_boxplot(df,c.name,"orange")
vs.visual_boxplot(df,d.name,"magenta")
vs.visual_boxplot(df,e.name,"pink")
vs.visual_boxplot(df,f.name,"cyan")



mse, r2,y_pred,y_test = ml.get_result(df,target_column)
print(f"------------------------------------------------------Среднеквадратичная ошибка: {mse:.2f}")
print(f"------------------------------------------------------Коэффициент детерминации R^2: {r2:.2f}")

vs.visual_predictions(y_test,y_pred,50,'Индекс','Значение выручки','Истинные и предсказанные значения выручки за день ')


df = non.add_none_values(df,a)
vs.visual_kde(df,a.name)
fg.fill_gaps_median(df,a)
mse, r2,y_pred,y_test = ml.get_result(df,target_column)
print(f"Среднеквадратичная ошибка на данных с заполненными медианой пропущенными значениями: {mse:.2f}")
print(f"Коэффициент детерминации R^2 на данных с заполненными медианой пропущенными значениями: {r2:.2f}")

vs.visual_predictions(y_test,y_pred,50,'Индекс','Значение выручки','Истинные и предсказанные значения выручки за день(с заполненными медианой пропущенными значениями)  ')


df = ml.load_data(file_path)
print(df.head())
print(df.info())


df = non.add_none_values(df,a)
vs.visual_kde(df,a.name)
fg.fill_gaps_mean(df,a)
mse, r2,y_pred,y_test = ml.get_result(df,target_column)
print(f"Среднеквадратичная ошибка на данных с заполненными средним пропущенными значениями: {mse:.2f}")
print(f"Коэффициент детерминации R^2 на данных с заполненными средним пропущенными значениями: {r2:.2f}")

vs.visual_predictions(y_test,y_pred,50,'Индекс','Значение выручки','Истинные и предсказанные значения выручки за день(с заполненными средним пропущенными значениями)  ')

df = non.add_none_values(df,a)
vs.visual_kde(df,a.name)
fg.fill_gaps_mode(df,a)
mse, r2,y_pred,y_test = ml.get_result(df,target_column)
print(f"Среднеквадратичная ошибка на данных с заполненными наиболее часто встречающимся пропущенными значениями: {mse:.2f}")
print(f"Коэффициент детерминации R^2 на данных с заполненными наиболее часто встречающимся пропущенными значениями: {r2:.2f}")

vs.visual_predictions(y_test,y_pred,50,'Индекс','Значение выручки','Истинные и предсказанные значения выручки за день(с заполненными наиболее часто встречающимся пропущенными значениями)  ')

