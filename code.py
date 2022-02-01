import pandas as p
import plotly.express as pe
import plotly.graph_objects as pg
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score 



df = p.read_csv("phone.csv")

colors= []

salary = df["EstimatedSalary"].tolist()
purchase = df["Purchased"].tolist()
age = df["Age"].tolist()

for i in purchase:
    if i == 1:
        colors.append("green")
    else:
        colors.append("red")


#Plot = pg.Figure(data = pg.Scatter(x=salary , y = age, mode="markers" , marker = dict(color=colors)))
#Plot.show()

factors = df[["EstimatedSalary" , "Age"]]

purchases = df["Purchased"]

salary_train , salary_test , purchase_train , purchase_test = train_test_split(factors,purchases,test_size = 0.25,random_state=0)


print(salary_train[0:10])

sc_x = StandardScaler()

salary_train = sc_x.fit_transform(salary_train)
salary_test = sc_x.transform(salary_test)

print(salary_train[0:10])


classifier = LogisticRegression(random_state = 0)
classifier.fit(salary_train , purchase_train)

purchase_pred = classifier.predict(salary_test)

print("accuracy --> " , accuracy_score(purchase_test,purchase_pred))

userAge = int(input("Enter Your Age -->"))
userSalary = int(input("Enter Your Salary -->"))

user_test = sc_x.transform([[userSalary , userAge]])

user_purchasePred = classifier.predict(user_test)

if (user_purchasePred[0] == 1):
    print("You May Purchase The Product")

else :
    print("You May Not Purchase The Product")
    
