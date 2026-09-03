import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,f1_score,recall_score,precision_score
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from ast import mod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

url="https://drive.usercontent.google.com/download?id=1XVm_K-RowGhVp3l7y84TiuEtlwd3gvgN&export=download&authuser=0&confirm=t"
df=pd.read_csv(url)
df.head()

# Data Checking
print("=== Dataset Information ===")
print(df.info())
print("\n=== Missing Values ===")
print(df.isnull().sum())
print("\n=== Duplicate Rows ===")
print(df.duplicated().sum())
print("\n=== Descriptive Statistics ===")
print(df.describe())
print(df['Academic_Failure_Risk'].value_counts())
df.groupby(['Burnout_Level','Academic_Failure_Risk']).size()

#data preprocessing
y=df['Academic_Failure_Risk']
x=df.drop(['Academic_Failure_Risk','Student_ID','Gender','Education_Level','Physical_Health_Score','Physical_Activity_Hours','Daily_AI_Tool_Usage_Hours','Age','Social_Isolation_Score'],axis=1)
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.20, random_state=42,stratify=y)
cat_cols = x_train.select_dtypes(include=['object','category']).columns
encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
x_train[cat_cols]= encoder.fit_transform(x_train[cat_cols])
x_test[cat_cols]= encoder.transform(x_test[cat_cols])
st=StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.transform(x_test)
sm=SMOTE(random_state=42,)
x_train,y_train=sm.fit_resample(x_train,y_train)

#random forest model
RFmodel=RandomForestClassifier(n_estimators=150,random_state=42,max_depth=10,n_jobs=-1)
RFmodel.fit(x_train,y_train)
RFpredection=RFmodel.predict(x_test)
RFaccuracy=accuracy_score(y_test,RFpredection)
RFcr=classification_report(y_test,RFpredection)
RFcm=confusion_matrix(y_test,RFpredection)
RFf1=f1_score(y_test,RFpredection)
RFrecall=recall_score(y_test,RFpredection)
RFprecision=precision_score(y_test,RFpredection)
print(f"accuracy of random forest :{RFaccuracy*100}%")
print(f"classification report of random forest :\n {RFcr}")
print(" Random Forest Confusion Matrix Display:")
RFdisp=ConfusionMatrixDisplay(confusion_matrix=RFcm,display_labels=['No Academic Failure Risk','Academic Failure Risk'])
RFdisp.plot(cmap='Purples')
plt.title('Confusion Matrix of Random Forest')
plt.show()

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
# Model Training
LogisticRegression_model=LogisticRegression()
LogisticRegression_model.fit(x_train,y_train)
# Model Predictions
LogisticRegression_prediction=LogisticRegression_model.predict(x_test)
# Evaluation:-
from sklearn import metrics
print(f"Accuracy = :{metrics.accuracy_score(y_test,LogisticRegression_prediction)*100:.4f}% \n")
print(f"Precision = :{metrics.precision_score(y_test,LogisticRegression_prediction)*100:.4f}% \n")
print(f"Recall = :{metrics.recall_score(y_test,LogisticRegression_prediction)*100:.4f}% \n")
print(f"F1-Score = :{metrics.f1_score(y_test,LogisticRegression_prediction)*100:.4f}% \n")
# Classification Report
print("Logistic Regression Classification Report:\n")
LR_classification_report = metrics.classification_report(y_test,LogisticRegression_prediction)
print(LR_classification_report)
# Confusion Matrix
print(" Logistic Regression Confusion Matrix:\n")
labels = ['No Academic Failure Risk','Academic Failure Risk']
cm = confusion_matrix(y_test,LogisticRegression_prediction )
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
display.plot(cmap=plt.cm.Greens)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

#Gaussian Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
from ast import mod
#Train Gaussian Naive Bayes
model_NB=GaussianNB()
model_NB.fit(x_train,y_train)
#Make predictions
pred_NB=model_NB.predict(x_test)
#Evaluate model
print("Accuracy of Gaussian Naive Bayes:\n")
print(accuracy_score(y_test,pred_NB))
print("\nClassification Report of Gaussian Naive Bayes:\n")
print(classification_report(y_test,pred_NB))
print("\nRecall Score of Gaussian Naive Bayes:\n")
print(recall_score(y_test,pred_NB))
print("\nPrecision Score of Gaussian Naive Bayes:\n")
print(precision_score(y_test,pred_NB))
print("\nF1 Score of Gaussian Naive Bayes:\n")
print(f1_score(y_test,pred_NB))
#confusion matrix
NB_labels = ["No Academic Failure Risk","Academic Failure Risk"]
cm_NB = confusion_matrix(y_test,pred_NB,labels=[0, 1])
ConfusionMatrixDisplay(cm_NB,display_labels=NB_labels).plot(cmap='Reds')
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.show()

#KNeighbors model
k_values = range(1, 21)
accuracy_values = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    accuracy_values.append(accuracy)
for k, accuracy in zip(k_values, accuracy_values):
    print(f"K = {k}, Accuracy = {accuracy:.4f}")
best_k = k_values[np.argmax(accuracy_values)]
best_accuracy = max(accuracy_values)
print(f"\nBest K: {best_k}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
plt.plot(k_values, accuracy_values, marker='o')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different K Values')
plt.xticks(k_values)
plt.show()
KNNmodel = KNeighborsClassifier(n_neighbors=best_k)
KNNmodel.fit(x_train, y_train)
KNNprediction = KNNmodel.predict(x_test)
KNNaccuracy = accuracy_score(y_test, KNNprediction)
KNNcr = classification_report(y_test, KNNprediction)
KNNcm = confusion_matrix(y_test, KNNprediction)
print(f"KNN Accuracy: {KNNaccuracy * 100:.2f}%")
print(f"\nKNN Classification Report:\n{KNNcr}")
KNNdisp = ConfusionMatrixDisplay(
    confusion_matrix=KNNcm,
    display_labels=[
        'No Academic Failure Risk',
        'Academic Failure Risk'])
KNNdisp.plot(cmap='Purples')
plt.title('Confusion Matrix of KNN')
plt.show()

# Decision Tree model
DTmodel=DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42)
DTmodel.fit(x_train,y_train)
DTpredection=DTmodel.predict(x_test)
DTaccuracy=accuracy_score(y_test,DTpredection)
DTcr=classification_report(y_test,DTpredection)
DTcm=confusion_matrix(y_test,DTpredection)
DTf1=f1_score(y_test,DTpredection)
DTrecall=recall_score(y_test,DTpredection)
DTprecision=precision_score(y_test,DTpredection)
print(f"accuracy of decision tree :{DTaccuracy*100}%")
print(f"classification report of decision tree:\n {DTcr} ")
print(" Decision Tree Confusion Matrix Display:")
DTdisp=ConfusionMatrixDisplay(
    confusion_matrix=DTcm,
    display_labels=['No Academic Failure Risk','Academic Failure Risk'])
DTdisp.plot(cmap='Oranges')
plt.title('Confusion Matrix of Decision Tree')
plt.show()
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1}

#Comparison
RFmodel_results = evaluate_model("Random Forest", y_test, RFpredection)
LogisticRegression_model_results = evaluate_model("Logistic Regression", y_test, LogisticRegression_prediction)
model_NB_results = evaluate_model("Naive bayes", y_test, pred_NB)
KNNmodel_results = evaluate_model("KNN", y_test, KNNprediction )
DTmodel_results = evaluate_model("Decision Tree", y_test, DTpredection)
results_data = pd.DataFrame([RFmodel_results, LogisticRegression_model_results, model_NB_results,DTmodel_results,KNNmodel_results])
results_data.set_index('Model', inplace=True)
results_data
#Best model
best_model = results_data['F1-score'].idxmax()
print("The Best Model is:")
print(best_model)
print(results_data.loc[best_model])
best_model=RFmodel

#Heat Map
plt.figure(figsize=(8, 5))
sns.heatmap(results_data, annot=True, cmap='Blues', fmt='.4f', linewidths=0.5)
plt.title('Model Performance Heat Map')
plt.show()

#saving the trained best model
joblib.dump(best_model,'best_model.pkl')
joblib.dump(encoder, "encoder.pkl")
joblib.dump(st,"scaler.pkl")
