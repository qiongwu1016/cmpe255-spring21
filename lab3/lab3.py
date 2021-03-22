import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#from sklearn.preprocessing import MinMaxScaler

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        #print(self.pima.head())
        self.X_test = None
        self.y_test = None
        

    def define_feature(self, cols):
        #print('Feature selected: ', cols)
    
        X = self.pima[cols]
        y = self.pima.label
        return X, y
    
    def train(self, cols):
        # split X and y into training and testing sets
        X, y = self.define_feature(cols)
        #scaler = MinMaxScaler()
        #scaler.fit(X)
        #X = scaler.transform(X)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        #print(['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'])
        #print(logreg.coef_)
        return logreg
    
    def predict(self, cols):
        model = self.train(cols)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    #classifer = DiabetesClassifier()
    #result = classifer.predict( ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'])






    cols = [['pregnant', 'insulin', 'bmi', 'age'], ['insulin','glucose', 'pregnant','bmi', 'pedigree','age'], ['pregnant', 'bmi', 'glucose', 'bp']]

    #classifer = DiabetesClassifier()
    #result = classifer.predict(cols[2])
    #print(f"Predicition={result}")
    #score = classifer.calculate_accuracy(result)
    #print(f"score={score}")
    #con_matrix = classifer.confusion_matrix(result)
    #print(f"confusion_matrix=${con_matrix}")
 
    

    print('| Experiement | Accuracy | Confusion Matrix | Comment |')
    print('|-------------|----------|------------------|---------|')
    print('| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |')

    for i in range(3):
        classifer = DiabetesClassifier()
        result = classifer.predict(cols[i])
        score = classifer.calculate_accuracy(result)
        con_matrix = classifer.confusion_matrix(result)
        print(f'| Solution %d| {score} | {con_matrix.tolist()} | {cols[i]} |'%(i+1))

    print('Solution 3 has the highest prediction accuracy on test dataset. ')
    print('TP = ', con_matrix[0, 0])
    print('TN = ', con_matrix[1, 1])
    print('FP = ', con_matrix[0, 1])
    print('FN = ', con_matrix[1, 0])
    print('Recall =', con_matrix[0, 0] / (con_matrix[0, 0] + con_matrix[1, 0]))
    print('Precision = ', con_matrix[0,0] / (con_matrix[0,0] + con_matrix[0, 1]))




