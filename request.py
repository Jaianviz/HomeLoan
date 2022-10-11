import requests

url = 'http://localhost:5000/predict' 
r = requests.post(url,json={'Gender':0, 'Married':0, 'Dependents':2, 'Education':0, 'Self_Employed':1,'ApplicantIncome':7000,
                                'CoapplicantIncome':8000,'LoanAmount':9000,'Loan_Amount_Term':360,'Credit_History':1,'Property_Area':2})

print(r.json())


