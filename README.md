# ner-bert-crf
使用BERT+CRF模型来完成NER任务


## NER Models
1. BiLSTM+CRF
2. Bert + CRF
3. RegexExpression-only NER Model
4. Whole-Sentence NER Model (BERT+LSTM/GRU)

---

## How to Run?
```shell
pip install -r requirements.txt
```

run main.py

---

## Project Structure
- in `model.py`, you can see there is a ModelHub that allows you to select 1 of the 4 models. you can do the selection in `main()` function from `main.py`.

![image](https://github.com/user-attachments/assets/ac416b36-0b80-44a2-89aa-3af5a084aa33)

![image](https://github.com/user-attachments/assets/31774bf1-59ba-4123-9793-a423f333b9fb)


## BERT
![image](https://github.com/user-attachments/assets/3720173b-90ae-4c3b-813a-ea0f90443f58)

![image](https://github.com/user-attachments/assets/521360a1-a08a-440f-b74d-82060360ce92)


## CRF
![image](https://github.com/user-attachments/assets/b07c5576-1c44-4a3c-9a9e-6946bb725a00)

![image](https://github.com/user-attachments/assets/db1539bc-650e-4502-8f33-8740911ea399)




## Running Results
### CRF + BiLSTM output
![image](https://github.com/user-attachments/assets/67ce9f4a-2bba-4a79-b2c2-f5b17326c5bf)




### RegexNERModel output
![image](https://github.com/user-attachments/assets/2926ef88-507d-4a78-ade6-5ac8cee16da0)


### Bert + CRF output
![image](https://github.com/user-attachments/assets/190bd57a-e7d0-4e88-b9e5-0283224a0d74)



### Whole Sentence NER output
![image](https://github.com/user-attachments/assets/80a1cb7d-450f-45e4-aef1-ccbd22bff750)



